"""
The upscaler: RGB frames in, RGB frames at twice the resolution out.

    from npu_upscale import Upscaler
    up = Upscaler()                     # NPU if present, CPU otherwise
    print(up.info.describe())           # always check where it really runs
    big = up.upscale(frame)             # HxWx3 uint8 -> 2Hx2Wx3 uint8
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .runtime import SessionInfo, create_session, reshape_model
from .tiling import tile_count, upscale_tiled

# The corrected model from notebook 02: +0.89 dB over bicubic in INT8.
DEFAULT_NPU_MODEL = "espcn_x2_v2_int8.onnx"
# The CPU gets the fp32 export: better quality, and no reason to quantize.
DEFAULT_CPU_MODEL = "espcn_x2_v2.onnx"


def bundled_model(name: str) -> Path:
    return Path(str(resources.files("npu_upscale") / "models" / name))


class Upscaler:
    """
    A neural x2 upscaler bound to one device.

    device            "auto" (NPU, else CPU with a warning), "npu" (NPU or
                      raise) or "cpu".
    strict            refuse NPU sessions in which any node would fall back to
                      the CPU. Leave on; it is what makes "npu" mean NPU.
    performance_mode  QNN HTP mode. "burst" for short jobs;
                      "sustained_high_performance" for long video runs on a
                      thin-and-light where burst clocks throttle.
    margin            tile overlap in input pixels. 8 covers the bundled
                      model's 4-pixel receptive-field radius with room to spare.
    frame_size        (width, height). Re-declare the model at exactly this
                      input size and run each frame in one pass instead of
                      tiles. Use it whenever every frame has the same size
                      (video, a game's render target). Other sizes still work,
                      through tiling.
    """

    def __init__(self, model: str | os.PathLike | None = None, *,
                 cpu_model: str | os.PathLike | None = None,
                 device: str = "auto", strict: bool = True,
                 performance_mode: str = "burst", margin: int = 8,
                 cache: bool = True, frame_size: tuple[int, int] | None = None):
        npu_model = Path(model) if model else bundled_model(DEFAULT_NPU_MODEL)
        if cpu_model is None:
            cpu_model = npu_model if model else bundled_model(DEFAULT_CPU_MODEL)
        cpu_model = Path(cpu_model)
        if frame_size is not None:
            width, height = frame_size
            npu_model = reshape_model(npu_model, height, width)
            cpu_model = reshape_model(cpu_model, height, width)

        self.session, self.info = create_session(
            npu_model, cpu_model, device=device, strict=strict,
            performance_mode=performance_mode, cache=cache)
        self.margin = margin

        source = self.session.get_inputs()[0]
        self.input_name = source.name
        in_shape = source.shape
        out_shape = self.session.get_outputs()[0].shape
        # Symbolic dimensions mean the model accepts any size: no tiling needed.
        self.dynamic = not all(isinstance(d, int) for d in in_shape[2:])
        if self.dynamic:
            self.tile_h = self.tile_w = None
            self.scale = 2
            if all(isinstance(d, int) for d in (in_shape[2], out_shape[2])):
                self.scale = out_shape[2] // in_shape[2]
        else:
            self.tile_h, self.tile_w = in_shape[2], in_shape[3]
            self.scale = out_shape[2] // in_shape[2]

    # -- core ---------------------------------------------------------------

    def _run(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]

    def upscale_chw(self, image: np.ndarray) -> np.ndarray:
        """CHW float32 in [0, 1] -> CHW float32, clipped to [0, 1]."""
        image = np.ascontiguousarray(image, dtype=np.float32)
        if self.dynamic:
            out = self._run(image[None])[0]
        else:
            out = upscale_tiled(image, self._run, self.tile_h, self.tile_w,
                                self.scale, self.margin)
        return np.clip(out, 0.0, 1.0, out=out)

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        HxW, HxWx3 or HxWx4 uint8 (RGB order) -> the same layout, upscaled.

        Alpha, if present, is upscaled by pixel replication: the network was
        trained on colour and has no business inventing transparency.
        """
        if image.dtype != np.uint8:
            raise TypeError(f"expected uint8 image, got {image.dtype}")
        gray = image.ndim == 2
        rgb = np.repeat(image[..., None], 3, axis=2) if gray else image
        alpha = None
        if rgb.shape[2] == 4:
            rgb, alpha = rgb[..., :3], rgb[..., 3]
        elif rgb.shape[2] != 3:
            raise ValueError(f"unsupported channel count {rgb.shape[2]}")

        chw = rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        out = self.upscale_chw(chw)
        result = (out * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)

        if gray:
            return np.ascontiguousarray(result[..., 0])
        if alpha is not None:
            big_alpha = alpha.repeat(self.scale, 0).repeat(self.scale, 1)
            result = np.concatenate([result, big_alpha[..., None]], axis=2)
        return np.ascontiguousarray(result)

    __call__ = upscale

    def stream(self, frames: Iterable[np.ndarray], workers: int = 2,
               prefetch: int = 4) -> Iterator[np.ndarray]:
        """
        Upscale a sequence of frames, in order, overlapping CPU-side colour
        conversion with NPU execution. ONNX Runtime sessions are thread-safe
        and release the GIL while running.
        """
        if workers <= 1:
            for frame in frames:
                yield self.upscale(frame)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = []
            for frame in frames:
                pending.append(pool.submit(self.upscale, frame))
                if len(pending) >= prefetch:
                    yield pending.pop(0).result()
            for future in pending:
                yield future.result()

    # -- measurement --------------------------------------------------------

    def tiles_for(self, height: int, width: int) -> int:
        if self.dynamic:
            return 1
        return tile_count(height, width, self.tile_h, self.tile_w, self.margin)

    def benchmark(self, height: int = 540, width: int = 960, seconds: float = 10.0,
                  warmup: int = 5) -> dict:
        """
        Frame latency at one input resolution, reported against frame budgets
        the way the study reported the 270x480 tile.
        """
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(warmup):
            self.upscale(frame)
        times = []
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline or len(times) < 3:
            start = time.perf_counter()
            self.upscale(frame)
            times.append((time.perf_counter() - start) * 1000)
        times = np.array(times)
        median = float(np.median(times))
        return {
            "device": self.info.device,
            "chip": self.info.chip.model or self.info.chip.name,
            "input": f"{width}x{height}",
            "output": f"{width * self.scale}x{height * self.scale}",
            "tiles": self.tiles_for(height, width),
            "frames": int(len(times)),
            "median_ms": round(median, 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "fps": round(1000.0 / median, 2),
            "budget_30fps": round(median / (1000 / 30) * 100, 1),
            "budget_60fps": round(median / (1000 / 60) * 100, 1),
        }

    def __repr__(self) -> str:
        return f"Upscaler(x{self.scale}, {self.info.describe()})"
