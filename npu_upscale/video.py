"""
Video files: decode, upscale on the NPU, encode, and carry the audio across.

Needs OpenCV (pip install "npu-upscale[video]"). Audio is copied from the
source with ffmpeg when ffmpeg is on PATH; otherwise the output is silent and
the function says so.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .upscaler import Upscaler


def _cv2():
    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise RuntimeError('video support needs OpenCV: pip install "npu-upscale[video]"') from error
    return cv2


def read_frames(path: str | Path) -> tuple[Iterator[np.ndarray], dict]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    meta = {
        "fps": capture.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    def frames():
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    return frames(), meta


def upscale_video(source: str | Path, target: str | Path, upscaler: Upscaler,
                  codec: str = "mp4v", workers: int = 2,
                  progress: Callable[[int, int, float], None] | None = None) -> dict:
    cv2 = _cv2()
    source, target = Path(source), Path(target)
    frames, meta = read_frames(source)
    size = (meta["width"] * upscaler.scale, meta["height"] * upscaler.scale)

    temp_dir = Path(tempfile.mkdtemp(prefix="npu_upscale_"))
    silent = temp_dir / f"video{target.suffix or '.mp4'}"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*codec), meta["fps"], size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for codec {codec!r}")

    count, start = 0, time.perf_counter()
    try:
        for frame in upscaler.stream(frames, workers=workers):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            count += 1
            if progress:
                progress(count, meta["frames"], time.perf_counter() - start)
    finally:
        writer.release()
    elapsed = time.perf_counter() - start

    audio = "none"
    ffmpeg = shutil.which("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(source),
                   "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy",
                   "-shortest", str(target)]
        if subprocess.run(command).returncode == 0:
            audio = "copied"
    if audio != "copied":
        shutil.move(str(silent), target)
        audio = "dropped (ffmpeg not found)" if not ffmpeg else "dropped (ffmpeg failed)"
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {"frames": count, "seconds": round(elapsed, 2),
            "fps": round(count / elapsed, 2) if elapsed else None,
            "output": f"{size[0]}x{size[1]}", "audio": audio}
