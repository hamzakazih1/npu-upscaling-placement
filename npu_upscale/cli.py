"""
npu-upscale: command-line front end.

    npu-upscale doctor                       what is this machine, does the NPU work?
    npu-upscale image in.png -o out.png      upscale one or more images
    npu-upscale video in.mp4 -o out.mp4      upscale a video (needs OpenCV)
    npu-upscale bench --size 960x540         frame latency against 30/60 fps budgets
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _add_device_args(parser: argparse.ArgumentParser, mode: str = "burst") -> None:
    parser.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto",
                        help="auto: NPU if it works, otherwise CPU (default)")
    parser.add_argument("--model", type=Path, help="custom QDQ INT8 ONNX model")
    parser.add_argument("--performance-mode", default=mode,
                        help=f"QNN HTP performance mode (default {mode})")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="let individual nodes fall back to CPU (not recommended)")
    parser.add_argument("--no-cache", action="store_true",
                        help="do not cache the compiled NPU context")


def _make_upscaler(args, frame_size=None):
    from .upscaler import Upscaler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upscaler = Upscaler(args.model, device=args.device,
                            strict=not args.allow_cpu_fallback,
                            performance_mode=args.performance_mode,
                            cache=not args.no_cache, frame_size=frame_size)
    print(f"running on: {upscaler.info.describe()}", file=sys.stderr)
    for note in upscaler.info.notes:
        print(f"  note: {note}", file=sys.stderr)
    return upscaler


def _parse_size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


# --------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    import onnxruntime as ort

    from . import runtime
    from .chips import detect_chip, emulation_warning
    from .placement import verify_placement
    from .upscaler import DEFAULT_NPU_MODEL, bundled_model

    chip = detect_chip()
    ok = True
    print("System")
    print(f"  OS              {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Python          {platform.python_version()} {platform.machine()}")
    print(f"  Chip            {chip.describe()}")
    print(f"                  identified from: {chip.source}")
    warning = emulation_warning(chip)
    if warning:
        print(f"  !! {warning}")
        ok = False

    print("\nONNX Runtime")
    print(f"  onnxruntime     {ort.__version__}")
    print(f"  onnxruntime-qnn {'installed at ' + runtime.qnn_package_dir() if runtime.qnn_package_dir() else 'not installed as a plugin package'}")
    style = runtime.qnn_style()
    print(f"  QNN             {style + ' execution provider' if style else 'NOT AVAILABLE'}")
    try:
        for device in ort.get_ep_devices():
            print(f"    device        {device.ep_name:<24} {device.device.type}")
    except AttributeError:
        pass

    if style is None:
        print("\n  The NPU cannot be used from this environment.")
        if chip.is_snapdragon_x:
            print("  Fix: ARM64 Python, then `pip install onnxruntime-qnn`, and keep the")
            print("  Qualcomm Hexagon NPU driver current through Windows Update.")
        return 1

    model = args.model or bundled_model(DEFAULT_NPU_MODEL)
    print(f"\nPlacement check ({Path(model).name})")
    try:
        report = verify_placement(model, device="npu", strict=False)
    except Exception as error:
        print(f"  !! NPU session failed: {error}")
        return 1
    for provider, count in sorted(report["providers"].items()):
        print(f"  {provider:<28} {count} node executions")
    print(f"  max |NPU - CPU| {report['max_abs_diff_vs_cpu']:.3e}")
    if report["cpu_ops"]:
        print(f"  !! fell back to CPU: {', '.join(report['cpu_ops'])}")
        ok = False
    else:
        print("  every node ran on the NPU")
    if report["max_abs_diff_vs_cpu"] > 0.05:
        print("  !! NPU output deviates from the CPU reference; investigate")
        ok = False

    print("\nLatency (bundled model, one 270x480 tile)")
    from .upscaler import Upscaler
    upscaler = Upscaler(args.model, device="npu")
    result = upscaler.benchmark(upscaler.tile_h or 270, upscaler.tile_w or 480, seconds=3)
    print(f"  {result['median_ms']} ms median, {result['budget_60fps']}% of a 60 fps frame")
    print(f"  {upscaler.info.describe()}")

    print("\nResult:", "READY" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def cmd_image(args) -> int:
    from PIL import Image
    import numpy as np

    inputs = []
    for item in args.inputs:
        path = Path(item)
        if path.is_dir():
            inputs.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES))
        else:
            inputs.append(path)
    if not inputs:
        print("no images found", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else None
    many = len(inputs) > 1 or (output is not None and output.is_dir())
    if many and output is not None and output.suffix:
        print("with several inputs, -o must be a directory", file=sys.stderr)
        return 1

    upscaler = _make_upscaler(args)
    for path in inputs:
        if many or output is None:
            directory = output or path.parent
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{path.stem}_x{upscaler.scale}{path.suffix}"
        else:
            target = output
        image = Image.open(path)
        mode = image.mode if image.mode in ("RGB", "RGBA", "L") else "RGB"
        array = np.asarray(image.convert(mode))
        start = time.perf_counter()
        result = upscaler.upscale(array)
        elapsed = (time.perf_counter() - start) * 1000
        Image.fromarray(result).save(target)
        print(f"{path} -> {target}  {array.shape[1]}x{array.shape[0]} -> "
              f"{result.shape[1]}x{result.shape[0]}  {elapsed:.1f} ms")
    return 0


def cmd_video(args) -> int:
    from .video import read_frames, upscale_video

    frame_size = None
    if not args.tiled:
        _, meta = read_frames(args.input)
        frame_size = (meta["width"], meta["height"])
    upscaler = _make_upscaler(args, frame_size)
    target = Path(args.output) if args.output else Path(args.input).with_name(
        f"{Path(args.input).stem}_x{upscaler.scale}.mp4")

    def progress(done, total, elapsed):
        if done % 10 == 0 or done == total:
            rate = done / elapsed if elapsed else 0
            of = f"/{total}" if total > 0 else ""
            print(f"\r  frame {done}{of}  {rate:.1f} fps", end="", file=sys.stderr)

    result = upscale_video(args.input, target, upscaler, codec=args.codec,
                           workers=args.workers, progress=progress)
    print(file=sys.stderr)
    print(f"{args.input} -> {target}  {result['frames']} frames at {result['output']}, "
          f"{result['fps']} fps, audio {result['audio']}")
    return 0


def cmd_bench(args) -> int:
    rows = []
    upscaler = None if args.native else _make_upscaler(args)
    for size in args.size:
        width, height = _parse_size(size)
        if args.native:
            upscaler = _make_upscaler(args, (width, height))
        rows.append(upscaler.benchmark(height, width, seconds=args.seconds))
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"\n{'input':>10} {'output':>10} {'tiles':>5} {'median ms':>10} "
          f"{'p95 ms':>8} {'fps':>7} {'30fps':>7} {'60fps':>7}")
    for r in rows:
        print(f"{r['input']:>10} {r['output']:>10} {r['tiles']:>5} {r['median_ms']:>10} "
              f"{r['p95_ms']:>8} {r['fps']:>7} {r['budget_30fps']:>6}% {r['budget_60fps']:>6}%")
    print("\n30fps/60fps columns: share of the frame budget one upscale consumes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="npu-upscale", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="diagnose NPU support on this machine")
    doctor.add_argument("--model", type=Path)
    doctor.set_defaults(func=cmd_doctor)

    image = sub.add_parser("image", help="upscale images or folders of images")
    image.add_argument("inputs", nargs="+")
    image.add_argument("-o", "--output", help="output file, or directory for several inputs")
    _add_device_args(image)
    image.set_defaults(func=cmd_image)

    video = sub.add_parser("video", help="upscale a video file")
    video.add_argument("input")
    video.add_argument("-o", "--output")
    video.add_argument("--codec", default="mp4v")
    video.add_argument("--workers", type=int, default=2)
    video.add_argument("--tiled", action="store_true",
                       help="use the 270x480 tiles instead of a model sized to the video")
    _add_device_args(video, mode="sustained_high_performance")
    video.set_defaults(func=cmd_video)

    bench = sub.add_parser("bench", help="frame latency against frame budgets")
    bench.add_argument("--size", nargs="+", default=["480x270", "960x540", "1280x720", "1920x1080"],
                       help="input sizes, WIDTHxHEIGHT")
    bench.add_argument("--seconds", type=float, default=5.0)
    bench.add_argument("--json", action="store_true")
    bench.add_argument("--native", action="store_true",
                       help="re-declare the model at each size instead of tiling")
    _add_device_args(bench)
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as error:  # user-facing tool: one clear line, not a trace
        from .runtime import PlacementError
        if isinstance(error, (PlacementError, RuntimeError, FileNotFoundError)):
            print(f"error: {error}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
