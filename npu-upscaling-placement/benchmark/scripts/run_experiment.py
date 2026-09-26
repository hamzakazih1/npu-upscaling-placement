"""
The experiment: does running the upscaler on the NPU leave the GPU more headroom?

Design, and why it is shaped this way.

ONNX Runtime has no working GPU path on this machine: DirectML has no ARM64
build, and QNN's Adreno backend rejects quantized convolutions at validation and
crashes at runtime on fp32 ones. So the GPU workload cannot be an ONNX model.

Instead the GPU is loaded by a real WebGL scene running in the browser, which
reaches the Adreno through the graphics driver -- a completely different path
that demonstrably works. The page measures its own frame rate and posts it here.
That is closer to the real situation than an inference proxy would have been: an
actual renderer competing for an actual GPU.

Conditions:

  baseline    GPU renders, nothing else running
  cpu         GPU renders + upscaler on CPU
  npu         GPU renders + upscaler on the Hexagon NPU

The measured quantity is the renderer's frame rate. If offloading to the NPU
helps, npu should hold closer to baseline than cpu does.

Usage:
    python run_experiment.py --upscaler espcn_x2_int8.onnx
    python run_experiment.py --upscaler espcn_x2_int8.onnx --duration 30 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort

from qnn_setup import ensure_qnn, make_qnn_session, qnn_devices

SAMPLES: list[dict] = []
SAMPLES_LOCK = threading.Lock()


class FpsHandler(BaseHTTPRequestHandler):
    """Receives frame-rate samples from the WebGL page."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            payload["received"] = time.time()
            with SAMPLES_LOCK:
                SAMPLES.append(payload)
        except Exception:
            pass
        self._respond()

    def do_OPTIONS(self):
        self._respond()

    def do_GET(self):
        self._respond(body=b'{"ok":true}')

    def _respond(self, body: bytes = b"{}"):
        self.send_response(200)
        # The page is opened from file:// or localhost, so it is cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # the default handler prints every request


class UpscalerLoop(threading.Thread):
    """Runs the upscaler continuously on one backend until stopped."""

    def __init__(self, model: Path, backend: str, shape):
        super().__init__(daemon=True)
        self.model = model
        self.backend = backend
        self.shape = shape
        self.stop_flag = threading.Event()
        self.iterations = 0
        self.latencies: list[float] = []
        self.error: str | None = None

    def run(self):
        try:
            if self.backend == "npu":
                session = make_qnn_session(self.model, device="npu")
            else:
                session = ort.InferenceSession(
                    str(self.model), providers=["CPUExecutionProvider"])

            name = session.get_inputs()[0].name
            data = np.random.randn(*self.shape).astype(np.float32)

            for _ in range(3):  # warmup, excluded from counts
                session.run(None, {name: data})

            while not self.stop_flag.is_set():
                start = time.perf_counter()
                session.run(None, {name: data})
                self.latencies.append((time.perf_counter() - start) * 1000)
                self.iterations += 1
        except Exception as exc:
            self.error = str(exc)


def samples_between(start: float, end: float) -> list[float]:
    """Frame-rate samples inside a time window, ignoring the calibration phase."""
    with SAMPLES_LOCK:
        rows = list(SAMPLES)
    return [r["mean_fps"] for r in rows
            if start <= r["received"] <= end
            and not r.get("calibrating", False)
            and r.get("mean_fps", 0) > 0]


def run_condition(name: str, model: Path, backend: str | None,
                  shape, duration: float, settle: float) -> dict:
    """Measure renderer frame rate under one condition."""
    worker = None
    if backend is not None:
        worker = UpscalerLoop(model, backend, shape)
        worker.start()

    # Let the system settle before measuring: thermal state and scheduler
    # behaviour both take a moment to reflect the new load.
    time.sleep(settle)

    start = time.time()
    time.sleep(duration)
    end = time.time()

    if worker is not None:
        worker.stop_flag.set()
        worker.join(timeout=30)

    fps = samples_between(start, end)
    result = {
        "condition": name,
        "backend": backend or "none",
        "fps_samples": len(fps),
        "fps_median": round(statistics.median(fps), 3) if fps else None,
        "fps_mean": round(statistics.fmean(fps), 3) if fps else None,
        "fps_stdev": round(statistics.stdev(fps), 3) if len(fps) > 1 else 0.0,
        "upscale_iters": worker.iterations if worker else 0,
        "upscale_median_ms": (round(statistics.median(worker.latencies), 3)
                              if worker and worker.latencies else None),
        "error": worker.error if worker else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 3, 270, 480])
    parser.add_argument("--duration", type=float, default=20.0,
                        help="seconds measured per condition")
    parser.add_argument("--settle", type=float, default=5.0,
                        help="seconds to wait after changing load before measuring")
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", type=Path, default=Path("../gpu_load.html"))
    parser.add_argument("--out", type=Path, default=Path("fps_results.csv"))
    args = parser.parse_args()

    if not args.upscaler.exists():
        raise SystemExit(f"upscaler not found: {args.upscaler}")

    ensure_qnn()
    has_npu = bool(qnn_devices("npu"))
    print(f"NPU available: {has_npu}")

    server = HTTPServer(("127.0.0.1", args.port), FpsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"frame-rate server listening on http://127.0.0.1:{args.port}")

    page = args.page.resolve()
    if page.exists():
        print(f"opening {page}")
        webbrowser.open(page.as_uri())
    else:
        print(f"!! page not found at {page} -- open gpu_load.html manually")

    print("\nWaiting for the page to connect and calibrate its load...")
    print("Put the browser window in the foreground and leave it there.")
    deadline = time.time() + 90
    while time.time() < deadline:
        with SAMPLES_LOCK:
            settled = [s for s in SAMPLES if not s.get("calibrating", False)]
        if len(settled) >= 4:
            break
        time.sleep(1)
    else:
        raise SystemExit("no calibrated samples received; is the page open and visible?")

    with SAMPLES_LOCK:
        latest = SAMPLES[-1]
    print(f"connected: {latest['mean_fps']:.1f} fps at "
          f"{latest['iterations']} iterations, "
          f"{latest['width']}x{latest['height']}\n")

    conditions = [("baseline", None), ("cpu", "cpu")]
    if has_npu:
        conditions.append(("npu", "npu"))

    # Randomised order so thermal drift becomes noise rather than a bias
    # favouring whichever condition ran while the machine was coolest.
    rng = np.random.default_rng(42)
    schedule = []
    for repeat in range(args.repeats):
        for index in rng.permutation(len(conditions)):
            schedule.append((repeat, conditions[index]))

    results = []
    for step, (repeat, (name, backend)) in enumerate(schedule, 1):
        print(f"[{step}/{len(schedule)}] repeat {repeat}  condition: {name}")
        result = run_condition(name, args.upscaler, backend,
                               tuple(args.shape), args.duration, args.settle)
        result["repeat"] = repeat
        results.append(result)

        if result["fps_median"] is None:
            print("      no frame-rate samples -- is the browser still visible?")
        else:
            line = f"      {result['fps_median']:.2f} fps"
            if result["upscale_iters"]:
                line += (f"  |  upscaler {result['upscale_iters']} runs, "
                         f"{result['upscale_median_ms']} ms")
            print(line)
        if result["error"]:
            print(f"      upscaler error: {result['error'][:90]}")

        if step < len(schedule):
            time.sleep(args.cooldown)

    server.shutdown()

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    with SAMPLES_LOCK:
        pd.DataFrame(SAMPLES).to_csv("fps_raw_samples.csv", index=False)

    print("\n" + "=" * 68)
    summary = df.groupby("condition")["fps_median"].agg(["median", "std", "count"])
    print(summary.round(3).to_string())

    if "baseline" in summary.index:
        base = summary.loc["baseline", "median"]
        print(f"\nframe rate relative to an unloaded GPU (baseline {base:.2f} fps):")
        for condition in summary.index:
            value = summary.loc[condition, "median"]
            print(f"  {condition:<9} {value:7.2f} fps   {value / base:.3f}x")

    if {"cpu", "npu"} <= set(summary.index):
        cpu = summary.loc["cpu", "median"]
        npu = summary.loc["npu", "median"]
        gain = (npu / cpu - 1) * 100
        noise = (summary.loc["npu", "std"] / npu * 100) if npu else 0
        print(f"\nNPU versus CPU for the upscaler: {gain:+.2f}% renderer frame rate")
        print(f"run-to-run variation: about {noise:.2f}%")
        if abs(gain) < noise:
            print("The difference is inside the noise floor. Not a result.")

    print(f"\nwritten: {args.out}, fps_raw_samples.csv")


if __name__ == "__main__":
    main()
