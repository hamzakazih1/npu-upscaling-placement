"""
The experiment: does moving the upscaler to the NPU give the GPU its budget back?

Three configurations, as specified:

  A  gpu_only_4k    GPU renders at full resolution, no upscaler
  B  gpu_upscale    GPU renders at 1080p AND runs the upscaler  (DLSS/FSR model)
  C  npu_upscale    GPU renders at 1080p, NPU runs the upscaler (the proposal)

The measured quantity is the RENDERER's throughput, not the upscaler's latency.
That distinction is the whole point: prior work established that upscalers run
on NPUs, but not what offloading does to the workload competing for the chip.

The "renderer" here is a synthetic GPU load rather than a real game. It is a
heavy ONNX model run in a loop on the GPU, reporting iterations per second. This
trades ecological validity for control and reproducibility -- a real game cannot
be held constant across runs. State this plainly in the write-up.

Usage:
    python benchmark_concurrent.py --upscaler espcn_x2.onnx --load render_load.onnx
    python benchmark_concurrent.py --upscaler espcn_x2.onnx --load render_load.onnx --repeats 10
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from qnn_setup import ensure_qnn, make_qnn_session, qnn_devices


def make_session(model: Path, target: str) -> ort.InferenceSession:
    """
    Session for one target: "qnn-npu", "qnn-gpu", or a built-in provider name.

    QNN is a plugin EP and must be selected by device. Passing its name in
    providers=[...] is ignored without error and yields a CPU-only session.
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if target.startswith("qnn-"):
        return make_qnn_session(model, device=target.split("-")[1],
                                session_options=options)

    return ort.InferenceSession(str(model), options, providers=[target])


@dataclass
class RunResult:
    config: str
    render_res: str
    upscaler_provider: str | None
    duration_s: float
    render_iters: int
    render_ips: float          # renderer iterations per second -- the headline
    upscale_iters: int
    upscale_ips: float
    upscale_mean_ms: float | None
    notes: str = ""


class Worker(threading.Thread):
    """
    Runs one model in a loop until told to stop, counting iterations.

    A thread per workload is what makes the concurrent case possible: both
    loops issue work to their own accelerator at the same time, which is the
    situation a game plus an upscaler actually creates.
    """

    def __init__(self, session: ort.InferenceSession, data: np.ndarray, name: str):
        super().__init__(daemon=True)
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.data = data
        self.name = name
        self.stop_flag = threading.Event()
        self.iterations = 0
        self.latencies: list[float] = []
        self.error: str | None = None

    def run(self) -> None:
        try:
            while not self.stop_flag.is_set():
                start = time.perf_counter()
                self.session.run(None, {self.input_name: self.data})
                self.latencies.append((time.perf_counter() - start) * 1000)
                self.iterations += 1
        except Exception as exc:  # surface rather than dying silently
            self.error = str(exc)


def warm(session: ort.InferenceSession, data: np.ndarray, iters: int = 5) -> None:
    """Exclude first-call costs: context creation, graph compilation, autotuning."""
    name = session.get_inputs()[0].name
    for _ in range(iters):
        session.run(None, {name: data})


def run_config(config: str, load_model: Path, upscaler_model: Path | None,
               load_provider: str, upscaler_provider: str | None,
               load_shape: tuple, upscale_shape: tuple,
               duration: float) -> RunResult:
    """Run one configuration for a fixed wall-clock duration."""
    def failed(message: str) -> RunResult:
        """A configuration that could not start is data, not an absent row."""
        return RunResult(
            config=config, render_res=f"{load_shape[3]}x{load_shape[2]}",
            upscaler_provider=upscaler_provider, duration_s=0.0,
            render_iters=0, render_ips=0.0, upscale_iters=0, upscale_ips=0.0,
            upscale_mean_ms=None, notes=f"setup failed: {message}",
        )

    try:
        load_session = make_session(load_model, load_provider)
        load_data = np.random.randn(*load_shape).astype(np.float32)
        warm(load_session, load_data)

        upscale_session = None
        upscale_data = None
        if upscaler_model is not None and upscaler_provider is not None:
            upscale_session = make_session(upscaler_model, upscaler_provider)
            upscale_data = np.random.randn(*upscale_shape).astype(np.float32)
            warm(upscale_session, upscale_data)
    except Exception as error:
        return failed(str(error))

    render_worker = Worker(load_session, load_data, "render")
    workers = [render_worker]
    if upscale_session is not None:
        upscale_worker = Worker(upscale_session, upscale_data, "upscale")
        workers.append(upscale_worker)
    else:
        upscale_worker = None

    # Start together so the contention window is the whole measurement.
    start = time.perf_counter()
    for worker in workers:
        worker.start()
    time.sleep(duration)
    for worker in workers:
        worker.stop_flag.set()
    for worker in workers:
        worker.join(timeout=30)
    elapsed = time.perf_counter() - start

    notes = "; ".join(f"{w.name}: {w.error}" for w in workers if w.error)

    upscale_mean = None
    if upscale_worker is not None and upscale_worker.latencies:
        upscale_mean = round(float(np.median(upscale_worker.latencies)), 3)

    return RunResult(
        config=config,
        render_res=f"{load_shape[3]}x{load_shape[2]}",
        upscaler_provider=upscaler_provider,
        duration_s=round(elapsed, 3),
        render_iters=render_worker.iterations,
        render_ips=round(render_worker.iterations / elapsed, 3),
        upscale_iters=upscale_worker.iterations if upscale_worker else 0,
        upscale_ips=round(upscale_worker.iterations / elapsed, 3) if upscale_worker else 0.0,
        upscale_mean_ms=upscale_mean,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--load", type=Path, required=True,
                        help="heavy model used as the renderer proxy")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds per configuration")
    parser.add_argument("--repeats", type=int, default=5,
                        help="repeats per configuration, for the variance estimate")
    parser.add_argument("--cooldown", type=float, default=20.0,
                        help="seconds between runs, to limit thermal drift")
    parser.add_argument("--gpu-provider", default="qnn-gpu",
                        help='"qnn-gpu" (Adreno via QNN) or a built-in provider name')
    parser.add_argument("--npu-provider", default="qnn-npu")
    parser.add_argument("--out", type=Path, default=Path("concurrent_results.csv"))
    args = parser.parse_args()

    ensure_qnn()
    available = list(ort.get_available_providers())
    if qnn_devices("npu"):
        available.append("qnn-npu")
    if qnn_devices("gpu"):
        available.append("qnn-gpu")
    print("targets:", available)
    for required in (args.gpu_provider, args.npu_provider):
        if required not in available:
            print(f"!! {required} unavailable -- related configs will be skipped")
    print()

    # Renderer proxy shapes. 4K is four times the pixels of 1080p, which is the
    # saving the upscaler is meant to buy back.
    shape_4k = (1, 3, 1080, 1920)
    shape_1080 = (1, 3, 540, 960)
    upscale_shape = (1, 3, 270, 480)

    plan = []
    if args.gpu_provider in available:
        plan.append(("A_gpu_only_4k", shape_4k, None, None))
        plan.append(("B_gpu_upscale", shape_1080, args.upscaler, args.gpu_provider))
    if args.npu_provider in available and args.gpu_provider in available:
        plan.append(("C_npu_upscale", shape_1080, args.upscaler, args.npu_provider))

    if not plan:
        raise SystemExit("no runnable configurations")

    # Randomised order: thermal drift then shows up as noise rather than as a
    # systematic bias favouring whichever config ran while the machine was cold.
    rng = np.random.default_rng(42)
    schedule = []
    for repeat in range(args.repeats):
        order = rng.permutation(len(plan))
        schedule.extend((repeat, plan[i]) for i in order)

    results = []
    total = len(schedule)
    for index, (repeat, (config, load_shape, upscaler, provider)) in enumerate(schedule, 1):
        print(f"[{index}/{total}] repeat {repeat}  {config}  "
              f"render {load_shape[3]}x{load_shape[2]}"
              f"{f'  upscaler on {provider}' if provider else ''}")
        try:
            result = run_config(
                config, args.load, upscaler,
                args.gpu_provider, provider,
                load_shape, upscale_shape, args.duration)
            result_dict = asdict(result)
            result_dict["repeat"] = repeat
            results.append(result_dict)
            print(f"      renderer {result.render_ips:.2f} it/s"
                  + (f" | upscaler {result.upscale_ips:.2f} it/s "
                     f"({result.upscale_mean_ms} ms)" if result.upscale_iters else ""))
            if result.notes:
                print(f"      NOTE: {result.notes}")
        except Exception as error:
            print(f"      FAILED: {error}")

        if index < total:
            time.sleep(args.cooldown)

    if not results:
        raise SystemExit("no successful runs")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print()
    print("=" * 70)
    summary = df.groupby("config")["render_ips"].agg(["median", "std", "count"])
    summary.columns = ["render_ips_median", "render_ips_sd", "runs"]
    print(summary.round(3).to_string())

    if "A_gpu_only_4k" in summary.index:
        baseline = summary.loc["A_gpu_only_4k", "render_ips_median"]
        print()
        print("Renderer throughput relative to 4K baseline:")
        for config in summary.index:
            ratio = summary.loc[config, "render_ips_median"] / baseline
            print(f"  {config}: {ratio:.3f}x")

    if {"B_gpu_upscale", "C_npu_upscale"} <= set(summary.index):
        gpu_side = summary.loc["B_gpu_upscale", "render_ips_median"]
        npu_side = summary.loc["C_npu_upscale", "render_ips_median"]
        gain = (npu_side / gpu_side - 1) * 100
        noise = summary.loc["C_npu_upscale", "render_ips_sd"] / npu_side * 100
        print()
        print(f"Offloading the upscaler to the NPU changes renderer throughput "
              f"by {gain:+.2f}%")
        print(f"Run-to-run variation is about {noise:.2f}%")
        if abs(gain) < noise:
            print("The difference is within the noise floor. Not a result.")

    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "providers": available,
    }
    Path("environment.json").write_text(json.dumps(environment, indent=2))
    print(f"\nwritten: {args.out} and environment.json")


if __name__ == "__main__":
    main()
