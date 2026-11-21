"""
Placement experiment, version 2: reproducible renderer-throughput measurement.

Same question as benchmark/scripts/run_experiment.py: does running the
upscaler on the Hexagon NPU, or on the CPU, slow down a GPU renderer? This
version fixes every defect found when the dissertation data were re-analysed:

  defect in v1                                   fix in v2
  ---------------------------------------------  -----------------------------------------------
  metric = rolling median of per-frame fps,      metric = frames / page-clock seconds over the
    which snaps to vsync steps                     whole window (true throughput)
  7 s rolling window, 5 s settle: windows mixed  no rolling window; settle 10 s
  page stall (28 s) went unnoticed               any report gap > 1 s or hidden page = invalid
                                                   run, repeated automatically
  NPU session creation ate into the settle time  settle starts only after the upscaler is warm
  canvas followed the window size                fixed canvas (default 1189x966)
  shader load auto-calibrated each session       fixed load (default 700 iterations)
  3 repeats                                      10 repeats by default
  free-running upscaler: 34x unequal work        --upscale-hz to fix the inference rate
  raw samples saved without condition labels     every report and every frame interval labelled
  no environment record                          env.json: versions, devices, model hashes, power

Usage (Windows on ARM, ARM64 Python, from the repository root):

    python reproduce\\laptop\\run_experiment_v2.py --upscaler models\\v1\\espcn_x2_int8.onnx

    # the dissertation's exact settings, to reproduce its numbers:
    python reproduce\\laptop\\run_experiment_v2.py --preset dissertation

Results go to reproduce/results/laptop/<timestamp>/ and are analysed at the
end. Compare them with the expected values:

    python reproduce\\analysis\\compare.py laptop reproduce\\results\\laptop\\<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "scripts"))
sys.path.insert(0, str(ROOT / "reproduce" / "analysis"))

PAGE = HERE / "gpu_load_v2.html"

# --------------------------------------------------------------------------
# Shared state between the HTTP thread and the experiment loop
# --------------------------------------------------------------------------

LOCK = threading.Lock()
REPORTS: list[dict] = []
PHASE = {"run_id": None, "attempt": None, "condition": None, "repeat": None, "phase": "startup"}


class Handler(BaseHTTPRequestHandler):
    """Serves the page and receives its reports, stamping each with the phase."""

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/gpu_load_v2.html"):
            body = PAGE.read_bytes()
            self._respond(body, "text/html; charset=utf-8")
        else:
            self._respond(b'{"ok":true}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._respond(b'{"ok":false}')
            return
        payload["received"] = time.time()
        with LOCK:
            payload.update({k: PHASE[k] for k in PHASE})
            REPORTS.append(payload)
        self._respond()

    def do_OPTIONS(self):
        self._respond()

    def _respond(self, body: bytes = b"{}", ctype: str = "application/json"):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def set_phase(**kwargs):
    with LOCK:
        PHASE.update(kwargs)


# --------------------------------------------------------------------------
# Upscaler worker
# --------------------------------------------------------------------------

class UpscalerLoop(threading.Thread):
    """
    Runs the upscaler on one backend until stopped.

    `ready` is set only after the session exists and the warmup runs are done,
    so the settle period never includes session creation (the v1 runner
    started its settle timer before the NPU session was built).

    With hz > 0 the loop issues one inference per period, so every backend
    performs the same amount of work. hz = 0 runs back-to-back, as v1 did.
    """

    def __init__(self, model: Path, backend: str, shape, hz: float, seed: int):
        super().__init__(daemon=True)
        self.model, self.backend, self.shape, self.hz = model, backend, shape, hz
        self.seed = seed
        self.ready = threading.Event()
        self.stop_flag = threading.Event()
        self.counting = threading.Event()
        self.iterations = 0
        self.latencies: list[float] = []
        self.missed_deadlines = 0
        self.error: str | None = None
        self.providers: list[str] = []

    def run(self):
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.log_severity_level = 3   # errors only; warnings would flood the console
            if self.backend == "npu":
                from qnn_setup import make_qnn_session
                session = make_qnn_session(self.model, device="npu", session_options=options)
            else:
                session = ort.InferenceSession(str(self.model), options,
                                               providers=["CPUExecutionProvider"])
            self.providers = session.get_providers()
            name = session.get_inputs()[0].name
            # Same input every run: a fixed-seed tensor in [0, 1].
            data = np.random.default_rng(self.seed).random(self.shape, dtype=np.float32)
            for _ in range(5):
                session.run(None, {name: data})
            self.ready.set()

            period = 1.0 / self.hz if self.hz > 0 else 0.0
            next_due = time.perf_counter()
            while not self.stop_flag.is_set():
                start = time.perf_counter()
                session.run(None, {name: data})
                elapsed = (time.perf_counter() - start) * 1000
                if self.counting.is_set():
                    self.latencies.append(elapsed)
                    self.iterations += 1
                if period:
                    next_due += period
                    sleep = next_due - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        if self.counting.is_set():
                            self.missed_deadlines += 1
                        next_due = time.perf_counter()
        except Exception as exc:  # recorded, and the run is marked invalid
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def reports_for(run_id, attempt, phase="measure"):
    with LOCK:
        return [r for r in REPORTS
                if r.get("run_id") == run_id and r.get("attempt") == attempt
                and r.get("phase") == phase]


def summarise_window(rows, duration, max_gap):
    """Throughput and validity checks for one measurement window."""
    pages = {r.get("page_id") for r in rows}
    if len(pages) > 1:
        return {"valid": False, "n_reports": len(rows),
                "reason": f"reports from {len(pages)} pages at once (close other tabs)"}
    rows = sorted(rows, key=lambda r: r["t_page_ms"])
    if len(rows) < 2:
        return {"valid": False, "reason": "fewer than 2 reports", "n_reports": len(rows)}
    frames = rows[-1]["frames"] - rows[0]["frames"]
    seconds = (rows[-1]["t_page_ms"] - rows[0]["t_page_ms"]) / 1000
    gaps = np.diff([r["t_page_ms"] for r in rows]) / 1000
    hidden = sum(1 for r in rows if r.get("visibility") != "visible")
    reasons = []
    if gaps.max() > max_gap:
        reasons.append(f"report gap {gaps.max():.2f}s > {max_gap}s (page stalled or throttled)")
    if hidden:
        reasons.append(f"{hidden} report(s) with page hidden")
    # Reports arrive every 0.5 s, so a complete window spans about duration - 0.5 s.
    if seconds < duration - 1.5:
        reasons.append(f"window covers {seconds:.1f}s of {duration}s")
    if any(r.get("calibrating") for r in rows):
        reasons.append("page still calibrating")
    return {
        "n_reports": len(rows), "frames": int(frames), "seconds": float(seconds),
        "throughput_fps": float(frames / seconds) if seconds > 0 else float("nan"),
        "max_report_gap_s": float(gaps.max()),
        "iterations": int(rows[-1]["iterations"]),
        "canvas": f"{rows[-1]['width']}x{rows[-1]['height']}",
        "valid": not reasons, "reason": "; ".join(reasons),
    }


def run_condition(run_id, attempt, condition, repeat, args):
    backend = None if condition == "baseline" else condition
    worker = None
    set_phase(run_id=run_id, attempt=attempt, condition=condition, repeat=repeat, phase="warmup")
    if backend:
        worker = UpscalerLoop(args.upscaler, backend, tuple(args.shape), args.upscale_hz, args.seed)
        worker.start()
        if not worker.ready.wait(timeout=120):
            worker.error = "upscaler did not become ready within 120 s"

    set_phase(phase="settle")
    time.sleep(args.settle)

    set_phase(phase="measure")
    if worker:
        worker.counting.set()
    started = time.time()
    time.sleep(args.duration)
    ended = time.time()
    set_phase(phase="teardown")

    if worker:
        worker.counting.clear()
        worker.stop_flag.set()
        worker.join(timeout=30)

    summary = summarise_window(reports_for(run_id, attempt), args.duration, args.max_gap)
    lat = worker.latencies if worker else []
    summary.update({
        "run_id": run_id, "attempt": attempt, "condition": condition, "repeat": repeat,
        "backend": backend or "none", "wall_start": started, "wall_end": ended,
        "upscale_iters": worker.iterations if worker else 0,
        "upscale_median_ms": float(np.median(lat)) if lat else None,
        "upscale_p90_ms": float(np.percentile(lat, 90)) if lat else None,
        "upscale_missed_deadlines": worker.missed_deadlines if worker else 0,
        "upscale_providers": ";".join(worker.providers) if worker else "",
        "error": worker.error if worker else None,
    })
    if worker and worker.error:
        summary["valid"] = False
        summary["reason"] = (summary.get("reason", "") + "; upscaler error").strip("; ")
    if backend == "npu" and worker and "QNNExecutionProvider" not in worker.providers:
        summary["valid"] = False
        summary["reason"] = (summary.get("reason", "") + "; NPU session not on QNN").strip("; ")
    return summary


# --------------------------------------------------------------------------
# Environment record
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(args) -> dict:
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": sys.version,
        "numpy": np.__version__,
        "upscaler": str(args.upscaler), "upscaler_sha256": sha256(args.upscaler),
        "page_sha256": sha256(PAGE),
    }
    try:
        import onnxruntime as ort
        env["onnxruntime"] = ort.__version__
        try:
            env["ep_devices"] = [f"{d.ep_name}:{d.device.type}" for d in ort.get_ep_devices()]
        except AttributeError:
            env["ep_devices"] = []
    except ImportError:
        env["onnxruntime"] = None
    try:
        import onnxruntime_qnn  # noqa: F401
        from importlib.metadata import version
        env["onnxruntime_qnn"] = version("onnxruntime-qnn")
    except Exception:
        env["onnxruntime_qnn"] = None
    if os.name == "nt":
        for key, cmd in (("power_scheme", ["powercfg", "/getactivescheme"]),
                         ("battery", ["powershell", "-NoProfile", "-Command",
                                      "(Get-CimInstance Win32_Battery | "
                                      "Select-Object -First 1 BatteryStatus,EstimatedChargeRemaining | "
                                      "ConvertTo-Json -Compress)"]),
                         ("refresh_rate_hz", ["powershell", "-NoProfile", "-Command",
                                              "(Get-CimInstance Win32_VideoController | "
                                              "Select-Object -First 1).CurrentRefreshRate"])):
            try:
                env[key] = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=15).stdout.strip()
            except Exception as exc:
                env[key] = f"unavailable: {exc}"
    return env


# --------------------------------------------------------------------------
# Browser
# --------------------------------------------------------------------------

BROWSERS = {
    "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"],
    "chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
}


def open_page(url: str, args):
    if args.browser == "none":
        print(f"Open this URL yourself, full screen, and leave it in front:\n  {url}")
        return None
    if args.browser == "default":
        webbrowser.open(url)
        return None
    exe = next((p for p in BROWSERS[args.browser] if Path(p).exists()), None)
    exe = exe or shutil.which("msedge" if args.browser == "edge" else "chrome")
    if not exe:
        print(f"!! {args.browser} not found; opening the default browser instead")
        webbrowser.open(url)
        return None
    # A throwaway profile, so flags apply and no extensions or tabs interfere.
    profile = tempfile.mkdtemp(prefix="npu-bench-")
    flags = [f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
             "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
             "--disable-backgrounding-occluded-windows", "--start-fullscreen"]
    if args.vsync_off:
        flags += ["--disable-gpu-vsync", "--disable-frame-rate-limit"]
    return subprocess.Popen([exe, *flags, url])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

PRESETS = {
    # The settings the dissertation used, except where v1 was defective.
    "dissertation": dict(upscaler=ROOT / "models" / "v1" / "espcn_x2_int8.onnx",
                         iterations=700, width=1189, height=966, duration=20.0,
                         settle=10.0, cooldown=15.0, repeats=10, upscale_hz=0.0,
                         vsync_off=False, seed=42),
    # Matched work: one upscale per rendered frame, vsync off.
    "matched": dict(upscaler=ROOT / "models" / "v1" / "espcn_x2_int8.onnx",
                    iterations=700, width=1189, height=966, duration=20.0,
                    settle=10.0, cooldown=15.0, repeats=10, upscale_hz=20.0,
                    vsync_off=True, seed=42),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), help="named settings; flags given explicitly override it")
    p.add_argument("--upscaler", type=Path)
    p.add_argument("--shape", type=int, nargs=4, default=[1, 3, 270, 480])
    p.add_argument("--iterations", type=int, help="shader load; 0 = auto-calibrate (not reproducible)")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--settle", type=float)
    p.add_argument("--cooldown", type=float)
    p.add_argument("--repeats", type=int)
    p.add_argument("--upscale-hz", type=float, dest="upscale_hz",
                   help="inferences per second; 0 = back-to-back as in v1")
    p.add_argument("--vsync-off", action="store_true", default=None, dest="vsync_off")
    p.add_argument("--seed", type=int)
    p.add_argument("--conditions", nargs="+", default=["baseline", "cpu", "npu"])
    p.add_argument("--max-gap", type=float, default=1.0, help="largest report gap in a valid window (s)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--browser", choices=["edge", "chrome", "default", "none"], default="edge")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--out", type=Path)
    p.add_argument("--allow-no-npu", action="store_true",
                   help="drop the npu condition if no QNN NPU is present (for testing only)")
    args = p.parse_args()

    defaults = dict(PRESETS["dissertation"])
    if args.preset:
        defaults.update(PRESETS[args.preset])
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.upscaler = Path(args.upscaler)
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out = ROOT / "reproduce" / "results" / "laptop" / stamp
    return args


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.upscaler.exists():
        raise SystemExit(f"upscaler not found: {args.upscaler}")
    args.out.mkdir(parents=True, exist_ok=True)

    if "npu" in args.conditions:
        try:
            from qnn_setup import ensure_qnn, qnn_devices
            has_npu = ensure_qnn(verbose=True) and bool(qnn_devices("npu"))
        except Exception as exc:
            print(f"QNN check failed: {exc}")
            has_npu = False
        if not has_npu:
            if not args.allow_no_npu:
                raise SystemExit("No QNN NPU device. Run benchmark/scripts/qnn_setup.py to diagnose, "
                                 "or pass --allow-no-npu to test without it.")
            args.conditions = [c for c in args.conditions if c != "npu"]
            print("!! continuing WITHOUT the npu condition (--allow-no-npu)")

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    env = environment(args)
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"upscaler sha256 {env['upscaler_sha256'][:16]}…  onnxruntime {env.get('onnxruntime')}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = (f"http://127.0.0.1:{args.port}/?iter={args.iterations}&w={args.width}"
           f"&h={args.height}&port={args.port}")
    browser = open_page(url, args)

    print("\nWaiting for the page… keep the browser window in front, plugged in, Best Performance.")
    deadline = time.time() + 120
    while time.time() < deadline:
        with LOCK:
            ok = [r for r in REPORTS if not r.get("calibrating")]
        if len(ok) >= 6:
            break
        time.sleep(0.5)
    else:
        raise SystemExit("no reports from the page; is it open and visible?")
    with LOCK:
        latest = REPORTS[-1]
    print(f"page connected: {latest['width']}x{latest['height']}, {latest['iterations']} iterations\n")
    if args.iterations and latest["iterations"] != args.iterations:
        raise SystemExit("page is not using the requested shader load")

    rng = np.random.default_rng(args.seed)
    schedule = []
    for repeat in range(args.repeats):
        for index in rng.permutation(len(args.conditions)):
            schedule.append((repeat, args.conditions[index]))

    runs = []
    for step, (repeat, condition) in enumerate(schedule, start=1):
        run_id = step
        for attempt in range(1, args.max_retries + 2):
            print(f"[{step}/{len(schedule)}] repeat {repeat}  {condition:<8} attempt {attempt}", end="  ")
            result = run_condition(run_id, attempt, condition, repeat, args)
            result["order"] = step
            runs.append(result)
            if result["valid"]:
                line = f"{result['throughput_fps']:.3f} fps"
                if result["upscale_iters"]:
                    line += f" | upscaler {result['upscale_iters']} runs, {result['upscale_median_ms']:.3f} ms"
                print(line)
            else:
                print(f"INVALID: {result['reason']}")
            set_phase(phase="cooldown")
            time.sleep(args.cooldown)
            if result["valid"]:
                break
        write_csv(args.out / "runs.csv", runs)   # saved as we go, in case of a crash

    set_phase(phase="done")
    server.shutdown()
    if browser is not None:
        browser.terminate()

    with LOCK:
        reports = list(REPORTS)
    intervals = []
    flat = []
    for r in reports:
        for i, ms in enumerate(r.get("intervals_ms", [])):
            intervals.append({"run_id": r["run_id"], "attempt": r["attempt"], "condition": r["condition"],
                              "phase": r["phase"], "seq": r["seq"], "i": i, "interval_ms": ms})
        flat.append({k: v for k, v in r.items() if k not in ("intervals_ms", "user_agent")})
    write_csv(args.out / "reports.csv", flat)
    write_csv(args.out / "frame_intervals.csv", intervals)
    env["user_agent"] = reports[-1].get("user_agent") if reports else None
    (args.out / "env.json").write_text(json.dumps(env, indent=2))

    invalid = sum(1 for r in runs if not r["valid"])
    print(f"\n{len(runs)} attempts, {invalid} invalid and repeated. Results in {args.out}")

    try:
        import analyse_laptop_run
    except ImportError as exc:   # e.g. no scipy wheel for this Python on ARM64
        print(f"\nAnalysis skipped ({exc}). Copy {args.out} to any machine with the analysis "
              "requirements and run:\n  python reproduce/analysis/compare.py laptop <folder>")
        return
    analyse_laptop_run.main(args.out)
    print(f"\nNext: python reproduce\\analysis\\compare.py laptop {args.out}")


if __name__ == "__main__":
    main()
