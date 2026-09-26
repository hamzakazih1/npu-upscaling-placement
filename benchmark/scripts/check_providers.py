"""
Step 1: verify the NPU is actually being used.

This runs before any benchmarking, because every later number depends on it.
ONNX Runtime will happily accept a QNN provider request, silently fall back to
CPU for unsupported operators, and report nothing. A benchmark built on that
measures CPU execution while calling it NPU.

What this does:
  - lists available execution providers
  - runs the model on each backend
  - enables profiling and reports which provider executed each node
  - flags any operator that fell back

Usage:
    python check_providers.py --model espcn_x2.onnx
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

# QNN is a plugin EP as of onnxruntime-qnn 2.6.0 and must be registered before
# it appears in get_available_providers(). See qnn_setup for why.
from qnn_setup import ensure_qnn, make_qnn_session, qnn_devices


def build_session(model_path: Path, target: str, profile_prefix=None):
    """
    Create a session for one target.

    QNN targets go through make_qnn_session, which selects by device. Passing
    "QNNExecutionProvider" in providers=[...] is silently ignored for plugin
    providers, producing a CPU-only session that looks like it worked.
    """
    options = ort.SessionOptions()
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix

    if target.startswith("qnn-"):
        return make_qnn_session(model_path, device=target.split("-")[1],
                                session_options=options)

    return ort.InferenceSession(str(model_path), options, providers=[target])


def placement_report(profile_path: Path) -> dict:
    """
    Parse an ONNX Runtime profile and count which provider ran each node.

    The profile records one entry per node execution with an
    'execution_provider' argument. Counting them is the only reliable way to
    know what actually ran where.
    """
    with open(profile_path) as handle:
        events = json.load(handle)

    providers = Counter()
    fallback_ops = []

    for event in events:
        args = event.get("args", {})
        provider = args.get("provider") or args.get("execution_provider")
        if not provider:
            continue
        providers[provider] += 1
        if "CPUExecutionProvider" in provider:
            fallback_ops.append(args.get("op_name", event.get("name", "?")))

    return {"providers": dict(providers), "cpu_nodes": fallback_ops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("espcn_x2.onnx"))
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 3, 270, 480])
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    ensure_qnn()
    print()
    print("onnxruntime:", ort.__version__)
    print("device:", ort.get_device())
    available = ort.get_available_providers()
    print("available providers:")
    for provider in available:
        print("   ", provider)
    print()

    # QNN is exposed once per backing device, so the NPU and the Adreno GPU
    # are separate targets even though both report as QNNExecutionProvider.
    testable = []
    if qnn_devices("npu"):
        testable.append("qnn-npu")
    if qnn_devices("gpu"):
        testable.append("qnn-gpu")
    testable.append("CPUExecutionProvider")

    print("targets to test:", testable)
    if "qnn-npu" not in testable:
        print("!! no QNN NPU device. Run: python qnn_setup.py")
        print()

    data = np.random.randn(*args.shape).astype(np.float32)
    reference = None

    for provider in testable:
        print("=" * 62)
        print(provider)
        print("=" * 62)

        prefix = f"profile_{provider.replace('ExecutionProvider', '').lower().replace('-', '_')}"
        for stale in glob.glob(f"{prefix}*.json"):
            os.remove(stale)

        try:
            session = build_session(args.model, provider, profile_prefix=prefix)
        except Exception as error:
            print(f"  FAILED to create session: {error}\n")
            continue

        input_name = session.get_inputs()[0].name
        try:
            for _ in range(3):
                output = session.run(None, {input_name: data})[0]
        except Exception as error:
            print(f"  FAILED to run: {error}\n")
            continue

        profile_path = Path(session.end_profiling())

        # Correctness: every backend must agree with the CPU reference. A fast
        # wrong answer is the failure mode this catches.
        if provider == "CPUExecutionProvider":
            reference = output
            print("  (reference output)")
        elif reference is not None:
            max_diff = float(np.abs(output - reference).max())
            verdict = "OK" if max_diff < 1e-2 else "MISMATCH - investigate"
            print(f"  max difference vs CPU: {max_diff:.3e}  {verdict}")

        if profile_path.exists():
            report = placement_report(profile_path)
            print("  node executions by provider:")
            for name, count in sorted(report["providers"].items()):
                print(f"    {name}: {count}")

            total = sum(report["providers"].values())
            on_qnn = report["providers"].get("QNNExecutionProvider", 0)
            if total:
                print(f"  nodes on QNN: {on_qnn}/{total} ({on_qnn / total:.0%})")

            if provider != "CPUExecutionProvider" and report["cpu_nodes"]:
                unique = sorted(set(report["cpu_nodes"]))
                print(f"  !! {len(unique)} operator type(s) fell back to CPU:")
                for op in unique[:15]:
                    print(f"       {op}")
                print("  Any timing for this provider includes CPU execution.")
            elif provider != "CPUExecutionProvider":
                print("  no CPU fallback detected")
        else:
            print("  no profile written; placement unverified")
        print()

    print("=" * 62)
    print("Proceed to benchmarking only for providers with no CPU fallback.")
    print("If QNN falls back on most nodes, the model needs redesigning before")
    print("any latency number is meaningful.")


if __name__ == "__main__":
    main()
