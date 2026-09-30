"""
Node-level placement verification: which provider actually ran each node.

Strict mode already prevents CPU fallback at session creation. This is the
independent check -- it reads ONNX Runtime's own profile -- and is what
`npu-upscale doctor` reports, so a result can be trusted without trusting
this package.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from .runtime import CPU, create_cpu_session, create_npu_session


def read_profile(path: str | os.PathLike) -> dict:
    with open(path) as handle:
        events = json.load(handle)
    providers: Counter = Counter()
    cpu_ops = []
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider") or args.get("execution_provider")
        if not provider:
            continue
        providers[provider] += 1
        if CPU in provider:
            cpu_ops.append(args.get("op_name", event.get("name", "?")))
    return {"providers": dict(providers), "cpu_ops": sorted(set(cpu_ops))}


def verify_placement(model: str | os.PathLike, device: str = "npu",
                     strict: bool = False, runs: int = 3) -> dict:
    """
    Run a model with profiling and report per-provider node executions, plus
    the largest deviation from the CPU reference output.

    strict defaults to False here on purpose: the point is to *see* what would
    fall back, which a strict session would refuse to show.
    """
    workdir = Path(tempfile.mkdtemp(prefix="npu_upscale_profile_"))
    prefix = str(workdir / device)
    if device == "npu":
        session, info = create_npu_session(model, strict=strict, cache=False,
                                           profile_prefix=prefix)
    else:
        session, info = create_cpu_session(model, profile_prefix=prefix)

    source = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 64 for d in source.shape]
    data = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(runs):
        output = session.run(None, {source.name: data})[0]
    report = read_profile(session.end_profiling())

    reference, _ = create_cpu_session(model)
    expected = reference.run(None, {source.name: data})[0]
    report["max_abs_diff_vs_cpu"] = float(np.abs(output - expected).max())

    total = sum(report["providers"].values())
    on_cpu = sum(n for p, n in report["providers"].items() if CPU in p)
    report["total_nodes"] = total
    report["fully_on_device"] = device == "cpu" or (total > 0 and on_cpu == 0)
    report["info"] = info
    return report
