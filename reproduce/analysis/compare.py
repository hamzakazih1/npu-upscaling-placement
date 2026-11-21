"""
Check a run against the expected results.

    python reproduce/analysis/compare.py dissertation
        Re-runs the re-analysis of the saved dissertation data and checks it
        matches reproduce/expected/dissertation_reanalysis.json exactly.

    python reproduce/analysis/compare.py laptop <results folder or summary.json>
        Checks a Snapdragon X run of reproduce/laptop/run_experiment_v2.py
        against reproduce/expected/laptop_expected.json.

    python reproduce/analysis/compare.py colab <colab_summary.json>
        Checks a run of reproduce/colab/03_reproducible_model.ipynb against
        reproduce/expected/colab_expected.json, and against
        reproduce/expected/colab_reference.json when that file exists.

Exit status is 0 when every required check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE.parent / "expected"


class Report:
    def __init__(self, title):
        self.title, self.rows, self.failed = title, [], 0

    def check(self, name, ok, detail, required=True):
        status = "PASS" if ok else ("FAIL" if required else "note")
        if not ok and required:
            self.failed += 1
        self.rows.append((status, name, detail))

    def info(self, name, detail):
        self.rows.append(("info", name, detail))

    def print(self):
        print(f"\n{self.title}\n" + "=" * 78)
        for status, name, detail in self.rows:
            print(f"[{status:<4}] {name:<38} {detail}")
        print("=" * 78)
        verdict = "REPRODUCED" if not self.failed else f"NOT REPRODUCED ({self.failed} required check(s) failed)"
        print(verdict)
        return 0 if not self.failed else 1


def get(d, path):
    for key in path.split("."):
        d = d[key]
    return d


def within(value, target, tol, kind="abs"):
    if kind == "rel":
        return abs(value - target) <= tol * abs(target)
    return abs(value - target) <= tol


# ------------------------------------------------------------------ dissertation

def deep_diff(a, b, path="", rtol=1e-9, out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing on one side")
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", rtol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", rtol, out)
    elif isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            out.append(f"{path}: {a!r} vs {b!r}")
        elif not (math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def compare_dissertation():
    sys.path.insert(0, str(HERE))
    import reanalyse_dissertation
    got = reanalyse_dissertation.main(figures=False)
    expected = json.loads((EXPECTED / "dissertation_reanalysis.json").read_text())
    r = Report("Re-analysis of the dissertation data vs expected")
    diffs = deep_diff(got, expected)
    r.check("every number matches (rel 1e-9)", not diffs,
            "identical" if not diffs else f"{len(diffs)} differ, first: {diffs[0]}")
    m = got["section3_throughput_model"]
    r.info("NPU vs baseline", f"{m['npu_vs_baseline_pct']['estimate']:+.2f}% "
           f"(95% CI {m['npu_vs_baseline_pct']['ci_low']:+.2f} to {m['npu_vs_baseline_pct']['ci_high']:+.2f})")
    return r.print()


# ------------------------------------------------------------------ laptop

def compare_laptop(target):
    target = Path(target)
    path = target / "summary.json" if target.is_dir() else target
    if not path.exists() and target.is_dir():
        sys.path.insert(0, str(HERE))
        import analyse_laptop_run
        analyse_laptop_run.main(target)
    got = json.loads(path.read_text())
    exp = json.loads((EXPECTED / "laptop_expected.json").read_text())
    r = Report(f"Snapdragon X run vs expected  ({path})")

    setup = exp["setup"]
    same_load = got["iterations"] == [setup["iterations"]] and got["canvas"] == [setup["canvas"]]
    r.check("shader load and canvas as dissertation", same_load,
            f"{got['iterations']} iterations, {got['canvas']} (expected {setup['iterations']}, {setup['canvas']})")
    for cond in ("baseline", "cpu", "npu"):
        n = got["throughput_by_condition"].get(cond, {}).get("n", 0)
        r.check(f"valid runs: {cond}", n >= exp["min_valid_runs_per_condition"],
                f"{n} (need >= {exp['min_valid_runs_per_condition']})")

    t = exp["throughput"]
    base = got["throughput_by_condition"].get("baseline", {}).get("mean", float("nan"))
    r.check("baseline throughput", within(base, t["baseline_fps"], t["baseline_rel_tol"], "rel"),
            f"{base:.3f} fps vs {t['baseline_fps']:.3f} +/- {t['baseline_rel_tol']:.0%}",
            required=same_load)

    eq = got.get("equivalence", {})
    for cond in ("npu", "cpu"):
        if cond not in eq:
            r.check(f"{cond} equivalent to baseline", False, "condition missing")
            continue
        e = eq[cond]
        expect = exp["equivalence"][cond]
        r.check(f"{cond} equivalent to baseline (+/-1%)", e["equivalent"] == expect["equivalent"],
                f"90% CI of ratio {e['ci90_low']:.4f}-{e['ci90_high']:.4f}, p {e['p_tost']:.4f} "
                f"(dissertation data: {expect['dissertation_ci90']})",
                required=expect["required"])

    lat = got.get("latency", {})
    for cond in ("npu", "cpu"):
        if cond in lat:
            spec = exp["latency_ms"][cond]
            v = lat[cond]["median_ms"]
            r.check(f"{cond} upscaler median latency", within(v, spec["value"], spec["rel_tol"], "rel"),
                    f"{v:.3f} ms vs {spec['value']} +/- {spec['rel_tol']:.0%}")
    if "cpu_over_npu" in lat:
        lo, hi = exp["latency_ms"]["cpu_over_npu_range"]
        r.check("CPU / NPU latency ratio", lo <= lat["cpu_over_npu"] <= hi,
                f"{lat['cpu_over_npu']:.1f}x (expected {lo}-{hi}x; dissertation 65.5x)")

    if "legacy_metric_model" in got and "npu" in got["legacy_metric_model"]:
        v = got["legacy_metric_model"]["npu"]
        r.info("v1 metric on this run (npu)", f"{v['estimate']:+.3f} fps, p {v['p']:.3g} "
               "(dissertation: -0.508 fps, an artefact)")
    return r.print()


# ------------------------------------------------------------------ colab

def compare_colab(target):
    got = json.loads(Path(target).read_text())
    exp = json.loads((EXPECTED / "colab_expected.json").read_text())
    r = Report(f"Colab run vs expected  ({target})")

    env = got["environment"]
    r.info("environment", f"{env['gpu']} | torch {env['torch']} | cuda {env['cuda']} | ort {env['onnxruntime']}")
    r.check("deterministic kernels only", not got.get("deterministic_warnings"),
            "no warnings" if not got.get("deterministic_warnings") else str(got["deterministic_warnings"])[:80],
            required=False)

    for key, value in exp["exact"].items():
        actual = get(got, key)
        r.check(key, actual == value, f"{actual} (expected {value})")
    allowed = set(exp["npu_safe_operators"])
    r.check("fp32 operators NPU-safe", set(got["model"]["operators"]) <= allowed, str(got["model"]["operators"]))
    for key, spec in exp["tolerance"].items():
        actual = get(got, key)
        r.check(key, within(actual, spec["value"], spec["tol"]),
                f"{actual:.4f} vs {spec['value']:.4f} +/- {spec['tol']}  ({spec['why']})")
    for key, spec in exp["minimum"].items():
        actual = get(got, key)
        r.check(key, actual >= spec["value"], f"{actual} (need >= {spec['value']})")

    slope = got["extended_test"]["cost_model"]["coefs"]["bicubic_psnr"]
    r.check("ext-test cost slope negative", slope["ci_high"] < 0,
            f"{slope['estimate']:+.4f} dB/dB (95% CI {slope['ci_low']:+.4f} to {slope['ci_high']:+.4f}); "
            "dissertation 10-image fit -0.0444", required=False)
    r.info("v1 INT8 model, same protocol", f"gain {got['test']['v1_int8_gain']['mean']:+.3f} dB")

    ref_path = EXPECTED / "colab_reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        same_env = all(ref["environment"][k] == env[k] for k in ("gpu", "torch", "cuda", "onnxruntime"))
        r.check("same data as reference", got["data_manifest"] == ref["data_manifest"],
                "file hashes match" if got["data_manifest"] == ref["data_manifest"] else "DIFFERENT DATA")
        if same_env:
            for key in ("model.final_weights_sha256", "model.int8_onnx_sha256"):
                r.check(f"bit-identical: {key}", get(got, key) == get(ref, key),
                        f"{get(got, key)[:16]}… vs {get(ref, key)[:16]}…")
            for key in ("test.psnr_fp32", "test.psnr_int8"):
                r.check(f"identical: {key}", within(get(got, key), get(ref, key), 1e-6),
                        f"{get(got, key):.6f} vs {get(ref, key):.6f}")
        else:
            r.info("reference", "different GPU or library versions: bit-identical check skipped, "
                   "tolerance checks above apply")
            for key in ("test.psnr_fp32", "test.psnr_int8"):
                r.check(f"close to reference: {key}", within(get(got, key), get(ref, key), 0.05),
                        f"{get(got, key):.4f} vs {get(ref, key):.4f} +/- 0.05")
    else:
        r.info("reference", "none yet: commit this colab_summary.json as "
               "reproduce/expected/colab_reference.json to enable exact checks")
    return r.print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dissertation", "laptop", "colab"):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "dissertation":
        sys.exit(compare_dissertation())
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(compare_laptop(sys.argv[2]) if mode == "laptop" else compare_colab(sys.argv[2]))


if __name__ == "__main__":
    main()
