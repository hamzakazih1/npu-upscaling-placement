"""
Analyse one run of reproduce/laptop/run_experiment_v2.py.

Primary analysis, fixed in advance so that two people running it get the same
kind of answer:

  unit of analysis   one valid measurement window (run)
  response           renderer throughput = frames / page-clock seconds
  model              OLS  throughput ~ condition   (baseline = reference)
  decision           equivalence (TOST) of NPU vs baseline within +/-1%
  secondary          quasi-Poisson rate model on 0.5 s report intervals;
                     the v1 metric (rolling median), recomputed to show
                     whether the artefact appears on this machine too

Usage:
    python reproduce/analysis/analyse_laptop_run.py reproduce/results/laptop/<timestamp>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

MARGIN = 0.01


def legacy_metric(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the dissertation's v1 metric from the raw frame intervals: the
    median of the last 150 per-frame fps values at each report, then the
    median of those across the window.
    """
    df = intervals.sort_values(["seq", "i"]).reset_index(drop=True)
    fps = 1000.0 / df["interval_ms"].clip(lower=0.001)
    df["rolling_median"] = fps.rolling(150, min_periods=1).median()
    at_report = df.groupby("seq").tail(1)
    measured = at_report[at_report.phase == "measure"]
    return (measured.groupby(["run_id", "attempt"])["rolling_median"].median()
            .rename("legacy_metric").reset_index())


def report_rates(reports: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = set(zip(valid.run_id, valid.attempt))
    measured = reports[reports.phase == "measure"]
    for (run_id, attempt), g in measured.groupby(["run_id", "attempt"]):
        if (run_id, attempt) not in keys:
            continue
        g = g.sort_values("t_page_ms")
        frames = g.frames.diff().iloc[1:]
        seconds = g.t_page_ms.diff().iloc[1:] / 1000
        for f, s in zip(frames, seconds):
            rows.append({"run_id": run_id, "condition": g.condition.iloc[0],
                         "frames": f, "seconds": s})
    return pd.DataFrame(rows)


def main(folder) -> dict:
    folder = Path(folder)
    runs = pd.read_csv(folder / "runs.csv")
    valid = runs[runs.valid.astype(str) == "True"].copy()
    conditions = [c for c in ("baseline", "cpu", "npu") if c in set(valid.condition)]
    if "baseline" not in conditions:
        raise SystemExit("no valid baseline runs; nothing to compare against")

    n = len(valid)
    one = np.ones(n)
    others = [c for c in conditions if c != "baseline"]
    dummies = [(valid.condition == c).astype(float).to_numpy() for c in others]
    order = valid.order.to_numpy() / 10.0

    final = su.ols(valid.throughput_fps, [one, *dummies], ["intercept", *others])
    with_order = su.ols(valid.throughput_fps, [one, *dummies, order], ["intercept", *others, "order_per10"])
    null = su.ols(valid.throughput_fps, [one], ["intercept"])

    base = valid[valid.condition == "baseline"].throughput_fps
    out = {
        "folder": str(folder),
        "attempts": int(len(runs)), "valid_runs": int(n),
        "invalid_reasons": runs[runs.valid.astype(str) != "True"].reason.fillna("").tolist(),
        "canvas": sorted(valid.canvas.unique().tolist()),
        "iterations": sorted(int(v) for v in valid.iterations.unique()),
        "throughput_by_condition": {
            c: {"mean": float(g.throughput_fps.mean()), "sd": float(g.throughput_fps.std(ddof=1)),
                "n": int(len(g))}
            for c, g in valid.groupby("condition")},
        "selection": [
            {"step": 0, "model": "condition + run order", "aic": with_order["aic"]},
            {"step": 1, "dropped": "run order (drift)", "f_p": su.nested_f(final, with_order),
             "aic": final["aic"]},
            {"step": 2, "dropped": "condition", "f_p": su.nested_f(null, final), "aic": null["aic"]},
        ],
        "model": {"coefs": final["coefs"], "r2": final["r2"], "adj_r2": final["adj_r2"]},
        "diagnostics": su.residual_checks(final),
        "relative_to_baseline_pct": {
            c: {k: 100 * final["coefs"][c][k] / final["coefs"]["intercept"]["estimate"]
                for k in ("estimate", "ci_low", "ci_high")} for c in others},
        "equivalence": {
            c: su.tost_ratio(valid[valid.condition == c].throughput_fps, base, MARGIN) for c in others},
        "latency": {},
    }

    for c in others:
        g = valid[valid.condition == c]
        out["latency"][c] = {"median_ms": float(g.upscale_median_ms.median()),
                             "p90_ms": float(g.upscale_p90_ms.median()),
                             "iters_per_run": float(g.upscale_iters.median())}
    if {"cpu", "npu"} <= set(others):
        out["latency"]["cpu_over_npu"] = out["latency"]["cpu"]["median_ms"] / out["latency"]["npu"]["median_ms"]

    reports_path = folder / "reports.csv"
    intervals_path = folder / "frame_intervals.csv"
    if reports_path.exists():
        rates = report_rates(pd.read_csv(reports_path), valid)
        if len(rates):
            design = [np.ones(len(rates))] + [(rates.condition == c).astype(float) for c in others]
            out["poisson_glm"] = su.poisson_rate_glm(rates.frames, rates.seconds, design,
                                                     ["baseline_rate", *others])
    if intervals_path.exists():
        iv = pd.read_csv(intervals_path)
        steps = (1000.0 / iv.interval_ms.clip(lower=0.001)).round(1).value_counts().head(4)
        out["per_frame_fps_top_values"] = {str(k): int(v) for k, v in steps.items()}
        legacy = legacy_metric(iv).merge(valid[["run_id", "attempt", "condition"]], on=["run_id", "attempt"])
        if len(legacy) == n:
            d = [(legacy.condition == c).astype(float) for c in others]
            lm = su.ols(legacy.legacy_metric, [np.ones(len(legacy)), *d], ["intercept", *others])
            out["legacy_metric_model"] = lm["coefs"]

    out = clean(out)
    (folder / "summary.json").write_text(json.dumps(out, indent=2))
    draw(valid, folder)

    print("\nThroughput by condition (frames / s, valid runs)")
    for c, s in out["throughput_by_condition"].items():
        print(f"  {c:<9} {s['mean']:8.3f}  sd {s['sd']:.3f}  n {s['n']}")
    print()
    print(su.coefs_table(final))
    for c in others:
        e = out["equivalence"][c]
        r = out["relative_to_baseline_pct"][c]
        print(f"\n{c} vs baseline: {r['estimate']:+.2f}% (95% CI {r['ci_low']:+.2f}% to {r['ci_high']:+.2f}%)")
        print(f"  equivalent within +/-{MARGIN:.0%}: {'YES' if e['equivalent'] else 'no'} "
              f"(90% CI ratio {e['ci90_low']:.4f}-{e['ci90_high']:.4f}, p {e['p_tost']:.4f})")
    if "legacy_metric_model" in out:
        print("\nv1 metric on the same frames (for comparison only):")
        for c in others:
            v = out["legacy_metric_model"][c]
            print(f"  {c}: {v['estimate']:+.3f} fps (p {v['p']:.3g})")
    print(f"\nwrote {folder / 'summary.json'}")
    return out


def draw(valid, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colours = {"baseline": "#7f7f7f", "cpu": "#d62728", "npu": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    groups = [c for c in ("baseline", "cpu", "npu") if c in set(valid.condition)]
    axes[0].boxplot([valid[valid.condition == c].throughput_fps for c in groups], tick_labels=groups)
    axes[0].set(ylabel="frames / s", title="Renderer throughput by condition")
    for c in groups:
        g = valid[valid.condition == c]
        axes[1].scatter(g.order, g.throughput_fps, color=colours[c], label=c)
    axes[1].set(xlabel="run order", ylabel="frames / s", title="Throughput over the session (drift check)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "throughput.png", dpi=130)
    plt.close(fig)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
