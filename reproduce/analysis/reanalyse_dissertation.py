"""
Re-analyse the dissertation's saved data, following the four-section standard
(data quality, EDA, linear models, count GLM).

Everything here is computed from files already in results/, so it reproduces
exactly on any machine: no hardware, no GPU, no randomness.

Usage (from the repository root):
    python reproduce/analysis/reanalyse_dissertation.py
    python reproduce/analysis/compare.py dissertation

Outputs go to reproduce/results/dissertation/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "reproduce" / "results" / "dissertation"

DURATION = 20.0   # run_experiment.py default --duration used in the dissertation
COOLDOWN = 15.0   # run_experiment.py default --cooldown


# ---------------------------------------------------------------- Section 1

def reconstruct_windows(raw: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """
    Recover each run's 20 s measurement window from the unlabelled raw samples.

    run_experiment.py kept a sample when start <= received <= end. For each
    run, in execution order, scan candidate start times in 10 ms steps and take
    the first window whose sample count, median and mean of `mean_fps` all
    match fps_results.csv to the three decimals it was saved with.
    """
    received = raw["received"].to_numpy()
    metric = raw["mean_fps"].to_numpy()
    earliest = received[0] - 30.0
    windows = []
    for _, run in runs.iterrows():
        found = None
        for start in np.arange(earliest, earliest + 80.0, 0.01):
            mask = (received >= start) & (received <= start + DURATION)
            if mask.sum() != run.fps_samples:
                continue
            if (abs(round(float(np.median(metric[mask])), 3) - run.fps_median) < 1e-9
                    and abs(round(float(metric[mask].mean()), 3) - run.fps_mean) < 1e-9):
                found = float(start)
                break
        if found is None:
            raise RuntimeError(f"could not reconstruct window for run {run.to_dict()}")
        windows.append({"condition": run.condition, "repeat": int(run["repeat"]),
                        "start": found, "end": found + DURATION})
        earliest = found + DURATION + COOLDOWN
    return pd.DataFrame(windows)


def data_quality(raw, runs, windows):
    raw = raw.copy()
    raw["dt"] = raw["t"].diff()
    gaps = raw[raw["dt"] > 2.0]
    t0 = raw["received"].iloc[0]
    labels = []
    for value in raw["received"]:
        hit = windows[(windows.start <= value) & (windows.end >= value)]
        labels.append(f"{hit.condition.iloc[0]}|{hit['repeat'].iloc[0]}" if len(hit) else "between")
    raw["run"] = labels
    raw["condition"] = raw["run"].str.split("|").str[0]

    meta = json.loads((RESULTS / "v2" / "model_meta_v2.json").read_text())
    nb = (ROOT / "notebooks" / "02_corrected_model.ipynb").read_text()
    v1 = pd.read_csv(RESULTS / "v1" / "eval_fp32.csv")
    v2 = pd.read_csv(RESULTS / "v2" / "eval_fp32_v2.csv")
    bic = v1.merge(v2, on="name", suffixes=("_v1", "_v2"))
    bic_diff = bic["bicubic_psnr_v2"] - bic["bicubic_psnr_v1"]
    control = pd.read_csv(RESULTS / "v1" / "t4_control_benchmark.csv")

    dq = {
        "raw_rows": int(len(raw)),
        "in_window_rows": int((raw.run != "between").sum()),
        "between_rows": int((raw.run == "between").sum()),
        "missing_values": int(raw.isna().sum().sum() - raw["dt"].isna().sum()),
        "duplicate_rows": int(raw.drop(columns=["dt"]).duplicated().sum()),
        "duplicate_timestamps": int(raw["t"].duplicated().sum()),
        "frame_counter_decreases": int((raw["frames"].diff() < 0).sum()),
        "calibrating_true": int(raw["calibrating"].astype(str).eq("True").sum()),
        "canvas": f"{int(raw.width.iloc[0])}x{int(raw.height.iloc[0])}",
        "iterations_values": sorted(int(v) for v in raw["iterations"].unique()),
        "report_gaps_over_2s": [
            {"start_s": round(float(r.received - r.dt - t0), 2), "length_s": round(float(r.dt), 3)}
            for r in gaps.itertuples()],
        "runs_sample_counts": {f"{r.condition}|{r.repeat}": int(r.fps_samples)
                               for r in runs.itertuples()},
        "meta_training_images": int(meta["training_images"]),
        "notebook_trains_on_first_120": "train_images[:120]" in nb,
        "v1_v2_bicubic_diff_mean_db": float(bic_diff.mean()),
        "v1_v2_bicubic_diff_max_db": float(bic_diff.max()),
        "t4_control_provider": str(control["provider"].iloc[0]),
    }
    return raw, dq


# ---------------------------------------------------------------- Section 2

def window_throughput(raw, windows):
    rows = []
    t0 = raw["received"].iloc[0]
    for order, w in enumerate(windows.itertuples(), start=1):
        d = raw[(raw.received >= w.start) & (raw.received <= w.end)].sort_values("t")
        rows.append({
            "order": order, "condition": w.condition, "repeat": w.repeat,
            "elapsed_s": float(w.start - t0),
            "frames": int(d.frames.iloc[-1] - d.frames.iloc[0]),
            "seconds": float(d.t.iloc[-1] - d.t.iloc[0]),
            "reported_median_metric": float(np.median(d.mean_fps)),
            "n_reports": int(len(d)),
        })
    out = pd.DataFrame(rows)
    out["throughput_fps"] = out.frames / out.seconds
    return out


def eda(raw, W, figures: bool):
    steps = raw["fps"].round(1).value_counts()
    share_24 = raw.groupby("condition")["fps"].apply(lambda s: float((s > 22).mean())).to_dict()
    result = {
        "per_frame_fps_top_values": {str(k): int(v) for k, v in steps.head(4).items()},
        "share_frames_at_24fps_step": share_24,
        "throughput_by_condition": {
            c: {"mean": float(g.throughput_fps.mean()), "sd": float(g.throughput_fps.std())}
            for c, g in W.groupby("condition")},
        "reported_metric_by_condition": {
            c: {"mean": float(g.reported_median_metric.mean()),
                "sd": float(g.reported_median_metric.std())}
            for c, g in W.groupby("condition")},
    }
    if figures:
        draw_eda(raw, W)
    return result


def draw_eda(raw, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"baseline": "#7f7f7f", "cpu": "#d62728", "npu": "#1f77b4"}
    t0 = raw["received"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(raw.fps, bins=np.arange(14, 28, 0.25), color="#555")
    for n in (5, 6, 7):
        ax.axvline(120 / n, ls="--", c="r", lw=0.8)
    ax.set(xlabel="instantaneous fps of one frame", ylabel="count",
           title="Per-frame fps sits on 120 Hz vsync steps")
    fig.tight_layout(); fig.savefig(OUT / "fig1_vsync_steps.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    for w in W.itertuples():
        ax.axvspan(w.elapsed_s, w.elapsed_s + DURATION, color=colours[w.condition], alpha=0.25)
    ax.plot(raw.received - t0, raw.mean_fps, "k-", lw=0.8, label="reported metric (rolling median)")
    thr = (raw.frames.diff() / raw.t.diff()).rolling(10, center=True).mean()
    ax.plot(raw.received - t0, thr, "g-", lw=0.8, label="frames / time")
    ax.set(xlabel="seconds", ylabel="fps", ylim=(19.5, 22.5),
           title="Reported metric vs true throughput (grey baseline, red CPU, blue NPU)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig2_timeline.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- Section 3

def model_throughput(W):
    n = len(W)
    one = np.ones(n)
    cpu = (W.condition == "cpu").astype(float).to_numpy()
    npu = (W.condition == "npu").astype(float).to_numpy()
    elapsed = W.elapsed_s.to_numpy() / 60
    rep1 = (W["repeat"] == 1).astype(float).to_numpy()
    rep2 = (W["repeat"] == 2).astype(float).to_numpy()

    full = su.ols(W.throughput_fps, [one, cpu, npu, elapsed], ["intercept", "cpu", "npu", "elapsed_min"])
    block = su.ols(W.throughput_fps, [one, cpu, npu, rep1, rep2], ["intercept", "cpu", "npu", "rep1", "rep2"])
    final = su.ols(W.throughput_fps, [one, cpu, npu], ["intercept", "cpu", "npu"])
    null = su.ols(W.throughput_fps, [one], ["intercept"])
    on_metric = su.ols(W.reported_median_metric, [one, cpu, npu], ["intercept", "cpu", "npu"])

    keep = ~((W.condition == "cpu") & (W["repeat"] == 1)).to_numpy()
    Wk = W[keep]
    sens = su.ols(Wk.throughput_fps,
                  [np.ones(keep.sum()), (Wk.condition == "cpu").astype(float),
                   (Wk.condition == "npu").astype(float)], ["intercept", "cpu", "npu"])

    base = W[W.condition == "baseline"].throughput_fps
    return {
        "selection": [
            {"step": 0, "model": "condition + elapsed", "aic": full["aic"]},
            {"step": 1, "model": "condition", "dropped": "elapsed",
             "f_p": su.nested_f(final, full), "aic": final["aic"]},
            {"step": "1b", "model": "condition + repeat block",
             "f_p": su.nested_f(final, block), "aic": block["aic"]},
            {"step": 2, "model": "intercept only", "dropped": "condition",
             "f_p": su.nested_f(null, final), "aic": null["aic"]},
        ],
        "corr_elapsed_rep2": float(np.corrcoef(elapsed, rep2)[0, 1]),
        "final": {"coefs": final["coefs"], "r2": final["r2"], "adj_r2": final["adj_r2"]},
        "diagnostics": su.residual_checks(final),
        "cooks_by_run": {f"{c}|{r}": float(d) for c, r, d in zip(W.condition, W["repeat"], final["cooks"])},
        "sensitivity_without_stalled_run": sens["coefs"],
        "same_model_on_reported_metric": on_metric["coefs"],
        "npu_vs_baseline_pct": {
            k: 100 * final["coefs"]["npu"][k] / final["coefs"]["intercept"]["estimate"]
            for k in ("estimate", "ci_low", "ci_high")},
        "tost_npu_vs_baseline_1pct": su.tost_ratio(W[W.condition == "npu"].throughput_fps, base, 0.01),
        "tost_cpu_vs_baseline_1pct": su.tost_ratio(W[W.condition == "cpu"].throughput_fps, base, 0.01),
        "_final_model": final,
    }


def model_quantisation():
    fp32 = pd.read_csv(RESULTS / "v2" / "eval_fp32_v2.csv")
    int8 = pd.read_csv(RESULTS / "v2" / "eval_int8_v2.csv")
    q = fp32.merge(int8, on="name", suffixes=("_fp32", "_int8"))
    q["cost"] = q.model_psnr_int8 - q.model_psnr_fp32
    one = np.ones(len(q))
    x = q.bicubic_psnr_fp32
    full = su.ols(q.cost, [one, x, q.gain_fp32], ["intercept", "bicubic_psnr", "fp32_gain"])
    final = su.ols(q.cost, [one, x], ["intercept", "bicubic_psnr"])
    null = su.ols(q.cost, [one], ["intercept"])
    quad = su.ols(q.cost, [one, x, x ** 2], ["intercept", "bicubic_psnr", "bicubic_psnr_sq"])
    worst = int(np.argmax(final["cooks"]))
    keep = np.arange(len(q)) != worst
    sens = su.ols(q.cost[keep], [one[keep], x[keep]], ["intercept", "bicubic_psnr"])
    return {
        "corr_bicubic_fp32psnr": float(np.corrcoef(x, q.model_psnr_fp32)[0, 1]),
        "vif_bicubic_gain": float(1 / (1 - np.corrcoef(x, q.gain_fp32)[0, 1] ** 2)),
        "selection": [
            {"step": 0, "model": "bicubic + fp32 gain", "aic": full["aic"]},
            {"step": 1, "dropped": "fp32_gain", "f_p": su.nested_f(final, full), "aic": final["aic"]},
            {"step": 2, "dropped": "bicubic_psnr", "f_p": su.nested_f(null, final)},
        ],
        "final": {"coefs": final["coefs"], "r2": final["r2"], "adj_r2": final["adj_r2"]},
        "diagnostics": su.residual_checks(final),
        "quadratic_term_f_p": su.nested_f(final, quad),
        "most_influential_image": str(q.name.iloc[worst]),
        "sensitivity_without_it": {"coefs": sens["coefs"], "r2": sens["r2"]},
        "spearman": float(stats.spearmanr(q.cost, x)[0]),
        "fp32_gain": su.paired_mean_ci(fp32.gain),
        "int8_gain": su.paired_mean_ci(int8.gain),
        "int8_cost": su.paired_mean_ci(q.cost),
        "int8_beats_bicubic": int((int8.gain > 0).sum()),
        "wilcoxon_int8_gain_p": float(stats.wilcoxon(int8.gain).pvalue),
    }


# ---------------------------------------------------------------- Section 4

def glm_frames(raw):
    iv = raw[raw.run != "between"].copy()
    iv["prev_run"] = raw["run"].shift().loc[iv.index]
    iv = iv[(iv.run == iv.prev_run) & (iv.dt < 2.0)]
    frames = iv["frames"].to_numpy() - raw["frames"].shift().loc[iv.index].to_numpy()
    design = [np.ones(len(iv)), (iv.condition == "cpu").astype(float),
              (iv.condition == "npu").astype(float)]
    out = su.poisson_rate_glm(frames, iv.dt.to_numpy(), design, ["baseline_rate", "cpu", "npu"])
    out["frames_per_interval_counts"] = {str(int(k)): int(v)
                                         for k, v in pd.Series(frames).value_counts().items()}
    return out


def latency():
    df = pd.read_csv(RESULTS / "placement" / "fps_results.csv")
    loaded = df[df.condition != "baseline"]
    med = loaded.groupby("condition").upscale_median_ms.median()
    iters = loaded.groupby("condition").upscale_iters.median()
    return {"npu_median_ms": float(med["npu"]), "cpu_median_ms": float(med["cpu"]),
            "cpu_over_npu": float(med["cpu"] / med["npu"]),
            "npu_iters_median": float(iters["npu"]), "cpu_iters_median": float(iters["cpu"])}


def clean(obj):
    """Make the result JSON-serialisable and drop private keys."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main(figures: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RESULTS / "placement" / "fps_raw_samples.csv")
    runs = pd.read_csv(RESULTS / "placement" / "fps_results.csv")

    windows = reconstruct_windows(raw, runs)
    raw, dq = data_quality(raw, runs, windows)
    W = window_throughput(raw, windows)
    W.to_csv(OUT / "window_throughput.csv", index=False)

    summary = {
        "section1_data_quality": dq,
        "section2_eda": eda(raw, W, figures),
        "section3_throughput_model": model_throughput(W),
        "section3_quantisation_model": model_quantisation(),
        "section4_poisson_glm": glm_frames(raw),
        "latency": latency(),
    }
    summary = clean(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["section3_throughput_model"]
    print("Re-analysis of the dissertation data")
    print("=" * 66)
    print(W[["order", "condition", "repeat", "throughput_fps", "reported_median_metric"]]
          .round(3).to_string(index=False))
    print(f"\nNPU vs baseline throughput: {m['npu_vs_baseline_pct']['estimate']:+.2f}% "
          f"(95% CI {m['npu_vs_baseline_pct']['ci_low']:+.2f}% to "
          f"{m['npu_vs_baseline_pct']['ci_high']:+.2f}%)")
    t = m["tost_npu_vs_baseline_1pct"]
    print(f"Equivalence within +/-1%: {'yes' if t['equivalent'] else 'no'} "
          f"(90% CI of ratio {t['ci90_low']:.4f} to {t['ci90_high']:.4f}, p = {t['p_tost']:.4f})")
    r = m["same_model_on_reported_metric"]["npu"]
    print(f"Same model on the dissertation's metric: {r['estimate']:+.3f} fps (p = {r['p']:.1e})"
          "  <- the artefact")
    q = summary["section3_quantisation_model"]["final"]["coefs"]["bicubic_psnr"]
    print(f"INT8 cost slope: {q['estimate']:+.4f} dB per dB bicubic "
          f"({q['ci_low']:+.4f} to {q['ci_high']:+.4f})")
    print(f"\nwrote {OUT.relative_to(ROOT)}/summary.json")
    return summary


if __name__ == "__main__":
    main(figures="--no-figures" not in sys.argv)
