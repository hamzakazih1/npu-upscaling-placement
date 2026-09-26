"""
Recompute every headline figure in this project from the saved result files.

The point is that no number in the README has to be taken on trust. Everything
below is derived from the CSVs in results/, not copied from a write-up.

Usage:
    python src/verify_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLACEMENT = ROOT / "results" / "placement"
V1 = ROOT / "results" / "v1"
V2 = ROOT / "results" / "v2"

VSYNC_HZ = (120, 60)


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * 68)


def placement() -> None:
    """The main experiment: does upscaler placement affect renderer frame rate?"""
    df = pd.read_csv(PLACEMENT / "fps_results.csv")
    raw = pd.read_csv(PLACEMENT / "fps_raw_samples.csv")

    rule("Renderer frame rate by condition")
    summary = df.groupby("condition").agg(
        fps=("fps_median", "median"),
        sd=("fps_median", "std"),
        runs=("fps_median", "size"),
    )
    print(summary.round(4).to_string())

    base = summary.loc["baseline", "fps"]
    print()
    for condition in summary.index:
        value = summary.loc[condition, "fps"]
        print(f"  {condition:<9} {value:7.3f} fps   {value/base:.4f}x baseline   "
              f"cost {(1 - value/base)*100:+.2f}%")

    npu, cpu = summary.loc["npu", "fps"], summary.loc["cpu", "fps"]
    print(f"\n  NPU vs CPU difference: {(npu/cpu - 1)*100:+.4f}%")
    print(f"  raw frame-rate samples: {len(raw)}")

    # The caveat that qualifies the null result (dissertation Section 5.1.3).
    rule("Vsync check on the loaded conditions")
    print(f"  loaded conditions sit at {npu:.3f} fps = {1000/npu:.2f} ms/frame")
    for hz in VSYNC_HZ:
        steps = {n: hz / n for n in range(1, 10)}
        near = [(n, f) for n, f in steps.items() if abs(f - npu) < 0.5]
        for n, f in near:
            print(f"  {hz} Hz / {n} = {f:.3f} fps = {1000/f:.2f} ms  "
                  f"<- loaded conditions are within {abs(f-npu):.3f} fps of this step")
    print("  Baseline varied across repeats "
          f"({', '.join(f'{v:.3f}' for v in sorted(df[df.condition=='baseline'].fps_median))}),"
          " which is not a step.")
    print("  The null result therefore cannot be separated from vsync pinning.")


def latency() -> None:
    """Upscaler latency, the unambiguous result."""
    df = pd.read_csv(PLACEMENT / "fps_results.csv")
    loaded = df[df.condition != "baseline"]

    rule("Upscaler latency")
    stats = loaded.groupby("condition").agg(
        ms=("upscale_median_ms", "median"),
        sd=("upscale_median_ms", "std"),
        inferences=("upscale_iters", "median"),
    )
    print(stats.round(3).to_string())

    npu_ms = stats.loc["npu", "ms"]
    cpu_ms = stats.loc["cpu", "ms"]
    print(f"\n  CPU / NPU latency ratio: {cpu_ms/npu_ms:.1f}x")
    print(f"  inference-count ratio:   {stats.loc['npu','inferences']/stats.loc['cpu','inferences']:.1f}x"
          "  (the conditions did not present equal competing work)")

    print("\n  Share of a frame budget:")
    for fps, budget in ((20, 50.0), (30, 1000/30), (60, 1000/60)):
        print(f"    {fps:>2} fps ({budget:5.1f} ms):  NPU {npu_ms/budget*100:6.1f}%   "
              f"CPU {cpu_ms/budget*100:7.1f}%")


def quality() -> None:
    """Reconstruction quality, and what quantization cost."""
    rule("Quality: first model")
    v1 = pd.read_csv(V1 / "eval_fp32.csv")
    print(f"  model {v1.model_psnr.mean():.3f} dB | bicubic {v1.bicubic_psnr.mean():.3f} dB "
          f"| gain {v1.gain.mean():+.3f} dB | beats bicubic {int((v1.gain>0).sum())}/{len(v1)}")
    print("  NOTE: these figures come from a re-run of the first notebook. The run")
    print("  reported in the dissertation scored 27.616 dB (gain -3.490 dB); its")
    print("  per-image CSV was not preserved. Both runs fail by a similar margin,")
    print("  which is the point: the failure was structural, not a bad draw.")

    rule("Quality: corrected model")
    fp32 = pd.read_csv(V2 / "eval_fp32_v2.csv")
    int8 = pd.read_csv(V2 / "eval_int8_v2.csv")
    for name, d in (("fp32", fp32), ("int8", int8)):
        print(f"  {name}    {d.model_psnr.mean():.3f} dB | bicubic {d.bicubic_psnr.mean():.3f} dB "
              f"| gain {d.gain.mean():+.3f} dB | beats bicubic {int((d.gain>0).sum())}/{len(d)}")

    cost = int8.model_psnr.mean() - fp32.model_psnr.mean()
    merged = fp32.merge(int8, on="name", suffixes=("_fp32", "_int8"))
    per_image = merged.model_psnr_int8 - merged.model_psnr_fp32
    r = np.corrcoef(merged.bicubic_psnr_fp32, per_image)[0, 1]
    print(f"\n  cost of INT8 quantization: {cost:+.3f} dB (sd {per_image.std():.3f})")
    print(f"  per-image cost ranges {per_image.min():+.3f} to {per_image.max():+.3f} dB")
    print(f"  correlation with bicubic PSNR: r = {r:.3f}")
    print("  (easier images lose more, because there is less headroom to absorb it)")


def eda() -> None:
    rule("Exploratory analysis of the training images")
    stats = pd.read_csv(V1 / "eda_image_stats.csv")
    r = stats.laplacian_var.corr(stats.bicubic_psnr)
    print(f"  {len(stats)} images | widths {stats.width.min()}-{stats.width.max()} "
          f"| heights {stats.height.min()}-{stats.height.max()}")
    print(f"  bicubic PSNR: mean {stats.bicubic_psnr.mean():.2f} dB, "
          f"range {stats.bicubic_psnr.min():.2f}-{stats.bicubic_psnr.max():.2f}")
    print(f"  correlation between detail and bicubic PSNR: r = {r:.3f}")
    print("  (more detail means harder to upscale, validating the difficulty proxy)")


def models() -> None:
    rule("Model metadata")
    for label, path in (("first", V1 / "model_meta.json"), ("corrected", V2 / "model_meta_v2.json")):
        meta = json.loads(path.read_text())
        print(f"  {label:<10} {meta['model']:<26} {meta['parameters']:>7,} params  "
              f"opset {meta['opset']}  ops {','.join(meta['operators'])}")


if __name__ == "__main__":
    print("Recomputing all headline figures from results/\n" + "=" * 68)
    placement()
    latency()
    quality()
    eda()
    models()
    print("\nDone. Compare against the tables in README.md.")
