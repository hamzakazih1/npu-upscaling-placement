# Reproducing the results

This folder reruns every part of the project in a way that gives the same
answer each time, and checks the answer automatically. It also fixes the
measurement and analysis defects found when the dissertation data were
re-analysed (summarised below).

There are three parts. Each ends with the same command, `compare.py`, which
prints a PASS/FAIL table and finishes with **REPRODUCED** or **NOT REPRODUCED**.

| Part | Runs on | Time | "Same result" means |
|---|---|---|---|
| 1. Re-analysis of the saved data | any computer | 1 min | every number identical |
| 2. Model training and quality | Google Colab, T4 GPU | ~20 min | bit-identical weights on the same GPU and library versions; within stated tolerances otherwise |
| 3. Placement and latency | Snapdragon X laptop | ~30 min | the same conclusion, with every timing inside its tolerance |

Hardware timings never repeat to the last digit, so part 3 checks against
tolerances, not exact values. The tolerances are in
[`expected/laptop_expected.json`](expected/laptop_expected.json), with the
reason for each one.

---

## 1. Re-analysis (any computer)

```bash
pip install -r reproduce/requirements.txt
python reproduce/analysis/reanalyse_dissertation.py
python reproduce/analysis/compare.py dissertation
```

This recomputes the data-quality audit, the EDA, both linear models, the
Poisson GLM and the latency figures from the CSVs in `results/`. The output
must match [`expected/dissertation_reanalysis.json`](expected/dissertation_reanalysis.json)
exactly. Main results:

| Result | Value |
|---|---|
| Renderer throughput, NPU vs baseline | +0.29% (95% CI −0.28% to +0.86%) |
| Equivalent within ±1% (TOST) | yes, p = 0.015 |
| The dissertation's 2.6% "cost" | an artefact of the rolling-median metric (−0.508 fps on that metric, none on throughput) |
| NPU vs CPU upscaler latency | 1.667 ms vs 109.241 ms (65.5×) |
| INT8 cost vs image difficulty | −0.044 dB per dB of bicubic PSNR (95% CI −0.063 to −0.026) |

## 2. Model training (Colab)

1. Open [`colab/03_reproducible_model.ipynb`](colab/03_reproducible_model.ipynb) in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all.

The last cells run `compare.py colab` and download `export_v3.zip`. To make
your run the reference that later runs must match exactly, commit its
`colab_summary.json` as `reproduce/expected/colab_reference.json`.

Changes from `02_corrected_model.ipynb`: architecture, data and
hyper-parameters are unchanged. What changed:

- Patch sampling is a pure function of (seed, epoch, index), so DataLoader workers can no longer draw different patches on each run.
- Only deterministic CUDA kernels are used, with TF32 off.
- Per-epoch checks use validation images 0801–0810. The test images (0891–0900) are used once, at the end.
- The v1 INT8 model is scored with the same protocol, so v1 and v3 can finally be compared like for like.
- Library versions, data file hashes and weight hashes are saved with the results.

## 3. Placement and latency (Snapdragon X laptop)

From the repository root, in PowerShell, with ARM64 Python:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\reproduce\laptop\setup_laptop.ps1
python reproduce\laptop\run_experiment_v2.py --preset dissertation
python reproduce\analysis\compare.py laptop reproduce\results\laptop\<timestamp>
```

Before running: plug the laptop in, set Best Performance, close other apps,
and leave the Edge window that opens alone and in front. The runner launches
Edge with a clean profile and background throttling off.

`--preset dissertation` uses the dissertation's model (`models/v1/espcn_x2_int8.onnx`),
shader load (700 iterations) and canvas (1189×966), with 10 repeats per
condition. `--preset matched` also fixes the upscaler to 20 inferences per
second and turns vsync off. That is the cleaner design for a follow-up.

| Defect in `benchmark/` (v1) | Fix in `laptop/` (v2) |
|---|---|
| Metric was a rolling median of per-frame fps, which snaps to vsync steps | frames ÷ page-clock seconds over the window |
| 7 s rolling window but 5 s settle, so windows mixed conditions | no rolling window, and a 10 s settle |
| A 28 s page stall went unnoticed | any report gap over 1 s, a hidden page, or a second open tab makes the run invalid, and it is repeated |
| NPU session creation ate into the settle time | settle starts only after the upscaler is warm |
| Canvas followed the window size, and the load auto-calibrated | fixed canvas and fixed shader load |
| 3 repeats | 10 repeats |
| Raw samples had no condition labels | every report and every frame interval is labelled |
| No record of the environment | `env.json` records versions, EP devices, model hashes, power scheme, battery and refresh rate |

Each run writes `config.json`, `env.json`, `runs.csv` (one row per attempt,
with the reason if it was invalid), `reports.csv`, `frame_intervals.csv`,
`summary.json` and `throughput.png`.

## Files

```
reproduce/
├── README.md
├── requirements.txt
├── analysis/
│   ├── stats_utils.py              OLS, F-tests, residual checks, TOST, Poisson GLM (numpy + scipy)
│   ├── reanalyse_dissertation.py   part 1
│   ├── analyse_laptop_run.py       analysis of one part-3 run
│   └── compare.py                  the checker for all three parts
├── colab/
│   └── 03_reproducible_model.ipynb part 2
├── laptop/
│   ├── setup_laptop.ps1
│   ├── gpu_load_v2.html            fixed-load WebGL renderer that reports frames and intervals
│   └── run_experiment_v2.py        part 3
└── expected/
    ├── dissertation_reanalysis.json
    ├── colab_expected.json
    └── laptop_expected.json
```
