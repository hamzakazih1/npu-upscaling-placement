# Results

| Path | Contents |
|---|---|
| `placement/fps_results.csv` | One row per condition per repeat: frame rate, upscaler latency, inference count |
| `placement/fps_raw_samples.csv` | 631 individual frame-rate samples posted by the WebGL page |
| `v1/eval_fp32.csv` | Per-image PSNR, first model (see the note below) |
| `v1/eda_image_stats.csv` | Per-image statistics for the 90 training images |
| `v1/t4_control_benchmark.csv` | Discrete-GPU control measurement |
| `v2/eval_fp32_v2.csv`, `v2/eval_int8_v2.csv` | Per-image PSNR, corrected model |
| `v2/training_history.csv` | Loss and bicubic gain per epoch |
| `figures/` | Exploratory analysis figures |

`python src/verify_results.py` recomputes every headline figure from these files.

**Note on `v1/`.** These come from a re-run of the first notebook, which scored
27.663 dB against the 27.616 dB reported in the dissertation; the original
per-image CSV was not preserved. The models differ because `PatchDataset`
samples crops with NumPy inside DataLoader workers, which do not inherit a
deterministic seed despite `SEED = 42` being set elsewhere. Both runs fail
against bicubic by a similar margin.
