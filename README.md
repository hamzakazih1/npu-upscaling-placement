# Can a Laptop NPU Take the Upscaling Work Off the GPU?

Measuring what happens to a renderer's frame rate when a neural upscaler is moved from the CPU to the idle NPU of a Snapdragon X Plus laptop — and an account of the five undocumented obstacles between a correctly specified machine and a single valid measurement.

MSc Artificial Intelligence dissertation, Brunel University London (CS5500).

---

## The question

DLSS, FSR and XeSS all improve frame rates by rendering at low resolution and upscaling the result. All of them run the upscaler on the GPU, which is already the bottleneck. Modern laptops contain an NPU that sits idle during graphics work.

So: does moving the upscaler to that idle silicon give the GPU its budget back?

**On this hardware, no.** But the measurement that says no is more interesting than it sounds, and the second measurement is unambiguous.

---

## Results

### The renderer does not care where the upscaler runs

Three conditions, three repeats each, randomised order, 631 frame-rate samples.

| Condition | Frame rate | SD across repeats | Relative to baseline |
|---|---:|---:|---:|
| baseline (GPU renders alone) | 20.576 fps | 0.088 | 1.000 |
| upscaler on NPU | 20.040 fps | 0.000 | 0.974 |
| upscaler on CPU | 20.040 fps | 0.000 | 0.974 |

Identical to three decimal places. Both cost the renderer 2.60%.

**This null result carries a caveat, and it is the honest headline.** Both loaded conditions sit at 20.040 fps, and 20 fps is itself a vsync divisor — 120/6, or 60/3 if the browser composites at 60 Hz. The measured 49.90 ms per frame is 0.1 ms off the 50.00 ms step, inside the measurement's own resolution. The baseline did vary across repeats (20.450, 20.576, 20.619), which is not a step, so the pipeline can report non-quantized values. But the possibility that adding either upscaler simply pushed the renderer onto a step, where it became pinned, cannot be excluded.

The correct reading is therefore: **no difference was observable at this measurement resolution, on an instrument that may have been quantized at exactly the value the loaded conditions occupied.** Resolving it needs GPU timer queries rather than frame rate.

### The upscaler latency is not ambiguous

| Backend | Median latency | Inferences in 20 s |
|---|---:|---:|
| Hexagon NPU | 1.667 ms | 7,067 |
| CPU | 109.241 ms | 205 |

A factor of **65.5**. Against frame budgets:

| Target | Budget | NPU | CPU |
|---|---:|---:|---:|
| 30 fps | 33.3 ms | 5.0% | 327.7% |
| 60 fps | 16.7 ms | 10.0% | 655.4% |

The NPU fits comfortably. The CPU cannot produce a frame in time under any target considered. And no GPU inference path exists on this platform at all (see below), so the GPU-resident arrangement DLSS uses could not be reproduced.

**What placement determines here is not headroom but feasibility.** The NPU is the only processor on the machine capable of running the workload inside a frame.

### Quantization cost

| Model | PSNR | Gain over bicubic | Beats bicubic |
|---|---:|---:|---:|
| fp32 | 33.250 dB | +1.162 dB | 10/10 |
| INT8 | 32.978 dB | +0.890 dB | 9/10 |
| bicubic | 32.088 dB | — | — |

INT8 costs **0.272 dB**. Per-image, the cost correlates at r = −0.888 with how easy the image already was: where bicubic is near-perfect there is no headroom to absorb quantization noise, which is why one image drops below the bicubic line.

---

## Five undocumented obstacles

This took longer than the experiment. Three of the five fail *silently* — they produce a working session, correct-looking output and no diagnostic, while running on hardware other than the one requested.

| # | Obstacle | Failure mode |
|---|---|---|
| 1 | QNN is a plugin EP and must be registered before it appears in `get_available_providers()` | Visible: provider missing on working hardware |
| 2 | Plugin EPs are **not** selected via `providers=[...]`; selection is by device through `add_provider_for_devices()` | **Silent**: CPU session reported as NPU |
| 3 | QNN is exposed once per backing device (NPU and GPU), so device *type* must be filtered too | **Silent**: wrong accelerator |
| 4 | The NPU is integer-only; an fp32 model runs entirely on CPU while reporting success | **Silent**: whole model on CPU |
| 5 | No GPU inference path exists (no ARM64 DirectML; QNN's GPU backend fails validation on quantized ops with error 3110 and crashes at runtime on fp32 with error 6999) | Visible / partial |

`benchmark/scripts/qnn_setup.py` handles 1 to 3 and **raises** rather than returning a CPU-only session. `benchmark/scripts/check_providers.py` verifies node placement before any timing is reported.

The practical lesson generalises: **placement verification belongs in any heterogeneous inference benchmark**, on the same footing as excluding warmup runs.

---

## Verify the numbers yourself

Nothing above has to be taken on trust:

```bash
pip install pandas numpy
python src/verify_results.py
```

It recomputes every figure in this README from the CSVs in `results/`, including the vsync check and the per-image quantization analysis.

---

## Layout

```
├── notebooks/
│   ├── 01_first_model.ipynb        EDA, first model, INT8 export (Colab)
│   └── 02_corrected_model.ipynb    corrected model with anchor residual
├── benchmark/                      Windows-on-ARM measurement suite
│   ├── gpu_load.html               WebGL renderer that reports its own frame rate
│   └── scripts/                    QNN setup, placement check, experiment runner
├── models/
│   ├── v1/espcn_x2_int8.onnx       the model used for the placement measurements
│   └── v2/                         corrected model, fp32 and INT8
├── results/
│   ├── placement/                  frame-rate results and 631 raw samples
│   ├── v1/, v2/                    evaluation, training history, metadata
│   └── figures/                    exploratory analysis figures
└── src/verify_results.py
```

---

## Method notes

**The renderer is a WebGL scene, not a game.** ONNX Runtime has no working GPU path on this machine, so the GPU workload could not be an inference model. A real WebGL scene reaches the Adreno through the graphics driver instead — a different code path that works, and closer to a real renderer competing for the GPU than an inference proxy would have been. It has no geometry pipeline and no CPU-side game logic, which is a stated limitation.

**The two loaded conditions did not present equal work.** Each ran its upscaler continuously, so the faster backend simply completed more: 7,067 inferences on the NPU against 205 on the CPU, a factor of 34.5. That weakens any claim the conditions were matched — and also means the NPU absorbed 34× more inference work for the same 2.60% renderer cost. Fixing the inference rate, one upscale per rendered frame, would be the cleaner design.

**Controls.** Randomised condition order against thermal drift, three repeats, 5 s settle and 15 s cooldown, warmup excluded, browser kept in the foreground.

---

## The first model failed, and that is documented

The first trained model ran perfectly on the NPU at 1.667 ms and was **3.49 dB worse than bicubic interpolation**, which is free. It executed correctly while being useless as an upscaler.

Two causes: no residual connection, so the network had to learn the entire upscaled image rather than a correction, and far too little training. The fix was an anchor residual — a fixed 1×1 convolution replicating the input into the channel dimension, added before depth-to-space, reproducing nearest-neighbour upsampling exactly — following the ABPN design referenced by Ignatov et al. (2022). The residual target has roughly five times lower variance than direct prediction.

The corrected notebook checks PSNR against bicubic every epoch and refuses to export a model that loses to it.

The placement and latency results were obtained with the first model and are unaffected: operator structure and arithmetic cost are essentially the same, and both models place identically on the NPU.

*Note on `results/v1/`: the CSVs there come from a re-run of the first notebook, which scored 27.663 dB against the 27.616 dB reported in the dissertation. The original per-image CSV was not preserved. The two runs fail by a similar margin, which supports the structural explanation. The re-run differs because `PatchDataset` samples crops with NumPy inside DataLoader workers, which do not inherit a deterministic seed — a partially-seeded pipeline, the same class of defect the experiment set out to catch.*

---

## Hardware

| | |
|---|---|
| Machine | Snapdragon X Plus laptop (X1P42100), Windows 11 25H2 |
| NPU | Qualcomm Hexagon, via ONNX Runtime QNN execution provider |
| GPU | Qualcomm Adreno, driven through WebGL |
| Memory | Unified — CPU, GPU and NPU share system RAM |
| Training | Google Colab T4 (also the discrete-memory control) |

Unified memory is why the question is worth asking here. On a discrete card, moving a 4K frame off the GPU costs more in PCIe transfer than the upscaling saves, which is why DLSS is GPU-resident. Here placement is free to vary.

---

## Licence

MIT for the code. DIV2K is used under its research terms. The ESPCN architecture follows Shi et al. (2016); the anchor residual follows the ABPN design referenced in the Mobile AI challenge reports (Ignatov et al., 2022).
