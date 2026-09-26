# Concurrent NPU Offload Benchmark — Windows on ARM

Measures whether moving a neural upscaler from the GPU to the NPU returns
throughput to the workload competing for the GPU.

**The measured quantity is the renderer's throughput, not the upscaler's
latency.** Prior work established that upscalers run on NPUs. The open question
is what offloading does to the workload sharing the chip.

## Configurations

| | Renderer | Upscaler | Represents |
|---|---|---|---|
| A | GPU at 4K | none | Baseline |
| B | GPU at 1080p | GPU | Current practice (DLSS / FSR) |
| C | GPU at 1080p | NPU | The proposal |

The claim under test is C > B on renderer throughput.

## Order of operations

```powershell
# 1. Environment. ARM64 Python is required; x64 under emulation cannot load QNN.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib onnx
pip install onnxruntime-qnn

# Verify QNN registers (it is a plugin EP and needs explicit registration)
python scripts\qnn_setup.py

# 2. Build the renderer proxy (can be done on Colab and copied across)
python make_render_load.py --out render_load.onnx

# 3. VERIFY NPU PLACEMENT. Do not skip.
#    The NPU is int8-only: an fp32 model runs entirely on CPU while reporting success.
python scripts\check_providers.py --model espcn_x2_int8.onnx

# 4. The experiment
python scripts\benchmark_concurrent.py `
    --upscaler espcn_x2_int8.onnx `
    --load render_load.onnx `
    --duration 15 --repeats 5 --cooldown 20
```

## Step 3 is the one that decides whether anything else is valid

ONNX Runtime accepts a QNN provider request, silently falls back to CPU for
unsupported operators, and reports nothing. A benchmark built on that measures
CPU execution while labelling it NPU.

`check_providers.py` enables node-level profiling and reports which provider
executed each node. If operators fall back, fix the model before benchmarking.
Do not proceed with a partial fallback and mention it in a footnote.

## Controls

- **Randomised run order.** Thermal drift on a fanless laptop otherwise appears
  as a systematic effect favouring whichever configuration ran while cold.
- **Cooldown between runs**, default 20s. Increase it if the machine is passively
  cooled and results trend downward across a session.
- **Warmup excluded** from every measurement.
- **Repeats for variance.** The script prints run-to-run variation and says so
  explicitly when a difference falls inside the noise floor.
- **Plugged in, Best Performance power mode**, and no background applications.

## Reading the output

`concurrent_results.csv` has one row per run. The summary reports median
renderer throughput per configuration with standard deviation, and the
percentage change from offloading, next to the noise floor.

A change smaller than the noise floor is not a result. The script says so.

## Expected obstacles

**QNN provider will not load.** Three causes, in order of likelihood.

Zero: plugin execution providers are not selected through `providers=[...]`.
Passing `"QNNExecutionProvider"` there is silently ignored and you get a
CPU-only session with no error. Selection is by device, via
`SessionOptions.add_provider_for_devices()`. `scripts/qnn_setup.py` handles
this; `make_qnn_session()` raises rather than returning a CPU session.

Note the machine exposes QNN twice — once for the Hexagon NPU and once for the
Adreno GPU — so device type must be filtered as well as provider name. That
also means the GPU configuration can run through QNN rather than DirectML,
which is better experimental control: same runtime, same graph, only the
device changes.

Then:

First, QNN is a *plugin* execution provider as of onnxruntime-qnn 2.6.0, so it
does not appear in `get_available_providers()` until the plugin library is
registered explicitly. Most documentation still describes the older built-in
style. `scripts/qnn_setup.py` handles this, and both scripts call it
automatically — run it alone to check:

```powershell
python scripts\qnn_setup.py
```

Second, x64 Python under emulation cannot load QNN at all. Check with
`python -c "import platform; print(platform.machine())"` — it must say ARM64.

Confirm the hardware is visible to Windows with:

```powershell
Get-PnpDevice | Where-Object {$_.FriendlyName -match "neural|NPU|hexagon"}
```

**Most operators fall back to CPU.** Simplify the model: convolutions, ReLU and
pixel-shuffle only, fixed input shapes, opset 13.

**Offloading shows no gain.** A plausible outcome. Published work on mobile LLM
inference reports scheduling overhead of 8–22x operator time for lightweight
operators, and a 1.5x cross-backend fallback penalty. A three-convolution model
may spend more time being dispatched than computing. If so, sweep model size —
the crossover point is the finding.

**Unified memory is shared.** The NPU does not get free bandwidth; it competes
with the GPU for the same memory. Offload may move the bottleneck rather than
remove it.
