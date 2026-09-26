# Running the experiment

## Why it is built this way

ONNX Runtime has no working GPU path on this machine. DirectML has no ARM64
build, and QNN's Adreno backend rejects quantized convolutions at validation
(`backendValidateOpConfig` error 3110) and crashes at runtime on fp32 ones
(`QNN graph execute error, code 6999`).

So the GPU workload is a real WebGL scene in the browser, which reaches the
Adreno through the graphics driver — a different path that works. The page
measures its own frame rate and posts it to a local server.

This is closer to the real situation than an inference proxy: an actual
renderer competing for an actual GPU, which is what a game plus an upscaler
creates.

## Conditions

| | GPU | Upscaler | Question |
|---|---|---|---|
| baseline | WebGL | none | How fast with the GPU to itself? |
| cpu | WebGL | CPU | What does upscaling on the CPU cost? |
| npu | WebGL | Hexagon NPU | Does the NPU cost less? |

The measured quantity is **the renderer's frame rate**.

## Steps

```powershell
cd C:\Users\hamza\Downloads\npu-offload-benchmark\scripts
.\.venv\Scripts\Activate.ps1

python run_experiment.py --upscaler espcn_x2_int8.onnx
```

It opens `gpu_load.html` in your browser automatically. Then:

1. **Leave the browser window in the foreground and visible.** Browsers throttle
   `requestAnimationFrame` in background tabs to a few frames per second, which
   would destroy the measurement.
2. Wait for calibration — about six seconds while the page tunes its shader load
   to land near 30 fps, leaving headroom below the vsync ceiling so contention is
   visible.
3. Leave the machine alone for the rest of the run. Roughly 9 conditions at
   ~40 s each plus cooldowns, so about 10 minutes.

Keep it plugged in, on Best Performance, and not on a soft surface.

## Reading the output

`fps_results.csv` has one row per condition per repeat. `fps_raw_samples.csv`
has every individual sample, useful for plotting frame rate over time.

The summary prints median frame rate per condition, each as a ratio of
baseline, and the NPU-versus-CPU difference next to the run-to-run noise. A
difference smaller than the noise floor is not a result, and the script says so.

## If something goes wrong

**"no calibrated samples received"** — the browser window was not visible, or
the page failed to load. Open `gpu_load.html` manually and check the HUD says
"reporting" rather than "server not reachable".

**Frame rate pinned at 60** — the shader load is too light for this GPU to drop
below vsync. Raise `TARGET_FPS` handling by editing `TARGET_FPS` down to 20 in
`gpu_load.html`, which forces heavier per-pixel work.

**NPU condition errors** — run `python check_providers.py --model
espcn_x2_int8.onnx` and confirm nodes still land on QNN.
