# One-time setup for the reproducible laptop run (Windows on ARM, Snapdragon X).
# Run from the repository root in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\reproduce\laptop\setup_laptop.ps1

$ErrorActionPreference = "Stop"

$arch = python -c "import platform; print(platform.machine())"
if ($arch -ne "ARM64") {
    Write-Host "!! Python is $arch, not ARM64. Install ARM64 Python from python.org;" -ForegroundColor Red
    Write-Host "   the QNN execution provider cannot load under x64 emulation." -ForegroundColor Red
    exit 1
}

python -m venv .venv-repro
.\.venv-repro\Scripts\Activate.ps1
python -m pip install --upgrade pip

# onnxruntime-qnn brings its own onnxruntime build; install it first and do
# not install plain onnxruntime afterwards, or it replaces the QNN build.
pip install onnxruntime-qnn
pip install -r reproduce\requirements.txt

# Record exactly what was installed, next to the results it produced.
pip freeze | Out-File -Encoding utf8 reproduce\laptop\pip-freeze.txt

python benchmark\scripts\qnn_setup.py
python benchmark\scripts\check_providers.py --model models\v1\espcn_x2_int8.onnx

Write-Host ""
Write-Host "Ready. Close other apps, plug in, set Best Performance, then run:" -ForegroundColor Green
Write-Host "  python reproduce\laptop\run_experiment_v2.py --preset dissertation" -ForegroundColor Green
