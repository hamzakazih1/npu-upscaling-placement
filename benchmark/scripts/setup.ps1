# Environment setup for Windows on ARM (Snapdragon X).
# Run in Windows Terminal (PowerShell). If script execution is blocked:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "=== Environment ===" -ForegroundColor Cyan
Write-Host "OS architecture : $env:PROCESSOR_ARCHITECTURE"
Write-Host "Processor       : $((Get-CimInstance Win32_Processor).Name)"

# ARM64 Python matters. An x64 Python runs under emulation and cannot load the
# QNN provider at all, which is the most common reason this setup fails.
$arch = python -c "import platform; print(platform.machine())"
Write-Host "Python arch     : $arch"
if ($arch -ne "ARM64") {
    Write-Host "!! Python is not ARM64. Install ARM64 Python from python.org." -ForegroundColor Red
    Write-Host "   The QNN execution provider will not load under emulation."
}

Write-Host "`n=== Virtual environment ===" -ForegroundColor Cyan
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "`n=== Packages ===" -ForegroundColor Cyan
pip install numpy pandas matplotlib onnx

# QNN reaches the Hexagon NPU and brings its own onnxruntime build.
# Do NOT also install onnxruntime-directml: there is no ARM64 DirectML, and
# the package overwrites the same onnxruntime module, removing QNN.
pip install onnxruntime-qnn

Write-Host "`n=== Available providers ===" -ForegroundColor Cyan
python -c "import onnxruntime as ort; print(ort.__version__); [print(' ', p) for p in ort.get_available_providers()]"

Write-Host "`nNext: python scripts\check_providers.py --model espcn_x2.onnx" -ForegroundColor Green
