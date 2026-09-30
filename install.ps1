# One-step install of npu-upscale on a Snapdragon X-series laptop
# (X, X Plus, X Elite, X2 Plus, X2 Elite) running Windows 11 on ARM.
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install.ps1
#
# Creates .venv next to this script, installs the package with the QNN
# runtime, and runs `npu-upscale doctor` to prove the NPU is reachable.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Processor : $((Get-CimInstance Win32_Processor).Name)"
Write-Host "OS arch   : $env:PROCESSOR_ARCHITECTURE"

# Windows PowerShell 5.1 compatible (no ternary operator).
$python = "python"
if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }
$arch = & $python -c "import platform; print(platform.machine())"
Write-Host "Python    : $arch"

if ($arch -ne "ARM64") {
    Write-Host ""
    Write-Host "!! This Python is $arch, running under emulation." -ForegroundColor Red
    Write-Host "   QNN cannot load in an emulated process, so the NPU is unreachable."
    Write-Host "   Install 'Windows installer (ARM64)' from https://www.python.org/downloads/windows/"
    Write-Host "   then run this script again (use: py -V:3.12-arm64 if several are installed)."
    exit 1
}

$npu = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -match "Hexagon|NPU|Neural" }
if ($npu) {
    $npu | ForEach-Object { Write-Host "NPU       : $($_.FriendlyName) [$($_.Status)]" }
} else {
    Write-Host "!! Windows lists no NPU device. Install the Qualcomm NPU driver via Windows Update." -ForegroundColor Yellow
}

& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
# onnxruntime-qnn is pulled in automatically on ARM64 Windows. Do not also
# install onnxruntime or onnxruntime-directml: they replace the same module.
& .\.venv\Scripts\python.exe -m pip install -e ".[video]"

Write-Host ""
& .\.venv\Scripts\npu-upscale.exe doctor
Write-Host ""
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1   then: npu-upscale image photo.png" -ForegroundColor Green
