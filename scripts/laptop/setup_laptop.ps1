# One-time setup of the GRACE app on a Windows laptop (needs Python 3.12 and internet).
# Run from the repository folder:  powershell -ExecutionPolicy Bypass -File scripts\laptop\setup_laptop.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Write-Host "Python not found. Install Python 3.12 from python.org (tick 'Add to PATH') and run this again." -ForegroundColor Red; exit 1 }
$ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($ver -ne "3.12") { Write-Host "Python $ver found; Python 3.12 is recommended (other versions may fail to install packages)." -ForegroundColor Yellow }

if (-not (Test-Path ".venv")) { Write-Host "Creating virtual environment .venv ..."; & python -m venv .venv }
$vpy = Join-Path $repo ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

# GPU or CPU build of PyTorch
$gpu = $false
try { & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object { Write-Host "NVIDIA GPU found: $_"; $gpu = $true } } catch { }
if ($gpu) {
    Write-Host "Installing PyTorch with CUDA (GPU) ..."
    & $vpy -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU found: installing the CPU build of PyTorch (Quick check fast, Careful check about 1 minute per video)."
    & $vpy -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
}
& $vpy -m pip install -r requirements-app.txt

Write-Host "Checking the install ..."
& $vpy -c "import torch, gradio, onnxruntime, face_alignment; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available()); print('gradio', gradio.__version__)"
Write-Host "Setup done. Start the app with: powershell -ExecutionPolicy Bypass -File scripts\laptop\start_app.ps1" -ForegroundColor Green
