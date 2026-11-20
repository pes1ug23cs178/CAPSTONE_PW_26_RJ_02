# Start the GRACE app on a laptop with a public link.
#   powershell -ExecutionPolicy Bypass -File scripts\laptop\start_app.ps1            (public link + local page)
#   powershell -ExecutionPolicy Bypass -File scripts\laptop\start_app.ps1 -Local     (this laptop only, no internet link)
param([switch]$Local)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vpy = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "Run scripts\laptop\setup_laptop.ps1 first." -ForegroundColor Red; exit 1 }

$env:GRACE_MODEL_REPO = "grace-cap/grace-deepfake-models"   # models download from Hugging Face on first start
$env:GRACE_ONLINE = "1"                                     # public mode: 50 MB uploads, privacy note
$env:TORCHDYNAMO_DISABLE = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $Local) { $env:GRACE_SHARE = "1" }
$gpu = & $vpy -c "import torch; print(torch.cuda.is_available())"
if ($gpu -ne "True") { $env:GRACE_DEVICE = "cpu"; $env:GRACE_FRAMES = "16" } else { $env:GRACE_FRAMES = "32" }
$Host.UI.RawUI.WindowTitle = "GRACE app"

Write-Host ""
Write-Host "Starting GRACE (GPU: $gpu). First start downloads the models (~1.3 GB) - give it a few minutes." -ForegroundColor Cyan
Write-Host "Local page:  http://127.0.0.1:7860"
if (-not $Local) { Write-Host "Public link: look for 'Running on public URL: https://....gradio.live' below (valid up to 72 h)." }
Write-Host "Keep this window open. Close it (or press Ctrl+C) to stop the app."
Write-Host ""
& $vpy app/app.py
