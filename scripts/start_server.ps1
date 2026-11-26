# Start the GRACE web app (app/server.py) and, unless -Local, a Cloudflare tunnel for the public link.
#   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1                 quick tunnel: random https://*.trycloudflare.com link
#   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Tunnel grace   named tunnel: your fixed address (docs/WEB_APP.md)
#   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Local          this computer only, http://127.0.0.1:8000
# Options: -Hf  load the models from Hugging Face (laptops without runs/); -Port 8000.
# Close the window or press Ctrl+C to stop both. Logs: runs\_web\server_<date>.log
param([switch]$Local, [string]$Tunnel = "", [switch]$Hf, [int]$Port = 8000)
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$cf = Join-Path $repo "tools\cloudflared.exe"
if (-not $Local -and -not (Test-Path $cf)) {
    Write-Host "Downloading cloudflared (Cloudflare's tunnel program) into tools\ ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Join-Path $repo "tools") | Out-Null
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf
}

$env:HOST = "127.0.0.1"             # the tunnel connects locally; nothing listens on the network directly
$env:PORT = "$Port"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCHDYNAMO_DISABLE = "1"
if ($Hf) { $env:GRACE_MODEL_REPO = "grace-cap/grace-deepfake-models" }
if (-not $Local) {
    $env:GRACE_ONLINE = "1"           # public: no dataset faces shown, privacy note, smaller uploads
    if (-not $env:GRACE_MAX_MB) { $env:GRACE_MAX_MB = "95" }   # Cloudflare's free plan stops uploads above 100 MB
}
$gpu = & $py -c "import torch; print(torch.cuda.is_available())"
if ($gpu -ne "True") { $env:GRACE_FRAMES = "16"; $env:GRACE_MODES = "Quick"; $env:GRACE_WORKERS_QUICK = "1" }

$logs = Join-Path $repo "runs\_web"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs ("server_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))
$Host.UI.RawUI.WindowTitle = "GRACE web app"
Write-Host "Starting the GRACE server (GPU: $gpu) ... log: $log" -ForegroundColor Cyan
$server = Start-Process -FilePath $py -ArgumentList "-m", "app.server" -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

try {
    $ready = $false
    for ($i = 0; $i -lt 300 -and -not $server.HasExited; $i++) {
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            if ($h.warm_workers -ge 1) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { Write-Host "The server did not start. See $log.err" -ForegroundColor Red; exit 1 }
    Write-Host "Server ready: http://127.0.0.1:$Port" -ForegroundColor Green
    if ($Local) {
        Write-Host "Local only. Keep this window open; press Ctrl+C to stop."
        Wait-Process -Id $server.Id
    } elseif ($Tunnel) {
        Write-Host "Opening the named tunnel '$Tunnel' (your fixed address). Keep this window open." -ForegroundColor Cyan
        & $cf tunnel --no-autoupdate run $Tunnel
    } else {
        Write-Host "Opening a quick tunnel: the public link is the https://....trycloudflare.com line below." -ForegroundColor Cyan
        Write-Host "(It changes each time this script starts. For a fixed address use -Tunnel, see docs\WEB_APP.md.)"
        & $cf tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
    }
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Write-Host "Stopped."
}
