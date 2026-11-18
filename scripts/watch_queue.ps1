# Live view of the background queue (read-only): refreshes every 5 s.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\watch_queue.ps1     (Ctrl+C to close; the queue keeps running)
$repo = Split-Path -Parent $PSScriptRoot
$q = Join-Path $repo "runs\_queue"
while ($true) {
    $state = Get-Content (Join-Path $q "state.json") -Raw | ConvertFrom-Json
    $running = @($state.PSObject.Properties | Where-Object { $_.Value.status -eq "running" } | ForEach-Object { $_.Name })
    $done = @($state.PSObject.Properties | Where-Object { $_.Value.status -eq "done" }).Count
    $hb = (Get-Date) - (Get-Item (Join-Path $q "heartbeat.txt")).LastWriteTime
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>$null
    Clear-Host
    Write-Host ("GRACE queue  {0:yyyy-MM-dd HH:mm:ss}   done: {1}   running: {2}   heartbeat: {3:N0}s ago   GPU: {4}" -f (Get-Date), $done, ($running -join ", "), $hb.TotalSeconds, $gpu) -ForegroundColor Cyan
    Write-Host "---- queue (runner.log) ----" -ForegroundColor Yellow
    Get-Content (Join-Path $q "runner.log") -Tail 5
    foreach ($job in $running) {
        Write-Host "---- $job.log ----" -ForegroundColor Yellow
        Get-Content (Join-Path $q "$job.log") -Tail 25 | Where-Object { $_ -notmatch "Warning|NOTE: Redirects|warnings.warn" }
    }
    Start-Sleep -Seconds 5
}
