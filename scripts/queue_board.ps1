# Live board of the queue: done / running / queued in order (read-only), refreshes every 10 s.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\queue_board.ps1     (Ctrl+C to close; the queue keeps running)
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$Host.UI.RawUI.WindowTitle = "GRACE queue board"
while ($true) {
    $out = & python (Join-Path $repo "scripts\queue_board.py") 2>&1 | Out-String
    Clear-Host
    Write-Host $out
    Start-Sleep -Seconds 10
}
