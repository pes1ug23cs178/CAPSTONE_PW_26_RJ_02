# Start queue_runner detached from any terminal (survives closing VS Code). Usage: powershell -File scripts/start_queue.ps1
$repo = Split-Path -Parent $PSScriptRoot
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$repo\scripts\run_queue.cmd`""; CurrentDirectory = $repo }
"queue runner launched: ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
