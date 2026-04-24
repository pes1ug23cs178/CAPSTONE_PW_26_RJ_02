Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Student3\Downloads\Deepak"
Set-Location -Path $ProjectRoot

$PythonExe = ".\.venv_cuda\Scripts\python.exe"
$TrainScript = ".\celebdf_v2\train_swin_deepfake.py"
$DataRoot = ".\ff++\faces"
$OutputDir = ".\outputs_ffpp_v2_20260423_172121"
$OutputDirName = Split-Path -Leaf $OutputDir
$LogFile = Join-Path $OutputDir "overnight_supervisor.log"

$MaxRestartAttempts = 100
$RestartCount = 0

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Get-RunProcesses {
    return Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match "train_swin_deepfake.py" -and
            $_.CommandLine -match $OutputDirName
        }
}

function Test-CompletionArtifacts {
    $metricsPath = Join-Path $OutputDir "evaluation_metrics.json"
    $bestPath = Join-Path $OutputDir "best_swin_deepfake.pth"
    $historyPath = Join-Path $OutputDir "training_history.json"
    $curvesPath = Join-Path $OutputDir "training_curves.png"
    $cmPath = Join-Path $OutputDir "confusion_matrix.png"

    return (
        (Test-Path $metricsPath) -and
        (Test-Path $bestPath) -and
        (Test-Path $historyPath) -and
        (Test-Path $curvesPath) -and
        (Test-Path $cmPath)
    )
}

function Invoke-DataLeakAudit {
    $auditCode = @"
from pathlib import Path
from celebdf_v2.train_swin_deepfake import detect_dataset_structure

bundle = detect_dataset_structure(
    [Path(r'.\\ff++\\faces').resolve()],
    val_split=0.2,
    seed=42,
    skip_corrupt=True,
)
print('AUDIT_OK')
print(f'TRAIN_SAMPLES={len(bundle.train_samples)}')
print(f'VAL_SAMPLES={len(bundle.val_samples)}')
print(f'NUM_DOMAINS={bundle.num_domains}')
"@

    $auditText = (& $PythonExe -c $auditCode 2>&1 | Out-String)
    Add-Content -Path $LogFile -Value $auditText

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Audit command failed with exit code $LASTEXITCODE"
        return $false
    }

    if ($auditText -notmatch "AUDIT_OK") {
        Write-Log "Audit did not report AUDIT_OK"
        return $false
    }

    Write-Log "Audit passed (split + leakage guard verified)."
    return $true
}

function Start-TrainingCycle {
    $args = @(
        $TrainScript,
        "--data-root", $DataRoot,
        "--output-dir", $OutputDir,
        "--model-name", "swin_base_patch4_window7_224",
        "--img-size", "224",
        "--epochs", "25",
        "--batch-size", "32",
        "--num-workers", "4",
        "--lr", "5e-5",
        "--min-lr", "1e-6",
        "--weight-decay", "1e-4",
        "--grad-clip", "1.0",
        "--val-split", "0.2",
        "--seed", "42",
        "--warmup-epochs", "4",
        "--early-stopping-patience", "5",
        "--early-stopping-min-delta", "1e-4",
        "--ema-decay", "0.9999",
        "--gradcam-samples", "8",
        "--save-every", "5",
        "--require-cuda"
    )

    $lastCheckpoint = Join-Path $OutputDir "last_checkpoint.pth"
    if (Test-Path $lastCheckpoint) {
        $args += @("--resume", $lastCheckpoint)
        Write-Log "Resuming from $lastCheckpoint"
    } else {
        Write-Log "No checkpoint found. Starting fresh in $OutputDir"
    }

    Write-Log "Launching training cycle"
    & $PythonExe @args 2>&1 | Tee-Object -FilePath $LogFile -Append
    return $LASTEXITCODE
}

Write-Log "FF++ overnight supervisor started."
Write-Log "Output directory: $OutputDir"

if (-not (Test-Path $DataRoot)) {
    Write-Log "Data root not found: $DataRoot"
    exit 1
}

while ($true) {
    if (Test-CompletionArtifacts) {
        Write-Log "Completion artifacts found. Training is already finished. Supervisor exiting."
        exit 0
    }

    $running = @(Get-RunProcesses)
    if ($running.Count -gt 0) {
        $pids = ($running | Select-Object -ExpandProperty ProcessId) -join ","
        Write-Log "Training process already active (PID(s): $pids). Monitoring..."
        Start-Sleep -Seconds 120
        continue
    }

    if (-not (Invoke-DataLeakAudit)) {
        $RestartCount += 1
        if ($RestartCount -ge $MaxRestartAttempts) {
            Write-Log "Max restart attempts reached during audit failures."
            exit 1
        }
        Write-Log "Retrying after audit failure in 90 seconds."
        Start-Sleep -Seconds 90
        continue
    }

    $exitCode = Start-TrainingCycle

    if ($exitCode -eq 0) {
        $metricsPath = Join-Path $OutputDir "evaluation_metrics.json"
        if (Test-Path $metricsPath) {
            Write-Log "Training completed successfully and metrics exist. Supervisor exiting."
            exit 0
        }

        Write-Log "Training exited with code 0 but metrics file missing. Will restart for safety."
    } else {
        Write-Log "Training crashed/stopped with exit code $exitCode. Preparing auto-resume."
    }

    $RestartCount += 1
    if ($RestartCount -ge $MaxRestartAttempts) {
        Write-Log "Max restart attempts reached. Supervisor exiting."
        exit 1
    }

    Write-Log "Restarting training in 90 seconds."
    Start-Sleep -Seconds 90
}
