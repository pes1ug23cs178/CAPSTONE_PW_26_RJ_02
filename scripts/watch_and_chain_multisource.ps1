<#
.SYNOPSIS
    Overnight multi-source baseline -> domain-adaptation training supervisor.

.DESCRIPTION
    Waits for an in-progress (or already-finished) multi-source baseline run,
    decides whether it's worth continuing that baseline further, then hands
    off to a domain-adaptation run — all unattended, with crash recovery.

    CHANGELOG (this revision) — each item ties back to a specific review
    point so the reasoning is traceable:

      1. Hidden window removed. Runs now use -NoNewWindow instead of
         -WindowStyle Hidden, so a crash isn't silently swallowed by a
         window nobody can see. Combined with #6 below, you get both: no
         intrusive window, AND a full transcript on disk either way.
      2. Before any --resume, the checkpoint file's existence is verified
         with a clear, actionable error instead of letting Python fail deep
         inside argument/torch.load handling.
      3. Same check applied to best_swin_deepfake.pth before the
         domain-adaptation phase launches.
      4. Should-ContinueBaseline no longer trusts metrics.json/history.json
         on their own. Test-RunHealth additionally verifies: the process
         actually exited 0 (when known), the metrics/history files were
         written AFTER this phase started (so stale leftovers from an
         earlier run in the same folder can't be mistaken for a fresh
         success), that either the full epoch target was reached or a
         legitimate early stop was recorded, and that validation loss is
         finite (no NaN/Inf).
      5. Metrics wait timeout raised from 30 minutes to a configurable
         default of 2 hours (-MetricsTimeoutMinutes).
      6. Every attempt's stdout/stderr is redirected to its own timestamped
         log file under logs/, regardless of window visibility.
      7. Retry logic: up to -MaxRetries attempts per phase. On a failed
         health check, the next attempt resumes from that phase's own
         last_checkpoint.pth. After the final attempt fails, a notification
         is logged (and emailed, if SMTP params are supplied).
      8. Before each attempt, Wait-ForFreeGpu polls `nvidia-smi` and waits
         (up to a timeout) for GPU memory to free up before launching,
         instead of assuming the GPU is idle.
      9. Optional -ShutdownWhenDone schedules a cancellable delayed
         `shutdown /s` once the whole pipeline finishes (success or not).

      BIGGEST RECOMMENDATION: -ContinueBaselineEpochs / -DomainAdaptEpochs
      (absolute, hard-coded epoch TOTALS) are replaced with
      -BaselineEpochIncrement / -DomainAdaptEpochIncrement. The actual
      --epochs value passed to train_swin_deepfake.py is now computed as
      (epochs already completed, read from the real metrics/history on
      disk) + (increment). A checkpoint at epoch 5 with increment=5 targets
      epoch 10; a checkpoint at epoch 8 targets epoch 13 — matching exactly
      regardless of where a previous run happened to stop.

      Bonus fix caught while implementing the above: the original script
      passed a small fixed --epochs value (e.g. 5) as an ABSOLUTE target to
      the domain-adapt phase while resuming from a baseline checkpoint whose
      stored epoch was already >= that number. train_swin_deepfake.py
      computes `range(start_epoch, args.epochs + 1)` from the checkpoint's
      saved epoch, so that range would have been empty and the
      domain-adaptation phase would have silently trained for ZERO epochs.
      Computing every phase's target as "current epoch + increment" (same
      fix as above) also eliminates this.

.PARAMETER TrainingPid
    PID of an already-running baseline process to wait on. 0 = auto-detect
    by matching -BaselineOutputDir against running command lines.

.PARAMETER BaselineOutputDir
    Output directory of the baseline run this supervisor attaches to.

.PARAMETER BaselineEpochIncrement
    How many additional epochs to train the baseline for, on top of however
    many it has already completed (see BIGGEST RECOMMENDATION above).

.PARAMETER DomainAdaptEpochIncrement
    Same idea, for the domain-adaptation phase (relative to the epoch count
    already stored in whatever checkpoint it resumes from).

.PARAMETER MetricsTimeoutMinutes
    Max time to wait for evaluation_metrics.json / training_history.json to
    appear before giving up and going straight to domain adaptation.

.PARAMETER MaxRetries
    Max attempts per phase before giving up on that phase.

.PARAMETER GpuMemThresholdMB
.PARAMETER GpuWaitTimeoutMinutes
    GPU-availability gate before each attempt (see Wait-ForFreeGpu).

.PARAMETER ShutdownWhenDone
    If set, schedules a delayed, cancellable OS shutdown once the pipeline
    finishes (success or failure).

.PARAMETER ShutdownDelaySeconds
    Grace period before the scheduled shutdown actually fires. Cancel with
    `shutdown /a` on the machine within this window.

.PARAMETER NotifyEmailTo / NotifyEmailFrom / NotifySmtpServer
    Optional. If all three are supplied, failure notifications are also
    emailed via Send-MailMessage. If any are missing, notifications are
    logged only (never fatal — email failures never abort the run).
#>

param(
    [int]$TrainingPid = 0,
    [string]$BaselineOutputDir = "outputs_multisource_baseline_15epochs",

    [Alias("ContinueBaselineEpochs")]
    [int]$BaselineEpochIncrement = 5,

    [Alias("DomainAdaptEpochs")]
    [int]$DomainAdaptEpochIncrement = 5,

    [int]$MetricsTimeoutMinutes = 120,
    [int]$MaxRetries = 3,

    [int]$GpuMemThresholdMB = 500,
    [int]$GpuWaitTimeoutMinutes = 30,

    [switch]$ShutdownWhenDone,
    [int]$ShutdownDelaySeconds = 300,

    [string]$NotifyEmailTo = "",
    [string]$NotifyEmailFrom = "",
    [string]$NotifySmtpServer = ""
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv_cuda\Scripts\python.exe"
$BaselineDir = Join-Path $RepoRoot $BaselineOutputDir
$MetricsPath = Join-Path $BaselineDir "evaluation_metrics.json"
$HistoryPath = Join-Path $BaselineDir "training_history.json"
$LastCheckpoint = Join-Path $BaselineDir "last_checkpoint.pth"
$BestCheckpoint = Join-Path $BaselineDir "best_swin_deepfake.pth"
$LogsDir = Join-Path $BaselineDir "logs"
$SupervisorLog = Join-Path $LogsDir "overnight_supervisor.log"

if (-not (Test-Path $PythonExe)) {
    throw "CUDA Python interpreter not found: $PythonExe"
}

if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# ===========================================================================
# Logging
# ===========================================================================

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "$stamp $Message"
    Add-Content -Path $SupervisorLog -Value $line
    Write-Host $line
}

# ===========================================================================
# Point 7 (partial): failure notification (log always, email if configured)
# ===========================================================================

function Send-FailureNotification {
    param(
        [string]$Subject,
        [string]$Body
    )
    Write-Log "NOTIFY: $Subject"
    Write-Log $Body
    if ($NotifyEmailTo -and $NotifyEmailFrom -and $NotifySmtpServer) {
        try {
            Send-MailMessage -To $NotifyEmailTo -From $NotifyEmailFrom -SmtpServer $NotifySmtpServer `
                -Subject $Subject -Body $Body -ErrorAction Stop
            Write-Log "Failure notification emailed to $NotifyEmailTo"
        } catch {
            Write-Log "Failed to send failure notification email: $_"
        }
    } else {
        Write-Log "Email notification not configured (NotifyEmailTo/NotifyEmailFrom/NotifySmtpServer not all set) - logged only."
    }
}

# ===========================================================================
# Process discovery (keeps the original auto-detect behavior, but also
# recovers the process StartTime and original CommandLine where possible,
# so Test-RunHealth can do a best-effort freshness/epoch check even for a
# run this supervisor did not itself launch).
# ===========================================================================

function Get-CommandLineProcessInfo {
    param([string]$MatchText)

    $escaped = [regex]::Escape($MatchText)
    $proc = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $escaped -and $_.Name -eq 'python.exe' } |
        Sort-Object CreationDate |
        Select-Object -Last 1

    if ($null -ne $proc) {
        return [pscustomobject]@{
            ProcessId   = [int]$proc.ProcessId
            StartTime   = $proc.CreationDate
            CommandLine = $proc.CommandLine
        }
    }

    return [pscustomobject]@{ ProcessId = 0; StartTime = $null; CommandLine = $null }
}

function Get-RequestedEpochsFromCommandLine {
    param([string]$CommandLine)
    if (-not $CommandLine) { return $null }
    if ($CommandLine -match '--epochs\s+(\d+)') {
        return [int]$Matches[1]
    }
    return $null
}

# ===========================================================================
# JSON helpers
# ===========================================================================

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try {
        return Get-Content -Raw $Path | ConvertFrom-Json
    } catch {
        Write-Log "Failed to parse JSON at $Path : $_"
        return $null
    }
}

function Convert-ToDoubleArray {
    param([object]$Value)
    if ($null -eq $Value) {
        return @()
    }
    return @($Value | ForEach-Object { [double]$_ })
}

function Test-AllFinite {
    param([double[]]$Values)
    foreach ($v in $Values) {
        if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) {
            return $false
        }
    }
    return $true
}

function Get-EpochsRunSoFar {
    param(
        [pscustomobject]$Metrics,
        [pscustomobject]$History
    )
    if ($null -ne $Metrics -and $null -ne $Metrics.epochs_ran) {
        return [int]$Metrics.epochs_ran
    }
    if ($null -ne $History -and $null -ne $History.train_loss) {
        return @($History.train_loss).Count
    }
    return 0
}

# ===========================================================================
# Point 4: a run is only trusted if it passes ALL of these, not just
# "the files happen to exist".
# ===========================================================================

function Test-RunHealth {
    param(
        [Nullable[datetime]]$PhaseStartTime,
        [Nullable[int]]$ExitCode,
        [string]$MetricsPath,
        [string]$HistoryPath,
        [Nullable[int]]$RequestedEpochs
    )

    if ($null -ne $ExitCode -and $ExitCode -ne 0) {
        Write-Log "Health check FAILED: process exit code = $ExitCode."
        return $false
    }
    if ($null -eq $ExitCode) {
        Write-Log "Health check NOTE: exit code unavailable (externally-tracked process) - skipping that check."
    }

    if (-not (Test-Path $MetricsPath) -or -not (Test-Path $HistoryPath)) {
        Write-Log "Health check FAILED: metrics/history file missing ($MetricsPath / $HistoryPath)."
        return $false
    }

    if ($null -ne $PhaseStartTime) {
        $metricsFresh = (Get-Item $MetricsPath).LastWriteTime -ge $PhaseStartTime
        $historyFresh = (Get-Item $HistoryPath).LastWriteTime -ge $PhaseStartTime
        if (-not $metricsFresh -or -not $historyFresh) {
            Write-Log ("Health check FAILED: metrics/history predate this phase's start ({0}) - " -f $PhaseStartTime) `
                + "stale leftovers from a previous run in the same folder, not produced by this attempt."
            return $false
        }
    } else {
        Write-Log "Health check NOTE: phase start time unknown - skipping stale-file check."
    }

    $Metrics = Read-JsonFile $MetricsPath
    $History = Read-JsonFile $HistoryPath
    if ($null -eq $Metrics -or $null -eq $History) {
        Write-Log "Health check FAILED: metrics/history could not be parsed as JSON."
        return $false
    }

    $epochsRan = Get-EpochsRunSoFar -Metrics $Metrics -History $History
    $stoppedEarly = [bool]$Metrics.stopped_early
    if ($null -ne $RequestedEpochs -and -not $stoppedEarly -and $epochsRan -lt $RequestedEpochs) {
        Write-Log "Health check FAILED: only $epochsRan/$RequestedEpochs epochs completed and stopped_early=false (looks like a crash/interruption, not a legitimate finish)."
        return $false
    }

    $valLosses = Convert-ToDoubleArray $History.val_loss
    if ($valLosses.Count -gt 0 -and -not (Test-AllFinite $valLosses)) {
        Write-Log "Health check FAILED: validation loss contains NaN/Infinity."
        return $false
    }

    Write-Log "Health check PASSED (epochs_ran=$epochsRan, stopped_early=$stoppedEarly, exit=$ExitCode)."
    return $true
}

function Should-ContinueBaseline {
    param(
        [pscustomobject]$Metrics,
        [pscustomobject]$History,
        [bool]$Healthy
    )

    if (-not $Healthy) {
        Write-Log "Decision: will NOT continue baseline further (failed health check)."
        return $false
    }
    if ($null -eq $Metrics -or $null -eq $History) {
        return $false
    }

    $valAucs = Convert-ToDoubleArray $History.val_auc
    if ($valAucs.Count -lt 5) {
        return $false
    }
    if (-not (Test-AllFinite $valAucs)) {
        Write-Log "Decision: will NOT continue baseline further (non-finite val_auc values)."
        return $false
    }

    $bestEpoch = [int]$Metrics.best_epoch
    $bestValAcc = [double]$Metrics.best_val_accuracy
    $rocAuc = [double]$Metrics.roc_auc
    $bestAuc = ($valAucs | Measure-Object -Maximum).Maximum
    $finalAuc = $valAucs[-1]

    if ($bestEpoch -ge 4 -and $bestValAcc -ge 0.60 -and $rocAuc -ge 0.70 -and $finalAuc -ge ($bestAuc - 0.01)) {
        return $true
    }

    return $false
}

# ===========================================================================
# Point 8: don't launch into a GPU someone/something else is still using.
# ===========================================================================

function Wait-ForFreeGpu {
    param(
        [int]$ThresholdMB = 500,
        [int]$TimeoutMinutes = 30
    )

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Log "nvidia-smi not found; skipping GPU availability check."
        return
    }

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($true) {
        $raw = $null
        try {
            $raw = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null
        } catch {
            Write-Log "nvidia-smi query failed ($_); proceeding without GPU check."
            return
        }

        # GPU index 0 specifically: train_swin_deepfake.py uses the default
        # CUDA device (torch.device('cuda') / get_device_properties(0)),
        # so that's the one we actually care about here.
        $line = @($raw) | Select-Object -First 1
        if (-not $line) {
            Write-Log "nvidia-smi returned no output; proceeding without GPU check."
            return
        }

        $parts = $line -split ',\s*'
        $usedMB = [double]$parts[0]
        $totalMB = [double]$parts[1]

        if ($usedMB -le $ThresholdMB) {
            Write-Log "GPU 0 free enough to proceed ($usedMB MB / $totalMB MB used, threshold $ThresholdMB MB)."
            return
        }
        if ((Get-Date) -ge $deadline) {
            Write-Log "GPU 0 still busy after $TimeoutMinutes min ($usedMB MB used) - proceeding anyway."
            return
        }
        Write-Log "GPU 0 busy ($usedMB MB used > threshold $ThresholdMB MB); waiting 30s before rechecking..."
        Start-Sleep -Seconds 30
    }
}

# ===========================================================================
# Points 1, 2/3, 6, 7 combined: launch one phase, with retries, redirected
# logs, no hidden window, and a checkpoint-existence guard before every
# attempt (not just the first one).
# ===========================================================================

function Invoke-TrainingPhase {
    param(
        [Parameter(Mandatory)] [string]$PhaseName,
        [Parameter(Mandatory)] [string[]]$BaseArguments,   # everything except --epochs/--resume
        [Parameter(Mandatory)] [string]$OutputDir,
        [Parameter(Mandatory)] [string]$InitialResumeCheckpoint,
        [Parameter(Mandatory)] [int]$RequestedEpochs,
        [int]$MaxRetries = 3,
        [int]$GpuThresholdMB = 500,
        [int]$GpuTimeoutMinutes = 30
    )

    $metricsPath = Join-Path $OutputDir "evaluation_metrics.json"
    $historyPath = Join-Path $OutputDir "training_history.json"
    $lastCkpt    = Join-Path $OutputDir "last_checkpoint.pth"

    $resumeFrom = $InitialResumeCheckpoint
    $exitCode = $null
    $stdoutLog = $null
    $stderrLog = $null

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {

        # Points 2/3: never launch --resume against a checkpoint that isn't there.
        if ($resumeFrom -and -not (Test-Path $resumeFrom)) {
            $msg = "[$PhaseName] Required checkpoint not found: $resumeFrom"
            Write-Log $msg
            throw $msg
        }

        # Point 8
        Wait-ForFreeGpu -ThresholdMB $GpuThresholdMB -TimeoutMinutes $GpuTimeoutMinutes

        $runArgs = $BaseArguments + @('--epochs', "$RequestedEpochs", '--resume', $resumeFrom)

        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $stdoutLog = Join-Path $LogsDir "$PhaseName.attempt$attempt.$stamp.stdout.log"
        $stderrLog = Join-Path $LogsDir "$PhaseName.attempt$attempt.$stamp.stderr.log"

        Write-Log "[$PhaseName] Attempt $attempt/$MaxRetries - target epochs=$RequestedEpochs, resume=$resumeFrom"
        Write-Log "[$PhaseName] Command: $PythonExe $($runArgs -join ' ')"
        Write-Log "[$PhaseName] stdout -> $stdoutLog"
        Write-Log "[$PhaseName] stderr -> $stderrLog"

        $phaseStart = Get-Date

        # Point 1: -NoNewWindow instead of -WindowStyle Hidden.
        # Point 6: always capture stdout/stderr to disk regardless.
        $proc = Start-Process -FilePath $PythonExe -ArgumentList $runArgs `
            -WorkingDirectory $RepoRoot -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

        Write-Log "[$PhaseName] Started PID $($proc.Id)"
        Wait-Process -Id $proc.Id
        $exitCode = $proc.ExitCode
        Write-Log "[$PhaseName] Attempt $attempt exited with code $exitCode"

        $healthy = Test-RunHealth -PhaseStartTime $phaseStart -ExitCode $exitCode `
            -MetricsPath $metricsPath -HistoryPath $historyPath -RequestedEpochs $RequestedEpochs

        if ($healthy) {
            Write-Log "[$PhaseName] Attempt $attempt completed successfully."
            return [pscustomobject]@{
                Success   = $true
                ExitCode  = $exitCode
                Attempts  = $attempt
                StdoutLog = $stdoutLog
                StderrLog = $stderrLog
            }
        }

        Write-Log "[$PhaseName] Attempt $attempt FAILED health check."

        if ($attempt -ge $MaxRetries) {
            break
        }

        # Point 7: retry from whatever checkpoint this attempt left behind.
        if (-not (Test-Path $lastCkpt)) {
            Write-Log "[$PhaseName] No checkpoint available to resume from after the failed attempt; cannot retry further."
            break
        }
        $resumeFrom = $lastCkpt
        Write-Log "[$PhaseName] Will retry from $resumeFrom ..."
    }

    Write-Log "[$PhaseName] All attempts exhausted without a healthy completion."
    return [pscustomobject]@{
        Success   = $false
        ExitCode  = $exitCode
        Attempts  = $MaxRetries
        StdoutLog = $stdoutLog
        StderrLog = $stderrLog
    }
}

# ===========================================================================
# Main
# ===========================================================================

$overallSuccess = $true

try {
    Write-Log "================================================================"
    Write-Log "Supervisor started for $BaselineOutputDir"
    Write-Log "BaselineEpochIncrement=$BaselineEpochIncrement DomainAdaptEpochIncrement=$DomainAdaptEpochIncrement MaxRetries=$MaxRetries MetricsTimeoutMinutes=$MetricsTimeoutMinutes"
    Write-Log "================================================================"

    # --- Locate / wait for the initial (externally-launched) baseline run ---
    $initialInfo = [pscustomobject]@{ ProcessId = 0; StartTime = $null; CommandLine = $null }
    if ($TrainingPid -gt 0) {
        try {
            $existing = Get-Process -Id $TrainingPid -ErrorAction Stop
            $initialInfo = [pscustomobject]@{
                ProcessId   = $TrainingPid
                StartTime   = $existing.StartTime
                CommandLine = $null
            }
        } catch {
            Write-Log "Provided -TrainingPid $TrainingPid is not a running process; will fall back to file-based waiting."
        }
    } else {
        $initialInfo = Get-CommandLineProcessInfo -MatchText $BaselineOutputDir
    }

    if ($initialInfo.ProcessId -gt 0) {
        Write-Log "Waiting for training PID $($initialInfo.ProcessId) to exit (started $($initialInfo.StartTime))"
        try {
            Wait-Process -Id $initialInfo.ProcessId -ErrorAction Stop
        } catch {
            Write-Log "PID $($initialInfo.ProcessId) already gone by the time we waited on it (probably finished already) - continuing."
        }
    } else {
        Write-Log "No live PID matched; waiting for checkpointed run to finish via output files instead."
    }

    # --- Point 5: wait for metrics/history, with a realistic timeout ---
    $deadline = (Get-Date).AddMinutes($MetricsTimeoutMinutes)
    while ((-not (Test-Path $MetricsPath) -or -not (Test-Path $HistoryPath)) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 15
    }

    $Metrics = Read-JsonFile $MetricsPath
    $History = Read-JsonFile $HistoryPath

    if ($null -eq $Metrics -or $null -eq $History) {
        Write-Log "Metrics were not generated within $MetricsTimeoutMinutes min; treating baseline as unhealthy and moving straight to domain adaptation."
        $initialHealthy = $false
    } else {
        $requestedEpochsGuess = Get-RequestedEpochsFromCommandLine -CommandLine $initialInfo.CommandLine
        $initialHealthy = Test-RunHealth -PhaseStartTime $initialInfo.StartTime -ExitCode $null `
            -MetricsPath $MetricsPath -HistoryPath $HistoryPath -RequestedEpochs $requestedEpochsGuess
    }

    $continueBaseline = Should-ContinueBaseline -Metrics $Metrics -History $History -Healthy $initialHealthy

    if ($continueBaseline) {
        # BIGGEST RECOMMENDATION: target = however many epochs are already
        # done, plus the increment - never a hard-coded absolute number.
        $epochsSoFar = Get-EpochsRunSoFar -Metrics $Metrics -History $History
        $baselineTarget = $epochsSoFar + $BaselineEpochIncrement
        Write-Log "Decision: continue baseline from $epochsSoFar -> $baselineTarget epochs (+$BaselineEpochIncrement)"

        $baselineBaseArgs = @(
            'train_swin_deepfake.py',
            '--multi-source',
            '--data-root', 'celebdf_v2/celebdf_v2', 'ff++',
            '--output-dir', $BaselineDir,
            '--batch-size', '16',
            '--lr', '1e-4',
            '--seed', '42',
            '--compression-aug',
            '--feature-norm', 'instance',
            '--early-stopping-metric', 'roc_auc',
            '--early-stopping-patience', '12',
            '--save-every', '1',
            '--require-cuda'
        )

        if (-not (Test-Path $LastCheckpoint)) {
            throw "Baseline continuation requested but checkpoint not found: $LastCheckpoint"
        }

        $baselineResult = Invoke-TrainingPhase -PhaseName "baseline" -BaseArguments $baselineBaseArgs `
            -OutputDir $BaselineDir -InitialResumeCheckpoint $LastCheckpoint `
            -RequestedEpochs $baselineTarget -MaxRetries $MaxRetries `
            -GpuThresholdMB $GpuMemThresholdMB -GpuTimeoutMinutes $GpuWaitTimeoutMinutes

        if (-not $baselineResult.Success) {
            $overallSuccess = $false
            Send-FailureNotification -Subject "Baseline continuation failed after $($baselineResult.Attempts) attempt(s)" `
                -Body "See $($baselineResult.StderrLog). Proceeding to domain adaptation anyway if a best checkpoint exists."
        }

        # Re-read: the continuation may have advanced epochs_ran further.
        $Metrics = Read-JsonFile $MetricsPath
        $History = Read-JsonFile $HistoryPath
    } else {
        Write-Log "Decision: do NOT continue baseline further; proceeding straight to domain adaptation."
    }

    # --- Point 3: verify the checkpoint domain-adapt depends on ---
    if (-not (Test-Path $BestCheckpoint)) {
        throw "Cannot start domain adaptation: best checkpoint not found ($BestCheckpoint)."
    }

    $epochsSoFarForDomainAdapt = Get-EpochsRunSoFar -Metrics $Metrics -History $History
    $domainAdaptTarget = $epochsSoFarForDomainAdapt + $DomainAdaptEpochIncrement
    Write-Log "Domain-adapt target epochs = $epochsSoFarForDomainAdapt (from baseline) + $DomainAdaptEpochIncrement = $domainAdaptTarget"
    Write-Log "(Bonus fix: this is computed relative to the checkpoint's own epoch count so the training loop's epoch range can never end up empty.)"

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $domainAdaptDir = Join-Path $RepoRoot "outputs_multisource_domain_adapt_$timestamp"
    New-Item -ItemType Directory -Path $domainAdaptDir -Force | Out-Null

    $domainAdaptBaseArgs = @(
        'train_swin_deepfake.py',
        '--multi-source',
        '--domain-adapt',
        '--data-root', 'celebdf_v2/celebdf_v2', 'ff++',
        '--output-dir', $domainAdaptDir,
        '--batch-size', '16',
        '--lr', '1e-4',
        '--seed', '42',
        '--compression-aug',
        '--feature-norm', 'instance',
        '--early-stopping-metric', 'roc_auc',
        '--early-stopping-patience', '12',
        '--save-every', '1',
        '--require-cuda'
    )

    $domainAdaptResult = Invoke-TrainingPhase -PhaseName "domain_adapt" -BaseArguments $domainAdaptBaseArgs `
        -OutputDir $domainAdaptDir -InitialResumeCheckpoint $BestCheckpoint `
        -RequestedEpochs $domainAdaptTarget -MaxRetries $MaxRetries `
        -GpuThresholdMB $GpuMemThresholdMB -GpuTimeoutMinutes $GpuWaitTimeoutMinutes

    if (-not $domainAdaptResult.Success) {
        $overallSuccess = $false
        Send-FailureNotification -Subject "Domain-adaptation phase failed after $($domainAdaptResult.Attempts) attempt(s)" `
            -Body "See $($domainAdaptResult.StderrLog)."
    } else {
        Write-Log "Domain-adaptation phase completed successfully -> $domainAdaptDir"
    }

} catch {
    $overallSuccess = $false
    Write-Log "FATAL: $_"
    Send-FailureNotification -Subject "Training supervisor crashed" -Body "$_"
} finally {
    Write-Log "Supervisor finished. Overall success = $overallSuccess"

    # Point 9: optional, cancellable, delayed shutdown - regardless of outcome.
    if ($ShutdownWhenDone) {
        Write-Log "ShutdownWhenDone set: shutting down in $ShutdownDelaySeconds seconds. Run 'shutdown /a' on this machine to cancel."
        try {
            & shutdown.exe /s /t $ShutdownDelaySeconds /c "Training supervisor finished (success=$overallSuccess); shutting down."
        } catch {
            Write-Log "Failed to schedule shutdown: $_"
        }
    }
}

if (-not $overallSuccess) {
    exit 1
}
exit 0