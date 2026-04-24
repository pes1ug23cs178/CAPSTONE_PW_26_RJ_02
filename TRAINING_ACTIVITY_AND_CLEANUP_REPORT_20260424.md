# Training Activity and Cleanup Report
Date: 2026-04-24
Workspace: C:/Users/Student3/Downloads/Deepak

## 1) Executive Summary
This workspace now has completed training outputs for both target datasets:
- CelebDF-v2 run: outputs_swin_v2
- FF++ run: outputs_ffpp_v2_20260423_172121

Both runs used the same Swin model family and tuning strategy (low LR + warmup + early stopping + leakage-safe split).

Current process status at report time:
- No active python process for train_swin_deepfake.py.

## 2) What Was Done (End-to-End)

### 2.1 Safety and Data-Leakage Controls
- Dataset root checks were run before launches.
- CUDA availability was verified in the CUDA venv.
- The training pipeline's leakage-safe split logic was used (detect_dataset_structure).
- Validation split mode used StratifiedGroupKFold and reported leakage=0 in run logs for both datasets.

Audit outcomes observed:
- CelebDF-v2:
  - total images: 208304
  - unique groups: 6518
  - train/val: 166649 / 41655
  - leakage: 0
- FF++:
  - total images: 104063
  - unique groups: 3285
  - train/val: 83251 / 20812
  - leakage: 0

### 2.2 Stability Bug Encountered and Fixed
A training crash occurred in DeepfakeDetector due to feature normalization shape:
- Error: Expected more than 1 spatial element when training, got input size [32, 1024, 1]

Fix applied:
- Changed feature normalization from InstanceNorm1d(feat_dim) over [B, F, 1]
- To InstanceNorm1d(1) over [B, 1, F]

Code locations updated:
- celebdf_v2/train_swin_deepfake.py (DeepfakeDetector init and forward)
- celebdf_v2/train_swin_deepfake.py changelog entry (#13)

Post-fix verification:
- CUDA smoke test passed (forward output shapes valid).
- Python compile check passed.
- No editor diagnostics remained in the file.

### 2.3 Training Launches Performed

CelebDF-v2 launch pattern:
- script: celebdf_v2/train_swin_deepfake.py
- data root: .\celebdf_v2\celebdf_v2\faces
- output dir: outputs_swin_v2
- key params: epochs=25, batch=32, lr=5e-5, warmup=4, early-stopping=5, ema=0.9999, require-cuda

FF++ launch pattern:
- script: celebdf_v2/train_swin_deepfake.py
- data root: .\ff++\faces
- output dir: outputs_ffpp_v2_20260423_172121
- key params: epochs=25, batch=32, lr=5e-5, warmup=4, early-stopping=5, ema=0.9999, require-cuda

### 2.4 Overnight Reliability Automation
- Added supervisor script: run_ffpp_overnight_supervisor.ps1
- Behavior implemented:
  - monitors active training process for specific output folder
  - re-runs leakage audit before each relaunch cycle
  - resumes from last_checkpoint.pth when available after interruption
  - exits on successful completion
- Additional safety patch added:
  - supervisor now checks for completion artifacts at loop start and exits without relaunch if run is already complete.

## 3) Final Results by Dataset

### 3.1 CelebDF-v2 (outputs_swin_v2)
Source files:
- outputs_swin_v2/evaluation_metrics.json
- outputs_swin_v2/training_history.json

| Metric | Value |
|---|---|
| Best epoch | 11 |
| Best validation accuracy | 0.997095 |
| Final validation accuracy | 0.997119 |
| Final validation loss | 0.334696 |
| Precision | 0.997311 |
| Recall | 0.999361 |
| F1-score | 0.998335 |
| ROC-AUC | 0.999891 |
| Epochs ran | 16 (early stopped) |
| Train samples | 166649 |
| Val samples | 41655 |
| Total train time (sec) | 14092.92 |
| Average epoch time (sec) | 880.81 |

Confusion matrix:
- [[5560, 97], [23, 35975]]

### 3.2 FF++ (outputs_ffpp_v2_20260423_172121)
Source files:
- outputs_ffpp_v2_20260423_172121/evaluation_metrics.json
- outputs_ffpp_v2_20260423_172121/training_history.json

| Metric | Value |
|---|---|
| Best epoch | 18 |
| Best validation accuracy | 0.972564 |
| Final validation accuracy | 0.971939 |
| Final validation loss | 0.261453 |
| Precision | 0.973298 |
| Recall | 0.979179 |
| F1-score | 0.976229 |
| ROC-AUC | 0.995091 |
| Epochs ran | 23 (early stopped) |
| Train samples | 83251 |
| Val samples | 20812 |
| Total train time (sec) | 10104.15 |
| Average epoch time (sec) | 439.31 |

Confusion matrix:
- [[8236, 329], [255, 11992]]

## 4) Artifacts Present

### 4.1 CelebDF-v2 output folder
- best_swin_deepfake.pth
- last_checkpoint.pth
- ckpt_epoch_005.pth
- ckpt_epoch_010.pth
- ckpt_epoch_015.pth
- evaluation_metrics.json
- training_history.json
- training_curves.png
- confusion_matrix.png
- val_predictions.csv
- gradcam/

### 4.2 FF++ output folder
- best_swin_deepfake.pth
- last_checkpoint.pth
- ckpt_epoch_005.pth
- ckpt_epoch_010.pth
- ckpt_epoch_015.pth
- ckpt_epoch_020.pth
- evaluation_metrics.json
- training_history.json
- training_curves.png
- confusion_matrix.png
- val_predictions.csv
- gradcam/

## 5) Directory Cleanup Performed
Cleanup policy used:
- conservative cleanup only (remove non-essential runtime clutter)
- preserve all models, checkpoints, metrics, curves, and predictions

Deleted:
- outputs_ffpp_v2_20260423_172121/overnight_supervisor.log
- all __pycache__ directories outside .venv-deepfake and .venv_cuda

Retained intentionally:
- run_ffpp_overnight_supervisor.ps1 (useful for future unattended reruns)
- all model artifacts and reports in outputs_swin_v2 and outputs_ffpp_v2_20260423_172121
- dataset directories and virtual environments

## 6) Direct Answer to User Question
Yes, the model pipeline was trained on both datasets and completed runs exist for both:
- CelebDF-v2: outputs_swin_v2
- FF++: outputs_ffpp_v2_20260423_172121

Both runs were executed with leakage-safe splitting and completed with saved metrics and artifacts.
