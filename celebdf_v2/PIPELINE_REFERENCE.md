# Deepfake Swin Pipeline Reference

## 1) Objective
This pipeline trains and evaluates a binary deepfake image classifier using a pretrained Swin Transformer on preprocessed face images only.

The pipeline is designed for:
- Automatic dataset structure detection
- Full-model fine-tuning (not head-only)
- GPU-first training with fail-fast CUDA checks
- Robust overnight execution with checkpoint resume
- Full evaluation and explainability outputs

## 2) Environment and Isolation
Project root:
- `C:\Users\Student3\Downloads\Deepak`

Isolated training environment:
- `.venv_cuda`

Why this is safe:
- Installs are inside `.venv_cuda` only
- Global Python environment is not modified
- Existing scripts/projects outside this folder are unaffected

## 3) Required Dependencies
Installed packages:
- torch
- torchvision
- timm
- scikit-learn
- matplotlib
- numpy
- opencv-python
- pillow

Install command:
```bash
pip install torch torchvision timm scikit-learn matplotlib numpy opencv-python pillow
```

Project requirements file:
- `celebdf_v2/requirements.txt`

## 4) Dataset Inputs
Expected dataset root:
- `celebdf_v2/celebdf_v2`

The loader supports three detection modes:
- Folder split mode: explicit `train`/`val` style directories
- CSV split mode: `splits/train_split.csv` + `splits/val_split.csv`
- Auto split mode: stratified split over discovered real/fake images

Detected classes:
- `real -> 0`
- `fake -> 1`

Current detected split in this workspace:
- Train samples: 275
- Val samples: 38
- Detection mode: CSV split

## 5) Model Details
Backbone:
- `swin_base_patch4_window7_224` (pretrained via timm)

Classification setup:
- Final head replaced for 2-class output
- Full fine-tuning enabled for all model parameters

Input resolution:
- 224 x 224 RGB

Normalization:
- ImageNet mean/std

## 6) Training Configuration
Loss:
- `CrossEntropyLoss` with class weights

Optimizer:
- `AdamW`

Learning rate schedule:
- `CosineAnnealingLR`

Mixed precision:
- Enabled automatically on CUDA

Default overnight settings:
- Epochs: 25
- Batch size primary: 32
- Batch size fallback on OOM: 16
- Workers: 4
- Save every epoch: enabled (`--save-every 1`)

## 7) Robustness and Resume Strategy
The trainer now writes:
- `last_checkpoint.pth` every epoch
- `best_swin_deepfake.pth` when validation accuracy improves
- Optional `checkpoint_epoch_XXX.pth` per epoch when save-every is enabled

Resume support:
- `--resume <checkpoint_path>`

Overnight launcher behavior:
- Verifies CUDA and exits if unavailable
- Starts with batch size 32 for high GPU usage
- Detects OOM and auto-retries with batch size 16
- Retries interrupted runs and resumes from `last_checkpoint.pth`
- Writes full logs to a timestamped log file
- Uses a fixed output directory (`outputs_swin_gpu_overnight`) so resume is automatic across restarts

## 8) Evaluation Outputs
Saved files in output directory:
- `best_swin_deepfake.pth`
- `last_checkpoint.pth`
- `evaluation_metrics.json`
- `training_history.json`
- `training_curves.png`
- `confusion_matrix.png`
- `val_predictions.csv`
- `gradcam/*.png`

Metrics reported:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Full classification report

## 9) Explainability
Grad-CAM is implemented for Swin Transformer:
- Captures activations/gradients from late Swin block
- Produces side-by-side visualizations:
  - original image
  - heatmap
  - overlay

Saved under:
- `output_dir/gradcam/`

## 10) Overnight Execution Plan
Launcher script:
- `celebdf_v2/run_overnight_gpu.ps1`

Run command:
```powershell
powershell -ExecutionPolicy Bypass -File .\celebdf_v2\run_overnight_gpu.ps1
```

What happens automatically:
1. Checks CUDA visibility in `.venv_cuda`
2. Starts training with CUDA required
3. Logs all output to a timestamped `.log`
4. Recovers from interruptions using `last_checkpoint.pth`
5. Completes final evaluation and saves all artifacts

Important behavior on rerun:
- If `outputs_swin_gpu_overnight/last_checkpoint.pth` already reached the target epochs, the launcher resumes, detects completion, and skips retraining.
- For a fresh new full run, delete or rename `outputs_swin_gpu_overnight` first.

## 11) Estimated Runtime on This Machine
GPU:
- NVIDIA RTX 4500 Ada Generation (24 GB)

Dataset size:
- 275 train / 38 val images

Measured runtime in this workspace:
- 25 epochs completed successfully
- Sum of epoch compute time: ~57.98 seconds
- Epoch 1 warm-up: ~19.95 seconds
- Typical later epochs: ~1.5 to 1.8 seconds each

Practical wall-clock expectation:
- Training (25 epochs): ~1 to 2 minutes
- Evaluation + Grad-CAM + plotting: ~1 to 3 minutes
- Total typical run time: ~2 to 5 minutes

If model files must be downloaded again (cold cache), add extra startup time.

One-time CUDA install time:
- ~25 to 35 minutes (already completed)

## 12) Key Files
- Trainer: `celebdf_v2/train_swin_deepfake.py`
- Overnight launcher: `celebdf_v2/run_overnight_gpu.ps1`
- Dependency list: `celebdf_v2/requirements.txt`
- This reference: `celebdf_v2/PIPELINE_REFERENCE.md`
