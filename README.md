# Generalized Deepfake Detection Framework

This repository contains training code and automation scripts for generalized deepfake detection with a Swin Transformer backbone.

## Included in this repo
- Training pipeline script: celebdf_v2/train_swin_deepfake.py
- Dependency list: celebdf_v2/requirements.txt
- Pipeline reference: celebdf_v2/PIPELINE_REFERENCE.md
- Overnight supervisor script: run_ffpp_overnight_supervisor.ps1
- Run/report notes: TRAINING_ACTIVITY_AND_CLEANUP_REPORT_20260424.md

## Intentionally excluded
This repo excludes large/local artifacts to keep git history clean and pushable:
- Datasets (CelebDF-v2 and FF++)
- Training outputs and checkpoints
- Virtual environments
- Installers and archives
- Runtime logs and generated evaluation artifacts

## Setup
1. Create and activate a Python environment.
2. Install dependencies from celebdf_v2/requirements.txt.
3. Ensure dataset paths are available locally before training.

## Training
Use celebdf_v2/train_swin_deepfake.py with your desired data root and output directory.
