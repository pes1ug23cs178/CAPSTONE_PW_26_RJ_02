# Sprint status (17 to 23 Nov 2026)

## Currently running

- GPU: idle (code track only so far)

## Needs a human

1. **Celeb-DF `List_of_testing_videos.txt` is missing.** Copy it to `splits/official/List_of_testing_videos.txt` and run `python data/make_splits.py`; the Celeb-DF test switches from interim to official automatically. Until then `celebdf_test.csv` is an INTERIM set with the official composition (108 Celeb-real, 70 YouTube-real, 340 Celeb-synthesis, seed 0). It is leak-free because Celeb-DF is never used in training, but its numbers are not comparable with published results.
2. **Main manual tab** (GRACE architecture spec, repository layout) is not on this machine. Working from the runbook alone.

## Environment

- Windows 11 Pro (10.0.26200), Python 3.12.3 (system install), 32 CPU threads
- GPU: NVIDIA RTX 4500 Ada, 24 GB, driver 596.71
- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, timm 1.0.26
- Added: open_clip 4.2.2, peft 0.21.3, gradio 6.32.0, onnxruntime 1.32.0, faiss 1.16.0, face-alignment, pandas, pytest
- ffmpeg: NOT installed (needed on Day 4 for H.264 robustness)
- Disk: on 17 Nov, C: had 3 GB free. Freed ~244 GB: removed `frames/` images (never used by the sprint; JSON metadata kept), duplicate git-LFS copies of old checkpoints, old intermediate checkpoints (each old run's `best_swin_deepfake.pth` kept), preprocessing `.pkl` resume files and the Python installer. C: now has ~234 GB free. Nothing written to D:.

## Data limitations found

- FF++ on disk is c23 and partial: aligned crops exist for only 839 / 1000 Deepfakes, 134 / 294 Face2Face and 39 / 199 FaceSwap videos (frames were never extracted for the rest, so this predates the cleanup). NeuralTextures and FaceShifter are absent. FaceSwap train has 25 videos (776 crops), so the balanced sampler re-uses those crops heavily.
- Celeb-DF YouTube-real: 278 of 300 videos on disk.

## Split counts (T1.3), please confirm

See `splits/counts.md` for the full table.

| Split | Videos | Crops |
| --- | --- | --- |
| ffpp_train | 1,445 (719 real, 611 DF, 90 F2F, 25 FS) | 46,058 |
| ffpp_val | 286 (140 real, 114 DF, 24 F2F, 8 FS) | 9,152 |
| ffpp_test | 280 (140 real, 114 DF, 20 F2F, 6 FS) | 8,823 |
| dfd_test | 1,274 (363 real, 911 fake) | 40,030 |
| celebdf_test (interim) | 518 (178 real, 340 fake) | 16,535 |

## Task log

- T1.1 done, 2026-11-17 11:15, snapshot committed (`3d97c51`), branch `sprint`, environment recorded
- T1.2 done, 2026-11-17 12:05, package scaffold; Swin, DANN/GRL, MMD, EMA, Grad-CAM ported; `pytest tests/test_shapes.py` passes (logits 2x2, feature 2x1024)
- T1.3 done, 2026-11-17 12:20, official FF++ splits + DFD + interim Celeb-DF written; `pytest tests/test_splits.py` passes (no shared video or source ID, no frames paths, all files exist)

## Decision rules applied

- (none yet)
