# Sprint status (17 to 23 Nov 2026)

## Currently running

- Queue runner: `scripts/start_queue.ps1` (detached; restart with `powershell -ExecutionPolicy Bypass -File scripts/start_queue.ps1`). State: `runs/_queue/state.json`, logs: `runs/_queue/<job>.log`, heartbeat: `runs/_queue/heartbeat.txt`.
- GPU job: `b1_swin_s42`, started 2026-11-17 11:51. Measured epoch time 12.0 min train + 1.3 min val (67 img/s) -> expected end ~14:15 including test evaluation.
- Next in queue: `landmarks_ffpp_real` (~1.6 h measured at 0.21 s/crop), `b2_swin_dann_s42` (~2.3 h), `old_ckpt_ffpp_v2_eval` (~15 min).

## Needs a human

0. A CARLA simulator (`CarlaUnreal-Win64-Shipping.exe`, another project on this PC) held the GPU at 99% until ~12:27 on 17 Nov; it has since closed and B1 went from 67 to ~198 img/s. **Please keep CARLA (or any other GPU program) closed this week**: with it open, GPU memory runs out and Windows silently swaps it to RAM, which stalled training for 10 minutes once.

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

- T1.4 done, 2026-11-17 12:25, `FaceFrameDataset` + `CellBalancedBatchSampler`; 100 batches give each (method, label) cell an equal share; frames paths raise
- T1.5 done, 2026-11-17 12:35, `evaluate.py`; random scores give video AUC 0.527 on celebdf_test; 200-crop subset runs in 28 s. Thresholded metrics are video-level with the FF++-val threshold
- T1.6 done, 2026-11-17 11:47, `train.py`; overfit-32 passes (bce < 0.05 at step 6); 1-epoch 200-step run finishes; reloaded best.pth reproduces trainer val AUC exactly (0.5788 vs 0.5788)
- T1.7 done, 2026-11-17 11:50, `queue_runner.py`; ordered dependency, failure -> dependent blocked, restart skips done jobs
- T1.8 done, 2026-11-17 12:10, B1/B2 configs, queue running B1 (epoch 1 val video AUC 0.992); old-checkpoint evaluation queued
- T2.1 done (code), 2026-11-17 12:20, landmark extraction: 64/64 test crops OK, check image shows points on eyes, nose, mouth and jaw; full run queued. Note: face-alignment's torch.compile needs Triton (missing on Windows) and silently failed every call; fixed with `TORCHDYNAMO_DISABLE=1`

## Decision rules applied

- (none yet)
