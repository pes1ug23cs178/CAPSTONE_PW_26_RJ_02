# Sprint status (17 to 23 Nov 2026)

## Currently running

- Queue runner: `scripts/start_queue.ps1` (detached; restart with `powershell -ExecutionPolicy Bypass -File scripts/start_queue.ps1`). State: `runs/_queue/state.json`, logs: `runs/_queue/<job>.log`, heartbeat: `runs/_queue/heartbeat.txt`.
- GPU job: `c2_clip_sbi_s42` (started 17:11, expected end ~19:00). Then B2 r2, B2-fake r2, G1, G2 DCT, G2-fft, G3, A3, robustness, students, explanations, app check, tables; then fillers (see queue.yaml).
- B1 epoch time with the GPU to itself: ~4 min (198 img/s).

- Extra work queued after the main pipeline (user asked to use spare GPU time): runbook fillers G3 seed 2, S3 seed 2, **D2** (G3 + unsupervised adaptation to 5,978 unlabelled non-test Celeb-DF videos; dataset-level DANN), B1 seed 1; then **X1** (G3 trained on FF++ + DFD, tested on FF++ and Celeb-DF), then a results refresh (eval, tables, summary, release).

## Needs a human

0. A CARLA simulator (`CarlaUnreal-Win64-Shipping.exe`, another project on this PC) held the GPU at 99% until ~12:27 on 17 Nov; it has since closed and B1 went from 67 to ~198 img/s. **Please keep CARLA (or any other GPU program) closed this week**: with it open, GPU memory runs out and Windows silently swaps it to RAM, which stalled training for 10 minutes once.

1. **Celeb-DF `List_of_testing_videos.txt` is missing.** Copy it to `splits/official/List_of_testing_videos.txt` and run `python data/make_splits.py`; the Celeb-DF test switches from interim to official automatically. Until then `celebdf_test.csv` is an INTERIM set with the official composition (108 Celeb-real, 70 YouTube-real, 340 Celeb-synthesis, seed 0). It is leak-free because Celeb-DF is never used in training, but its numbers are not comparable with published results.
2. **Main manual tab** (GRACE architecture spec, repository layout) is not on this machine. Working from the runbook alone.

## Environment

- Windows 11 Pro (10.0.26200), Python 3.12.3 (system install), 32 CPU threads
- GPU: NVIDIA RTX 4500 Ada, 24 GB, driver 596.71
- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8. **timm 1.0.31** (installing open_clip upgraded it from 1.0.26; every sprint run uses 1.0.31; kept to avoid breaking open_clip)
- Added (pinned in `requirements.txt`): open_clip 4.2.2, peft 0.21.3, gradio 6.32.0, onnxruntime 1.32.0, onnxscript 0.7.2, faiss 1.16.0, face-alignment 1.5.0, imageio-ffmpeg 0.8.1, pandas, pytest
- ffmpeg: no system install; the bundled `imageio-ffmpeg` binary (ffmpeg 9.0.2, libx264) is used for H.264
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
- T2.1 done, 2026-11-17 13:07, full landmark run: 0 failures / 27,417 FF++ train+val real crops (34 min); `cache/landmarks_check.png` shows points on eyes, brows, nose, mouth, jaw
- T3.1 done, 2026-11-17 14:03, fixed SBI sets: sbi_val and sbi_test 2,000 images each from 140 videos; regions full face 818-837, mouth 507-510, eyes 364-385, nose/cheeks 287-292; each transform in 476-1,060 images. FF++ test real landmarks: 0 / 4,400 failures
- T2.2 done (code), 2026-11-17 12:15, CLIP ViT-L/14 + LoRA: 1.573 M LoRA params; cls L2-normalized; patches 1024x16x16; batch 32 peak 7.5 GB (no grad checkpointing needed); trainable-only checkpoints 12.6 MB, reload exact
- T2.4 done, 2026-11-17 12:25, SBI engine: 204 unit tests pass; `cache/sbi_check.png` drawn (**a human should look at it**); loader with SBI 784 img/s
- T2.3/T2.5 queued (C1, C2)
- T3.1-T3.4 code done, 2026-11-17 12:30: concept head (top-10% mean), decision layer (w=1, r zero-init), DCT (constant image -> DC only, orthonormal) and FFT branches in float32, gated fusion; 7 unit tests pass; G1/G2/G2-fft configs; smoke run of G2 passed on the GPU
- T4.1 code done: domain head on z (D=5, MMD between fake domains); `scripts/make_g3_configs.py` picks G3's branch by FF++ **validation** AUC and writes G3 + A3 configs
- T4.2 done: `explain/video_cues.py` (frame timeline, concept flicker, CLIP identity drift) wired into evaluate; constant sequence -> 0
- T4.3 code done: JPEG / downscale / H.264 (libx264 veryfast, cached per video) degradations, `scripts/eval_all.py` with robustness figures
- T4.4 code done: `scripts/plot_tsne.py`
- T5.1-T5.4 code done: SBI pool (40k), teacher cache (+ 1e-3 check), EfficientNet-B0 student (4.71 M params, 0.39 GFLOPs), distillation losses (S0/S2/S3 tested), ONNX export (opset 18: the legacy exporter mis-exported EfficientNet by ~2 logits; the torch.export exporter matches to 2e-6), benchmark, agreement
- T6.1-T6.4 code done: concept maps, contributions, intervention (all 10 off leaves b + r(z): test passes), Grad-CAM (Swin), Grad-CAM++ (student), attention rollout (CLIP), FAISS retrieval, calibration, explanation figure + metrics, Gradio app. App face crops: SFD detector + eye-level rotation, margin 1.02x the box (measured on 40 training crops; the runbook's 1.3x fallback is much looser than our training crops)
- T7.2/T7.4 prep: `make_tables.py`, `make_release.py`, README, `docs/interfaces.md`, pinned `requirements.txt`; all queued after the training runs

## Results so far (test video AUC; thresholds from FF++ val)

| Run | FF++ test | DFD | Celeb-DF (interim) | Val video AUC (best epoch) | GPU h |
| --- | --- | --- | --- | --- | --- |
| B1 Swin-B `b1_swin_s42_20261117-1307` | 0.9948 | 0.9104 | 0.8992 | 0.9942 (8) | 0.78 |
| C1 CLIP + LoRA `c1_clip_lora_s42_20261117-1530` | 0.9996 | 0.9425 | 0.9362 | 1.0000 (3) | 1.68 |
| ~~B2 (collapsed run, superseded by r2)~~ | 0.9987 | 0.9140 | 0.8696 | 0.9988 (2) | 0.92 |

- B1 checks: in-domain 0.9948 (< 0.995 alarm), cross-dataset max 0.910 (< 0.95 alarm). Celeb-DF fakes vs Celeb-real only: 0.911; vs YouTube-real only: 0.881, so no single real group drives the number. Celeb-DF fakes get low probabilities (mean 0.26), so accuracy at the FF++ threshold is 0.71 despite AUC 0.90: a calibration shift across datasets.
- C1 vs B1: cross-dataset video AUC DFD +0.032, Celeb-DF +0.037; in-domain 0.9996 (> 0.995 alarm; same splits and checks as the B2 alarm, which found no leakage; frozen CLIP + LoRA is expected to be near-perfect in-domain). Celeb-DF threshold accuracy 0.65 (same calibration shift as B1).
- **B2 collapsed** after epoch 3: val video AUC 0.999 (ep 2) -> 0.06 (ep 4, inverted) -> 0.500 (ep 8-9, constant output). Its best checkpoint is epoch 2 (GRL alpha 0.76 by then), so its test numbers are essentially an early-stopped B1 and do not show a working DANN. See decision 14:25.
- **Leakage alarm (rule: in-domain test AUC > 0.995) for B2's FF++ test 0.9987: checked, no leakage found.** Split test passes (6/6); 0 frames paths read; no DFD actor in FF++ training; only FF++ in training. Per-method: real vs Deepfakes 0.999 (114 videos), Face2Face 0.997 (20), FaceSwap 0.995 (6). The FF++ test is 81% Deepfakes (the easiest method) and lacks NeuralTextures, so near-perfect in-domain AUC is expected. **Please confirm (human check requested by the runbook).**
- **Old checkpoint `outputs_ffpp_v2_20260423_172121` on the new splits** (`runs/_old_ffpp_v2/`): FF++ val 1.000, FF++ test 1.000, DFD 0.999, Celeb-DF 0.956 video AUC. Its own split was a random video split of ALL of `ff++/faces` (incl. DFD), so it trained on most of our FF++ val/test and DFD videos: those three numbers are contaminated and only show it memorised them. Celeb-DF was never in its data, so 0.956 is a genuine cross-dataset number. **Leakage alarm (Swin cross-dataset > 0.95) checked:** its data_root was `ff++/faces` only; no Celeb-DF. Interpretation: it trained on ~2x more videos than B1 including DFD's 1,274 diverse actor videos; this suggests adding DFD as a training source would raise Celeb-DF AUC (B1: 0.899). Not in this week's plan; noted for the report and next steps.
- B1 measured: 4 min/epoch (~200 img/s), 47 min total + 4 min test evaluation.

## Decision rules applied

- 2026-11-17 12:31 **B1 crashed at epoch 3** (exit 1, run folder renamed `runs/_crashed_b1_swin_s42_20261117-1150`): I edited `data/dataset.py` while it trained, and Windows DataLoader workers re-import code from disk, so the epoch-3 validation workers loaded the new class. Fix: every queue job now runs through `scripts/run_snapshot.py`, which freezes a copy of the code in `runs/_code/` per job. B1 re-queued as `b1_swin_s42_r2`. Epoch 1-2 val video AUC before the crash: 0.992, 0.994.
- 2026-11-17 14:25 **DANN design conflict found in B2** (runbook spec: domains = real + 3 FF++ methods, domain CE on all samples). As the GRL strength ramps up, the domain head collapses to chance (CE = ln 4 exactly, accuracy 0.25) and real/fake train BCE rises from 0.09 (epoch 2) to 0.50 (epoch 4): with "real" as a domain, the reversed gradient removes exactly the real-vs-fake signal. Action: B2 runs to completion as specified (a negative result worth reporting); added `domain_on: fake` (domain CE only on fake samples, so manipulation methods are aligned with each other, never with real) and use it for G3 (with MMD on fake domains, as the runbook already says) and for a new control run `b2f_swin_dann_fake` queued before C1. **Deviation from T4.1** ("domain CE on all samples"); to run the spec version of G3 instead, set `domain_on: all` in `scripts/make_g3_configs.py`.
- 2026-11-17 15:25 **B2-fake collapsed too (val video AUC 0.999 -> 0.22 -> 0.10 at epochs 4-5), so excluding "real" was not the whole cause. Root cause found:** the domain head read the features *before* the classifier's LayerNorm. The adversary could then fool it by shrinking the single learnable InstanceNorm gamma (B2: 9e-4, B2-fake: 5e-4 already at epoch 2), while the scale-invariant real/fake head gave no resisting gradient; once gamma neared LayerNorm's eps and crossed 0, the features flipped sign (inverted AUC) and then vanished (AUC 0.5). Fix: the domain head now reads the same normalised features as the classifier, and norm scales/biases get no weight decay (decay had already pushed B1's gamma to 0.0045). Unit test added. B2-fake stopped at epoch 5; both collapsed runs renamed `runs/_collapsed_*` (kept as evidence); `b2_swin_dann_s42_r2` and `b2f_swin_dann_fake_s42_r2` re-queued after C2, before G1-G3.
- 2026-11-17 16:30 C1 val video AUC saturated at 1.000 from epoch 3, so selection by video AUC alone froze on epoch 3 (ties never "improve"). From C2 onward, ties are broken by val frame AUC (`train.py`). C1 keeps epoch-3 selection (its code is frozen in its snapshot).
- 2026-11-17 12:33 GRACE smoke test (G2: CLIP + LoRA + DCT + gated fusion + concept head, 30 steps) passed end to end.
