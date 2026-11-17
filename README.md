# GRACE: generalizable, robust and explainable deepfake detection

Face-crop deepfake detector trained on FaceForensics++ and tested on unseen datasets (DeepFakeDetection,
Celeb-DF v2). The teacher combines a CLIP ViT-L/14 backbone with LoRA, self-blended training images (SBI),
a concept-bottleneck head, a DCT or FFT frequency branch with gated fusion, and domain-adversarial alignment
(DANN + MMD). A distilled EfficientNet-B0 student runs on a CPU through ONNX. A Gradio app shows verdicts with
concept maps and evidence.

Results: `results/tables.md` (generated). Sprint log and open issues: `STATUS.md`. Data formats: `docs/interfaces.md`.

## 1. Environment

Windows 11 or Linux, one NVIDIA GPU (developed on an RTX 4500 Ada, 24 GB), Python 3.12.

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -c configs/constraints.txt
```

Keep other GPU programs closed while training: when GPU memory runs out, Windows swaps it to RAM silently and
training slows to a crawl.

## 2. Data

Expected layout under `data_root` (`configs/paths.yaml`, default: the repository root):

```text
ff++/faces/{real/REAL, fake/<Deepfakes|Face2Face|FaceSwap|DeepFakeDetection>}/<video>/aligned/*.png
celebdf_v2/celebdf_v2/faces/{real/REAL, fake/Celeb-synthesis}/<video>/aligned/*.png
```

Only aligned 224 x 224 face crops under `faces/` are used; any path under `frames/` is rejected.

```bash
python data/make_splits.py              # writes splits/*.csv (official FF++ split; DFD + Celeb-DF test only)
python -m pytest tests/test_splits.py   # leakage guard: no video or FF++ source ID in two splits
python scripts/extract_landmarks.py     # 68-point landmarks on FF++ train/val real crops (for SBI)
python scripts/extract_landmarks.py --splits ffpp_test --no-check-image
python scripts/make_sbi_sets.py         # fixed SBI evaluation sets (concept accuracy, pointing game)
```

For the official Celeb-DF test set, copy `List_of_testing_videos.txt` to `splits/official/` before
`make_splits.py`; without it an interim set with the same composition is used (see `splits/counts.md`).

## 3. Training

Every model is one YAML in `configs/` and one command:

```bash
python train.py --config configs/b1_swin.yaml --seed 42        # B1 Swin-B baseline
python train.py --config configs/c1_clip_lora.yaml --seed 42   # C1 CLIP + LoRA
python train.py --config configs/g3_full.yaml --seed 42        # G3 full teacher (written by scripts/make_g3_configs.py)
python scripts/make_sbi_pool.py && python scripts/cache_teacher.py
python distill.py --config configs/s3_kd_full.yaml --seed 42   # S3 distilled student
```

Long runs go through the queue, which survives crashes and closed terminals and runs each job from a frozen
copy of the code:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_queue.ps1   # reads queue.yaml; state in runs/_queue/
```

Each run writes `runs/<name>_s<seed>_<time>/` (config, command, git commit, log, `best.pth` with EMA weights,
`val_threshold.json`, test metrics and predictions) and appends a row to `experiments.csv`.

## 4. Evaluation

```bash
python evaluate.py --ckpt runs/<run>/best.pth --split splits/celebdf_test.csv [--degrade jpeg:50|down:0.5|h264:34]
python scripts/eval_all.py --models b1_swin c1_clip_lora g3_full s3_kd_full     # robustness grid + figures
python scripts/make_tables.py                                                     # results/tables.md
```

Video AUC (mean probability over a video's crops) is the headline metric. Thresholds come from FF++
validation only; test splits are used only by `evaluate.py`.

## 5. Demo app

```bash
python scripts/export_onnx.py          # release/grace_student.onnx (+ decision weights .json)
python scripts/calibrate.py            # temperature scaling on FF++ val
python app/app.py                      # http://localhost:7860
```

*Quick* mode runs the student via ONNX Runtime on the CPU; *Deep* mode runs the G3 teacher on the GPU.

## 6. Reproducing the main table from released checkpoints

```bash
python scripts/make_release.py         # (author side) release/ with SHA256SUMS
python evaluate.py --ckpt release/grace_teacher_g3.pth --split splits/celebdf_test.csv \
    --threshold-from release/grace_teacher_g3_val_threshold.json --out-dir runs/_repro
```

## Limitations

FF++ on disk is c23 and partial (FaceSwap 39 and Face2Face 134 videos with crops; no NeuralTextures or
FaceShifter); two cross-dataset test sets; faces only, no audio; one or two seeds per model.
