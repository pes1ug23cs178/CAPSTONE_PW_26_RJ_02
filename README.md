# GRACE: Generalized Deepfake Detection Framework with Scalable Multimodal Integration

**Capstone Project PW26_RJ_02 · UE23CS441A Capstone Project Phase 3 · PES University, Department of CSE · August – December 2026**

| Team | SRN |
| --- | --- |
| Charan L | PES1UG23CS159 |
| Deepak S N | PES1UG23CS178 |
| Samrudh N Shindhe | PES1UG23CS508 |
| Samarth Gowda G M | PES1UG23CS502 |

**Guide:** Dr. Jyothi R, Associate Professor, Department of CSE, PES University

---

## What this project does

Deepfake detectors usually score almost perfectly on the dataset they were trained on, yet they fail on videos
from new sources, lose accuracy on compressed uploads, and give no reason for their answer. **GRACE**
(Generalisable, Robust And Concept-Explainable detection) addresses all three:

- **Generalises to unseen data.** It is trained only on FaceForensics++ and tested on two datasets it never sees:
  Google's DeepFakeDetection (DFD) and Celeb-DF v2.
- **Survives compression.** A JPEG-aligned DCT frequency branch keeps accuracy under strong JPEG compression and
  resizing.
- **Explains every decision.** The fake score is computed from **10 named forensic clues**, for example "the edge
  around the face is uneven" or "the mouth area looks edited". Each clue comes with a heat-map. The clue labels are
  free: we create our own training fakes and record how each one was made.
- **Runs anywhere.** A 4.7 M-parameter student model, distilled from the large teacher, runs on a CPU in about 4 ms
  per face.
- **Speaks plain language.** A web app answers "Likely fake / Can't tell / Looks real", shows the three clues behind
  the answer on the face, and keeps every technical detail one click away.

![GRACE architecture](ESA%20Report/build/figs/grace_architecture_overview.png)

## Key results

Video-level AUC; models trained only on FaceForensics++ (c23). The cross-dataset score is the mean of DFD and
Celeb-DF. ± is the standard deviation over 3 random seeds.

| Model | FF++ test | DFD (unseen) | Celeb-DF (unseen) | Cross-dataset |
| --- | --- | --- | --- | --- |
| B1 Swin-B baseline (Phase-2 backbone) | 0.997 ± 0.002 | 0.903 ± 0.016 | 0.889 ± 0.009 | 0.896 ± 0.008 |
| C1 CLIP ViT-L/14 + LoRA | 1.000 | 0.943 | 0.936 | 0.939 |
| G1 + 10-clue concept bottleneck | 1.000 | 0.972 | 0.911 | 0.941 |
| G2 + DCT branch, gated fusion | 1.000 | 0.974 | 0.922 | 0.948 |
| **F1 final teacher** (≈80% of each decision explained by clues) | 0.997 ± 0.003 | 0.951 ± 0.003 | 0.901 ± 0.014 | **0.926 ± 0.008** |
| S3 student (4.7 M params, CPU) | 0.978 ± 0.005 | 0.844 ± 0.011 | 0.808 ± 0.021 | 0.826 ± 0.010 |
| X1 = F1 recipe trained on FF++ + DFD | 0.998 ± 0.001 | (in training) | **0.959 ± 0.007** | – |

The Celeb-DF numbers above use an interim 518-video test set with the official composition. The official
`List_of_testing_videos.txt` was added on 20 Nov 2026, and every model is being re-evaluated on it. All tables in
`results/` are regenerated automatically from the run folders.

**Other findings**

- **Compression:** at JPEG quality 30, F1 keeps a cross-dataset AUC of 0.92, while the Swin-B baseline falls to 0.82.
  Very strong H.264 re-encoding (CRF 40) is still hard for every model (about 0.64).
- **Explanations:** GRACE's clue maps point at the edited region 94% of the time. Grad-CAM on Swin-B manages 16%,
  and a random map 25%.
- **What did not help** (reported honestly):
  - domain-adversarial training (five variants);
  - a global FFT branch;
  - unsupervised adaptation to Celeb-DF;
  - distillation, which did not raise the student's clean accuracy (it transfers the explanations instead).
- **Biggest lever:** more varied labelled data. Adding DFD to training raised Celeb-DF AUC from 0.90 to 0.96.

## Evaluation integrity

The Phase-2 prototype reported 98.98% validation accuracy. An audit in Phase 3 found that frames of 2,446 of the
2,451 validation videos were also in the training set. Phase 3 therefore uses:

- the official FF++ split by source video, with DFD and Celeb-DF used only for testing;
- automated tests (`tests/test_splits.py`) that stop training if any video, FF++ source ID or Celeb-DF identity
  appears in two splits;
- checkpoint selection, thresholds and calibration on FF++ validation only;
- one evaluation script for frame- and video-level metrics;
- a log line in `experiments.csv` for every run, with its code commit, seed and GPU hours.

## Repository layout

```text
app/               Gradio web app (app.py) and the analysis pipeline without UI code (pipeline.py)
configs/           One YAML per model (B1, C1, C2, G1, G2, G3, A3, F1, F2, S0, S2, S3, D2, X1, ...)
data/              Leak-free splits, dataset, balanced sampler, self-blended images (SBI), degradations, teacher cache
models/            CLIP+LoRA backbone, DCT/FFT branches, gated fusion, concept head + decision, domain head,
                   Swin baseline, EfficientNet-B0 student, losses, EMA
explain/           Concept maps, contributions, interventions, Grad-CAM, attention rollout, retrieval, video cues
scripts/           Evaluation grid, calibration, ONNX export, explanation metrics, figures, tables, release, queue tools
tests/             242 automated tests (pytest)
splits/            Split CSVs and counts (official FF++ split, DFD, Celeb-DF)
results/           Generated tables, robustness grid, ROC / robustness / t-SNE / explanation figures
docs/              Data and file formats (interfaces.md)
ESA Report/        Phase-3 report (Word + PDF), Review-3 slides, guidelines, templates and their generators
train.py, distill.py, evaluate.py, queue_runner.py, queue.yaml, experiments.csv, STATUS.md
```

Datasets, checkpoints (`runs/`, `release/`) and caches (`cache/`) are not stored in the repository.

## Setup

Windows 11 or Linux, one NVIDIA GPU (developed on an RTX 4500 Ada, 24 GB), Python 3.12.

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -c configs/constraints.txt
```

On Windows, set `TORCHDYNAMO_DISABLE=1` before running anything that detects faces: face-alignment's compiled
model needs Triton, which Windows lacks.

## Data

Expected layout under `data_root` (`configs/paths.yaml`, default: the repository root):

```text
ff++/faces/{real/REAL, fake/<Deepfakes|Face2Face|FaceSwap|DeepFakeDetection>}/<video>/aligned/*.png
celebdf_v2/celebdf_v2/faces/{real/REAL, fake/Celeb-synthesis}/<video>/aligned/*.png
```

Only aligned 224 × 224 face crops are used; any path under `frames/` is rejected.

```bash
python data/make_splits.py              # splits/*.csv (official FF++ split; DFD + Celeb-DF test only)
python -m pytest tests/test_splits.py   # leakage guard
python scripts/extract_landmarks.py     # 68-point landmarks for self-blending
python scripts/make_sbi_sets.py         # fixed self-blended evaluation sets (concept accuracy, pointing game)
```

For the official Celeb-DF test set, put `List_of_testing_videos.txt` in `splits/official/` before running
`make_splits.py`.

## Training

Every model is one YAML file and one command:

```bash
python train.py --config configs/b1_swin.yaml --seed 42          # Swin-B baseline
python train.py --config configs/c1_clip_lora.yaml --seed 42     # CLIP + LoRA
python train.py --config configs/f1_grace_dct.yaml --seed 42     # final teacher (F1)
python scripts/make_sbi_pool.py && python scripts/cache_teacher.py
python distill.py --config configs/s3_kd_full.yaml --seed 42     # distilled student (S3)
```

Long experiments run through a dependency-aware queue. Each job runs from a frozen copy of the code and is logged:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_queue.ps1   # reads queue.yaml; state in runs/_queue/
```

## Evaluation

```bash
python evaluate.py --ckpt runs/<run>/best.pth --split splits/celebdf_test.csv [--degrade jpeg:30|down:0.25|h264:40]
python scripts/eval_all.py --models b1_swin c1_clip_lora f1_grace_dct s3_kd_full --seeds 42 1 2
python scripts/explain_eval.py          # pointing game, mask IoU, deletion/insertion, interventions
python scripts/make_tables.py           # results/tables.md
```

## Web app

**Live demo:** runs on a team laptop with a public link (`docs/LAPTOP_DEMO.md`). The trained models are public at
https://huggingface.co/grace-cap/grace-deepfake-models and download automatically.

```bash
python scripts/export_onnx.py && python scripts/calibrate.py && python scripts/make_release.py
python app/app.py                       # http://127.0.0.1:7860
```

- **Careful check** (default) runs the final teacher on the GPU. **Quick check** runs the student via ONNX Runtime
  on any CPU.
- The page shows a plain-language verdict card, the three clues behind it (drawn on the face), what to do next,
  and a technical panel: probability, timeline, concept maps, switch-off test, nearest training faces and an HTML
  report with the file's SHA-256.
- API endpoints: `/check`, `/clear`, `/rescore`, `/overlay_concept`, `/overlay_opacity` (usable with `gradio_client`).
- Browser test: `python scripts/ui_check.py <video> <out_prefix> [quick]` (14 checks, Playwright on Edge).

## Tests

```bash
python -m pytest tests                  # 242 tests: splits, SBI engine, models, data, degradations, explanations, pose filter
```

## Phase-3 report and Review-3 slides

| File | What it is |
| --- | --- |
| `ESA Report/output/PW26_RJ_02_Phase3_Report.docx` / `.pdf` | 82-page report built on the department templates and guidelines |
| `ESA Report/output/PW26_RJ_02_Review3.pptx` | 27-slide Review-3 presentation on the department template |
| `ESA Report/REPORT_PROMPT.md` | Requirements brief: rules, decisions, fixed details |
| `ESA Report/build/` | Report generator (`build_report.py`), diagrams and figures |
| `ESA Report/ppt_build/build_ppt.py` | Slide generator |

The report and slides read every number from `results/`, so they can be rebuilt after new results:
`python "ESA Report/build/build_report.py"` (needs Microsoft Word for the table of contents and the PDF), then
`python "ESA Report/ppt_build/build_ppt.py"`.

## Limitations

- The FF++ copy available is c23 and partial: there are no NeuralTextures or FaceShifter videos, and only a few
  Face2Face and FaceSwap videos.
- Two unseen test sets were used, and both predate diffusion-based face synthesis.
- The system is visual and frame-based. Audio and audio-visual analysis are future work: the decision layer is
  designed to accept extra clues such as lip-speech mismatch.
- Very strong video compression remains hard, and extreme head poses are excluded from the verdict.

## Acknowledgement

We thank our guide Dr. Jyothi R, the project coordinator Dr. Chaitra B and the Department of Computer Science and
Engineering, PES University, for their guidance and support.
