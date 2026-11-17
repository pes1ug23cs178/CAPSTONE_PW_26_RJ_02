# Interfaces

Every script reads and writes these formats, so any piece can be rerun or replaced on its own. Do not change a format without updating this section's copy in `docs/interfaces.md` in the repository.

### Split CSV (`splits/*.csv`), one row per face crop

```csv
video_id,frame_path,label,dataset,method,source_id,landmarks_path
ffpp_000,ff++/faces/real/REAL/000/aligned/0001.png,0,ffpp,real,000,cache/landmarks/ffpp_000/0001.npy
ffpp_000_003,ff++/faces/fake/Deepfakes/000_003/aligned/0001.png,1,ffpp,Deepfakes,000,
cdf_id0_0001,celebdf_v2/celebdf_v2/faces/fake/Celeb-synthesis/id0_id1_0001/aligned/0001.png,1,celebdf,Celeb-synthesis,id0,
```

`frame_path` is relative to `data_root`. `label` is 0 for real and 1 for fake. `source_id` is the FF++ source video or the Celeb-DF identity; the overlap test runs on it. `landmarks_path` is empty where not extracted.

### Batch dictionary (what the DataLoader yields)

```text
image          float32 [B,3,224,224] in [0,1], unnormalized
label          float32 [B]           0 real, 1 fake
concepts       float32 [B,10]        recipe labels (zeros for real)
concept_mask   float32 [B,10]        1 = label valid (real, SBI), 0 = no label (FF++ fakes)
boundary       float32 [B,16,16]     blend-boundary map (zeros for real)
boundary_mask  float32 [B]           1 for real and SBI, 0 for FF++ fakes
domain_id      int64   [B]           0 real, 1 Deepfakes, 2 Face2Face, 3 FaceSwap, 4 SBI
video_id, method, path               lists of str
```

### Model output dictionary (every model's `forward`)

```text
logit          [B]          fake logit; probability = sigmoid(logit). Two-class heads return l_fake - l_real
z              [B,512]      fused feature (B1/B2: backbone feature [B,1024])
concepts       [B,10]       concept score logits          (concept models only)
maps           [B,10,H,W]   concept map logits, H=W=16 teacher, 7 student
domain_logits  [B,D]        (DANN models only)
gate_mean      scalar       (fusion models only)
```

### Config YAML (`configs/<name>.yaml`)

```yaml
name: g3_full
parent: g2_dct              # the run this one changes by one switch
model:
  backbone: clip_vitl14_lora   # swin_b | clip_vitl14_lora | clip_vitl14_frozen | efficientnet_b0
  lora: {r: 16, alpha: 32, dropout: 0.1}
  freq_branch: dct             # none | dct | fft
  fusion: gated                # none | gated | concat
  head: concept                # linear | concept
  domain: {enabled: true, num_domains: 5, dann_weight: 0.1, mmd_weight: 0.1}
data:
  train: splits/ffpp_train.csv
  val: splits/ffpp_val.csv
  tests: [splits/ffpp_test.csv, splits/dfd_test.csv, splits/celebdf_test.csv]
  sbi: true
  cells: [real, Deepfakes, Face2Face, FaceSwap, sbi]
loss: {concept: 0.5, map: 0.5, w_l1: 1.0e-4, residual: 1.0e-3}
optim:
  lr: {lora: 2.0e-4, freq: 2.0e-3, fusion: 1.0e-3, heads: 1.0e-3}
  weight_decay: 0.05
  epochs: 12
  steps_per_epoch: 1500
  batch_size: 32
  warmup_epochs: 1
  ema: 0.999
  precision: bf16
```

### Run folder (`runs/<name>_s<seed>_<yyyymmdd-hhmm>/`)

```text
config.yaml  command.txt  git_commit.txt  train.log
best.pth                 # {model, ema, config, epoch, val_video_auc}
val_threshold.json       # {threshold, val_balanced_acc}
temperature.json         # written by calibrate.py
metrics_<split>[_<degrade>].json
predictions_<split>.csv  # video_id, frame_path, label, prob
roc_<split>.png
```

### Metrics JSON

```json
{"run": "g3_full_s42_20261120-0910", "split": "celebdf_test", "degrade": "clean",
 "n_videos": 518, "n_frames": 15540,
 "video_auc": 0.0, "frame_auc": 0.0, "eer": 0.0,
 "threshold": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
 "concept_share": null, "gate_mean": null}
```

### `experiments.csv`

```csv
run_id,config,parent,seed,commit,start,end,gpu_hours,best_epoch,val_video_auc,ffpp_test_video_auc,dfd_test_video_auc,celebdf_test_video_auc,notes
```

### `queue.yaml`

```yaml
jobs:
  - name: b1_swin_s42
    cmd: python train.py --config configs/b1_swin.yaml --seed 42
  - name: b2_swin_dann_s42
    cmd: python train.py --config configs/b2_swin_dann.yaml --seed 42
    after: b1_swin_s42
```


> Repository notes (17 Nov 2026): `video_id` for FF++ fakes is `ffpp_<method>_<AAA_BBB>` (method added so the same pair under Deepfakes/Face2Face/FaceSwap stays unique); thresholded metrics (acc, precision, recall, F1) are video-level; `predictions_*.csv` also carry `logit`, `c1..c10`, `concept_part`, `gate` where the model has them; `video_cues_*.csv` holds per-video flicker, identity drift and the per-frame score timeline.
