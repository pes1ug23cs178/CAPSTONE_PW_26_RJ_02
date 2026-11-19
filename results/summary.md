# Sprint summary (draft generated from results files)

## Setup

- Train: FaceForensics++ c23 (Deepfakes, Face2Face, FaceSwap; official split by source video), aligned face crops only.
- Unseen tests: DeepFakeDetection (DFD) and Celeb-DF v2; FF++ test for in-domain. Celeb-DF test mode: interim.
- Metric: video AUC (mean fake probability over a video's crops); thresholds from FF++ validation only.
- Models: Swin-B baseline (B1, B2 + DANN), CLIP ViT-L/14 + LoRA (C1), + self-blended images (C2), + concept bottleneck (G1), + frequency branch with gated fusion (G2 DCT / G2-fft), + DANN (G3); final teacher chosen on FF++ val between F1 (G2-DCT + residual 1e-2 + harder val selection) and F2 (F1 + DANN on patch features); EfficientNet-B0 students S0/S2/S3 distilled from the final teacher.
- Hardware: one RTX 4500 Ada (24 GB); seeds 42, 1 and 2 for the final teacher (F1), B1 and all three students (S0, S2, S3); X1 and D2 two seeds; other models one seed (42). Numbers are seed means where more than one seed exists.

## Findings

- Replacing Swin-B with CLIP + LoRA raises cross-dataset video AUC (mean of DFD and Celeb-DF): 0.901 → 0.939 (B1 Swin-B → C1 CLIP + LoRA).
- Adding self-blended images lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.939 → 0.931 (C1 CLIP + LoRA → C2 + SBI).
- The concept bottleneck raises cross-dataset video AUC (mean of DFD and Celeb-DF): 0.931 → 0.941 (C2 + SBI → G1 + concepts).
- The DCT branch with gated fusion raises cross-dataset video AUC (mean of DFD and Celeb-DF): 0.941 → 0.948 (G1 + concepts → G2 + DCT fusion).
- The FFT branch with gated fusion lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.941 → 0.924 (G1 + concepts → G2-fft + FFT fusion).
- DANN on Swin-B with real as a domain (runbook spec) lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.901 → 0.870 (B1 Swin-B → B2 Swin-B + DANN (real as a domain)).
- DANN on Swin-B over fake domains only leaves essentially unchanged (±0.005) cross-dataset video AUC (mean of DFD and Celeb-DF): 0.901 → 0.898 (B1 Swin-B → B2-fake Swin-B + DANN (fake domains only)).
- Removing the concept head from G3 (A3) lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.935 → 0.928 (G3 full teacher (+ DANN) → A3 G3 w/o concepts).
- Residual penalty 1e-2 + harder val selection on G2-DCT (F1) lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.948 → 0.926 (G2 + DCT fusion → F1 G2-DCT + residual 1e-2 + hard-val selection).
- DANN/MMD on patch features (F2 vs F1) lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.926 → 0.764 (F1 G2-DCT + residual 1e-2 + hard-val selection → F2 F1 + DANN/MMD on patch features).
- Domain terms on patch features instead of z (F2 vs G3) lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.935 → 0.764 (G3 full teacher (+ DANN) → F2 F1 + DANN/MMD on patch features).
- Full distillation over a teacher-free student lowers cross-dataset video AUC (mean of DFD and Celeb-DF): 0.842 → 0.826 (S0 student, no teacher → S3 student, full KD).
- Under JPEG quality 30, raises cross-dataset video AUC (mean of DFD and Celeb-DF) at jpeg:30: 0.827 → 0.833 (S0 student, no teacher → S3 student, full KD).
- Under JPEG quality 30, raises cross-dataset video AUC (mean of DFD and Celeb-DF) at jpeg:30: 0.823 → 0.833 (B1 Swin-B → S3 student, full KD).
- The S3 student has 4.71 M parameters and 0.39 GFLOPs vs 306.94 M and 81.46 GFLOPs for the teacher (f1_grace_dct); ONNX CPU latency 4.2 ms per face.
- Pointing game on held-out SBI blends: F1 c1 0.52, F1 region 0.94, B1 Grad-CAM 0.15, C1 rollout 0.01, S3 c1 0.72, random 0.25.

Decisions taken by the runbook rules:

- 2026-11-18 05:06 G3 base = g2_fft: g2_fft best on val ({'g1_concepts': 0.9999, 'g2_dct': 0.9999, 'g2_fft': 1.0})
- 2026-11-18 20:49 final teacher = f1_grace_dct (f1_grace_dct_s42_20261118-1523) on val select score: f1_grace_dct 0.9614; f2_grace_pdann 0.8174
- 2026-11-18 21:31 teacher = final teacher f1_grace_dct: f1_grace_dct_s42_20261118-1523

## What did not work

(Fill in from the negative deltas above and `results/decisions.md`.)

## Limitations

- Reduced FF++: c23 only; FaceSwap 39 and Face2Face 134 videos have crops; no NeuralTextures or FaceShifter.
- Two cross-dataset sets; Celeb-DF uses an interim test list (same composition as the official one).
- Faces only, no audio; one or two seeds per model.
- The app's face crops use SFD + eye alignment (margin 1.02x), not the original preprocessing code.
- Head pose: FF++ is near-frontal; faces beyond its pose range (|yaw| > 0.57) are excluded from the app verdict.
- The student is less reliable than the teacher on re-cropped and on hard real Celeb-DF faces; the app defaults to the teacher (Deep).
- Explanations localise well (pointing game) but deletion tests do not show that removing the highlighted region flips the decision; distillation transfers concept scores better than map locations.

## Next steps toward a paper

- Full FF++ (all methods, c23 + c40), more unseen sets (DFDC-P, DF40), three seeds for every model, a user study of the explanations.
