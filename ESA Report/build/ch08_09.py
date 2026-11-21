"""Chapter 8 (Proposed Methodology) and Chapter 9 (Implementation and Pseudocode)."""
from pathlib import Path

import pandas as pd

from data import R, cdf_note

FIG = Path(__file__).resolve().parent / "figs"


def ch8():
    return [
        ("chapter", "Proposed Methodology"),
        ("h2", "Overview"),
        ("p", "The methodology has five parts: (i) a leak-free data protocol; (ii) a self-blended data engine whose recipe "
              "provides concept labels; (iii) a teacher whose decision passes through a concept bottleneck; (iv) "
              "explanation-preserving distillation into a compact student; and (v) a controlled experiment plan in which "
              "every model differs from its parent by a single component. All models are trained on FF++ only, and "
              "every selection decision uses FF++ validation data only."),
        ("h2", "Datasets and Leak-Free Protocol"),
        ("p", "Training uses the official FF++ split by source video [@ffpp]: 720 training, 140 validation and 140 test "
              "source videos. A manipulated video is assigned to the split of its target video. DFD actor videos, which "
              "are stored with the FF++ real videos, are excluded from FF++ training and used only as part of the DFD "
              "test set. Celeb-DF v2 is used only for testing. " + cdf_note()),
        ("p", "Only aligned face crops are used; the full video frames are never read. Automated tests check that no video "
              "ID, FF++ source ID or Celeb-DF identity appears in two splits, and that every referenced file exists. "
              "Checkpoints are selected on FF++ validation, the decision threshold maximises balanced accuracy on FF++ "
              "validation, and calibration temperatures are fitted on FF++ validation. Video scores are the mean of the "
              "frame probabilities of a video."),
        ("h2", "Evaluation Integrity Audit"),
        ("p", "Before any Phase-3 model was trained, the Phase-2 pipeline was audited. Its multi-source Swin-B model had "
              "reported 97.98% validation accuracy (AUC 0.997) after a single epoch, and the single-dataset runs reported "
              "99.71% on Celeb-DF and 97.19% on FF++; the Phase-2 report quoted 98.98% on Celeb-DF. The audit found that "
              "every full-frame image received the same grouping key, so the frames of almost all validation videos "
              "(2,446 of 2,451) were also in the training set. Metrics were also computed on the early-stopping set, with "
              "stale moving-average weights, and only at frame level. The high numbers therefore measured recall of "
              "training videos rather than detection."),
        ("p", "The protocol of Section 8.2 removes each of these problems: splits by source video with automated tests, "
              "selection only on validation data, a separate test set, EMA weights evaluated after reloading the saved "
              "checkpoint, and video-level metrics. Section 10.10 re-evaluates an old checkpoint under the new protocol."),
        ("h2", "Self-Blended Data Engine with Recipe Concepts"),
        ("p", "For a real training face with 68 landmarks [@fan], a mask m is built from the convex hull of a randomly "
              "chosen region: the full face (probability 0.40), the mouth (0.25), the eyes and brows (0.20) or the nose "
              "and cheeks (0.15). With probability 0.5 the mask is warped elastically, and it is softened with a Gaussian "
              "kernel of random size. A source copy s of the face is altered by at least one of a colour shift, blur or "
              "sharpening, down- and up-sampling, and a small geometric shift. The forgery is the blend in Equation "
              "{eq:sbi}, with blend ratio a ∈ {0.25, 0.5, 0.75, 1}."),
        ("eq", "sbi", "x̃ = x ⊙ (1 − a·m) + s ⊙ (a·m)"),
        ("p", "The recipe of each forgery is converted into a soft concept vector y^{c} ∈ [0, 1]^{10} (Table {tab:concepts}) "
              "and a boundary map (Equation {eq:boundary}), which is pooled to 16 × 16. Real faces receive all-zero "
              "concept labels. FF++ fakes have no recipe, so their concept and boundary losses are masked out. A frozen "
              "pool of 40,000 self-blended faces is used for the students, and two fixed sets of 2,000 self-blended faces "
              "each (validation and test, from different videos) are used to measure concept accuracy and explanation "
              "localisation. Figure {fig:sbi} shows examples."),
        ("eq", "boundary", "b = 4 · m · (1 − m)"),
        ("tab", "concepts", "The ten forensic concepts, their labels and the plain-language names shown in the application.",
         ["ID", "Concept", "Label from the recipe", "Shown to the user as"],
         [["c1", "Blend strength", "a", "The face looks pasted on"],
          ["c2", "Colour mismatch", "a × colour-shift strength", "Skin colour does not quite match"],
          ["c3", "Blur in a region", "a if the source was blurred", "Part of the face is unusually blurry"],
          ["c4", "Over-sharpened region", "a if the source was sharpened", "Part of the face looks over-sharpened"],
          ["c5", "Resampling traces", "a if down/up-sampled", "Signs of resizing"],
          ["c6", "Mouth region edited", "1 if the mouth mask was used", "The mouth area looks edited"],
          ["c7", "Eye region edited", "1 if the eye mask was used", "The eye area looks edited"],
          ["c8", "Whole face swapped", "1 if the full-face mask was used", "The whole face may be replaced"],
          ["c9", "Misaligned geometry", "a if shifted or scaled", "The face is slightly out of place"],
          ["c10", "Irregular blend boundary", "1 if the mask was warped", "The edge around the face is uneven"]],
         [1.2, 4.0, 5.2, 6.6]),
        ("fig", "sbi", FIG / "sbi_examples.png", "Self-blended faces produced by the data engine: real face, self-blended fake and blend mask, with the logged region, blend ratio a and transforms.", 15.5),
        ("h2", "Concept Bottleneck Decision and Teacher Loss"),
        ("p", "Let c ∈ ℝ¹⁰ be the concept scores and σ the sigmoid function. The fake logit is given by Equation {eq:logit}, "
              "where w ∈ ℝ¹⁰ and b are learned and r is a small MLP on the fused feature z, initialised to zero."),
        ("eq", "logit", "ℓ = wᵀσ(c) + b + r(z)"),
        ("p", "The teacher is trained with the loss in Equation {eq:teacherloss}: binary cross-entropy on the label, masked "
              "binary cross-entropy between the concept scores and the recipe labels, a masked boundary loss on the "
              "first concept map, an L1 penalty on w, and a penalty λ_{r} on the residual. Optional domain terms are added "
              "in the ablations of Section 8.7."),
        ("eq", "teacherloss", "L_{T} = BCE(ℓ, y) + 0.5·BCE_{m}(c, y^{c}) + 0.5·BCE_{m}(M_{1}, b) || + 10^{−4}·‖w‖_{1} + λ_{r}·mean(r(z)^{2})"),
        ("p", "The share of the decision explained by the concepts is measured by the concept share in Equation "
              "{eq:share}, computed over a test set. The first concept-bottleneck models (G1, G2) used λ_{r} = 10^{−3} and "
              "reached a concept share of only about 0.25–0.29: the residual carried most of the decision. The final "
              "model (F1) uses λ_{r} = 10^{−2}, which raises the concept share to about 0.8 with little loss of accuracy "
              "(Section 10.3)."),
        ("eq", "share", "share = Var[wᵀσ(c)] / Var[ℓ]"),
        ("h2", "Frequency Branch and Gated Fusion"),
        ("p", "For the luminance channel Y, each 8 × 8 block B is transformed with the orthonormal DCT-II matrix D "
              "(Equation {eq:dct}). The coefficients are log-compressed, standardised per channel and encoded into "
              "h_{f} ∈ ℝ^{256}. With projections p_{c} = W_{c}·h_{cls} and p_{f} = W_{f}·h_{f}, the fused feature is computed with a "
              "learned gate (Equation {eq:gate})."),
        ("eq", "dct", "C = D · B · Dᵀ,    log(1 + |C|)"),
        ("eq", "gate", "g = σ(W_{g}[p_{c} ; p_{f}]),    z = LN(g ⊙ p_{c} + (1 − g) ⊙ p_{f})"),
        ("h2", "Domain Alignment (Ablations)"),
        ("p", "To test whether domain-adversarial training helps, a gradient reversal layer [@ganin] feeds z (or, in one "
              "variant, the patch features) to a domain classifier over the training sources, with reversal strength "
              "following Equation {eq:grl} at training progress p, plus a multi-kernel MMD term [@mmd] between fake "
              "domains. The variants are: real faces as a domain (B2, the Phase-2 design), fake domains only (B2-fake), "
              "domain terms on z for the full teacher (G3), domain terms on patch features (F2), and unsupervised "
              "adaptation to unlabelled Celeb-DF videos that are not in the test set (D2)."),
        ("eq", "grl", "λ_{p} = 2 / (1 + exp(−10p)) − 1"),
        ("h2", "Model Selection Rule"),
        ("p", "On clean FF++ validation data the video AUC of every CLIP-based model is between 0.999 and 1.000, so it "
              "cannot rank them. The final teacher was therefore chosen with a harder validation score (Equation "
              "{eq:select}): the mean of the clean validation AUC, the validation AUC after H.264 re-encoding at CRF 40, "
              "and the AUC on the self-blended validation set. Only validation data enter this score. F1 (score 0.961) "
              "was preferred to F2 (0.817), and the same score selects the best epoch of these models."),
        ("eq", "select", "S = (AUC_{val, clean} + AUC_{val, h264:40} + AUC_{SBI-val}) / 3"),
        ("h2", "Explanation-Preserving Distillation"),
        ("p", "The teacher's outputs on clean training faces are computed once in full precision and cached. Each "
              "training face is then shown to the student in a degraded form x̂ = D(x): JPEG quality 20–60, 0.25–0.5× "
              "resampling, or both. The student loss (Equation {eq:kdloss}) combines the label loss, the temperature-"
              "scaled divergence from the teacher's prediction (T = 4) [@hinton], a feature-matching term between the "
              "projected student feature and the teacher's z, a concept-matching term and a map-matching term."),
        ("eq", "kdloss", "L_{S} = 0.5·BCE + 0.5·T^{2}·KL(p_{T} ‖ p_{S}) + ‖P·z_{S} − z_{T}‖^{2} || + 0.5·BCE(c_{S}, σ(c_{T})) + 0.5·‖σ(M_{S}) − pool(σ(M_{T}))‖^{2}"),
        ("p", "Three students isolate the effect of the teacher: S0 is trained without a teacher (with the recipe "
              "losses), S2 uses only the logit term, and S3 uses all terms. All three see the same degraded inputs and "
              "select their epoch on FF++ validation frames compressed at JPEG quality 50."),
        ("h2", "Evaluation Metrics"),
        ("p", "The headline metric is the video-level area under the ROC curve (AUC). Equal error rate, accuracy, "
              "precision, recall and F1 are reported at the FF++-validation threshold. The cross-dataset score is the "
              "mean of the DFD and Celeb-DF AUCs. Robustness is measured on test crops re-encoded with JPEG at quality "
              "90, 70, 50 and 30, downscaled to 0.5 and 0.25 and back, and re-encoded per video with H.264 at CRF 28, 34 "
              "and 40. Probabilities are calibrated by temperature scaling [@guo], with T fitted on FF++ validation "
              "(Equation {eq:temp})."),
        ("eq", "temp", "p = σ(ℓ / T)"),
        ("p", "Explanations are evaluated on the held-out self-blended test faces, whose manipulated region is known "
              "exactly: the pointing game [@pointing] (does the map's maximum fall inside the region?), the IoU between "
              "the thresholded map and the region, deletion and insertion curves [@rise], the drop in the fake "
              "probability when the top concept is switched off compared with a random concept, the AUC of each concept "
              "against its recipe label, and teacher–student agreement (Spearman correlation of concept scores and IoU "
              "of maps)."),
        ("h2", "Experiment Plan"),
        ("p", "Table {tab:matrix} lists the models. Each row differs from its parent by one component, so every comparison "
              "isolates one design decision. The final teacher, the Swin-B baseline, all three students and the extra "
              "experiments X1 and D2 were trained with three random seeds (42, 1 and 2)."),
        ("tab", "matrix", "Experiment matrix. All models are trained on FF++ unless stated.",
         ["ID", "Model", "Question answered"],
         [["B1", "Swin-B", "Generalisation of the Phase-2 backbone under the new protocol"],
          ["B2 / B2-fake", "B1 + DANN (real as a domain / fake domains only)", "Did the Phase-2 domain alignment help?"],
          ["C1", "CLIP ViT-L/14 + LoRA, linear head", "Effect of the foundation backbone (H1)"],
          ["C2", "C1 + self-blended images", "Effect of self-blended data"],
          ["G1", "C2 + concept bottleneck", "Cost or gain of the bottleneck (H2)"],
          ["G2 / G2-fft", "G1 + DCT / FFT branch with gated fusion", "Does frequency evidence help, and which kind? (H3)"],
          ["G3", "Better of G2 / G2-fft + DANN and MMD on z", "Full teacher as planned"],
          ["A3", "G3 with a linear head", "Accuracy cost of the concepts"],
          ["F1", "G2-DCT + residual penalty 10⁻² + hard-validation selection", "Final teacher candidate"],
          ["F2", "F1 + DANN/MMD on patch features", "Final teacher candidate"],
          ["S0 / S2 / S3", "EfficientNet-B0: no teacher / logit KD / full KD", "Does the teacher help the student? (H4)"],
          ["D2", "F1 + unsupervised adaptation to Celeb-DF", "Extra: does target adaptation help?"],
          ["X1", "F1 trained on FF++ + DFD", "Extra: does more diverse data help?"],
          ["S3x / S0x / S3xx", "Students with the X1 teacher and/or DFD data", "Extra: data or teacher for students?"]],
         [2.6, 7.2, 7.2]),
    ]


ALGS = [
    ("alg_split", "Leak-free split construction and checking", [
        "input : face-crop folders; official FF++ split (source IDs); Celeb-DF test list",
        "output: train / val / test CSVs (one row per crop)",
        "for each FF++ video v:",
        "    s <- split of target_source_id(v)          # fakes follow their target video",
        "    append crops(v) to split s",
        "DFD actors and DFD fakes            -> dfd_test",
        "Celeb-DF videos in official list    -> celebdf_test",
        "for key in {video_id, source_id, identity}:",
        "    assert no value of key appears in two splits    # else stop",
        "assert every frame_path exists and no path contains 'frames/'",
    ]),
    ("alg_sbi", "Self-blended image with recipe labels", [
        "input : real face x, landmarks L (68 x 2), random generator g",
        "region <- sample {full .40, mouth .25, eyes .20, nose .15}",
        "m <- convex_hull_mask(L[region]); if g < 0.5: m <- elastic_warp(m)",
        "m <- gaussian_blur(m, random odd kernel); m <- m / max(m)",
        "s, rec <- transform(x)       # >= 1 of colour, blur|sharpen, resample, shift",
        "a <- sample {0.25, 0.5, 0.75, 1.0}",
        "x_fake <- x * (1 - a*m) + s * (a*m)",
        "c <- zeros(10); c1 <- a; c2 <- a*rec.colour; c3..c5 <- a*rec.{blur,sharpen,resample}",
        "c9 <- a*rec.geometry; c10 <- [mask was warped]; c6|c7|c8 <- 1 for mouth|eyes|full",
        "b <- resize(4*m*(1-m), 16x16) / max",
        "return x_fake, c, b",
    ]),
    ("alg_train", "One teacher training step", [
        "batch <- balanced sample over cells {real, sbi, Deepfakes, Face2Face, FaceSwap}",
        "x <- light_augment(batch.image)               # flip, colour jitter, JPEG 60-100, blur",
        "with bf16 autocast:",
        "    cls, P <- CLIP_LoRA(x);  h_f <- DCT_branch(x)",
        "    z, gate <- gated_fusion(cls, h_f)",
        "    c, M <- concept_head(P)                    # scores = mean of top-10% cells",
        "    logit, concept_part, r <- decision(c, z)",
        "    L <- BCE(logit, y) + 0.5*BCE_mask(c, y_c) + 0.5*BCE_mask(M1, boundary)",
        "         + 1e-4*|w|_1 + lambda_r*mean(r^2)",
        "L.backward(); clip_grad(1.0); AdamW.step(); scheduler.step(); EMA.update(model)",
        "after each epoch: evaluate EMA weights on FF++ val; keep best by selection score",
    ]),
    ("alg_kd", "One distillation step (student S3)", [
        "batch <- balanced sample over cells {real, sbi pool, Deepfakes, Face2Face, FaceSwap}",
        "t <- TeacherCache.lookup(batch.frame_path)    # logit, z, concepts, maps (clean input)",
        "x_hat <- degrade(batch.image)                 # JPEG 20-60 and/or 0.25-0.5x resample",
        "s <- student(x_hat)",
        "L <- 0.5*BCE(s.logit, y) + 0.5*T^2*KL(sig(t.logit/T) || sig(s.logit/T))",
        "     + MSE(P*s.z, t.z) + 0.5*BCE(s.concepts, sig(t.concepts))",
        "     + 0.5*MSE(sig(s.maps), avgpool(sig(t.maps), 7x7))",
        "L.backward(); AdamW.step(); EMA.update(student)",
    ]),
    ("alg_eval", "Video-level evaluation of one model on one test set", [
        "input : run folder, split, degradation d",
        "model <- load(best.pth) with EMA weights; thr <- threshold from FF++ val",
        "for each crop i in split: p_i <- sigmoid(model(degrade(crop_i, d)).logit)",
        "for each video v: p_v <- mean(p_i for crops i of v)",
        "AUC_video <- roc_auc(labels_v, p_v);  AUC_frame <- roc_auc(labels_i, p_i)",
        "EER, acc, precision, recall, F1 at thr (video level)",
        "write metrics_<split>_<d>.json, predictions_<split>_<d>.csv, roc_<split>_<d>.png",
    ]),
    ("alg_app", "Application analysis of an uploaded file", [
        "input : file f, check type (Careful | Quick)",
        "if f is missing, of the wrong type or larger than 200 MB: show message; stop",
        "frames <- [image] or 32 evenly spaced video frames",
        "faces <- [crop_with_pose(fr) for fr in frames if a face is found]",
        "if too few faces: show 'We couldn't find a face'; stop",
        "results <- engine.run(faces)                  # prob, concepts, maps, contributions",
        "used <- [r for r in results if pose_ok(r.yaw, r.pitch)]",
        "p <- mean(sigmoid(r.logit / T) for r in used)",
        "verdict <- Likely real (p < .35) | Can't tell (< .65) | Likely fake",
        "clues <- top-3 concepts by mean contribution with score >= 0.25",
        "render verdict card, clue cards (orange glow from concept maps), report",
    ]),
]


def ch9():
    exp = R.exp.copy()
    gpu_h = pd.to_numeric(exp["gpu_hours"], errors="coerce").sum()
    n_runs = len(exp)
    out = [
        ("chapter", "Implementation and Pseudocode"),
        ("h2", "Technology Stack"),
        ("tab", "stack", "Technology stack.",
         ["Purpose", "Technology"],
         [["Language", "Python 3.12"],
          ["Deep learning", "PyTorch 2.11 with CUDA 12.8 [@pytorch]; bf16 mixed precision"],
          ["Pretrained models", "OpenCLIP ViT-L/14 [@openclip]; timm Swin-B and EfficientNet-B0 [@timm]"],
          ["Parameter-efficient tuning", "PEFT (LoRA adapters) [@lora]"],
          ["Faces and landmarks", "face-alignment (SFD detector, 68-point landmarks) [@fan], OpenCV"],
          ["Video", "OpenCV decoding; FFmpeg/libx264 through imageio-ffmpeg for H.264 tests"],
          ["Evaluation and plots", "scikit-learn, pandas, matplotlib, t-SNE [@tsne]"],
          ["Retrieval", "FAISS [@faiss]"],
          ["Deployment", "ONNX and ONNX Runtime [@onnxruntime]; Gradio 6 web application [@gradio]"],
          ["Testing", "pytest (242 tests); Playwright browser checks; gradio_client API checks"]],
         [4.6, 12.4]),
        ("h2", "Code Organisation"),
        ("tab", "repo", "Repository structure.",
         ["Folder / file", "Contents"],
         [["data/", "Splits, dataset, balanced sampler, self-blending, degradations, teacher cache"],
          ["models/", "CLIP-LoRA backbone, DCT/FFT branches, fusion, concept head and decision, domain head, Swin, student, losses, EMA"],
          ["explain/", "Concept maps, contributions, interventions, Grad-CAM, attention rollout, retrieval, video cues"],
          ["app/", "pipeline.py (analysis without UI) and app.py (Gradio page and API)"],
          ["configs/", "One YAML file per model (26 configurations)"],
          ["scripts/", "Evaluation grid, calibration, explanation metrics, figures, tables, release, queue tools"],
          ["train.py, distill.py, evaluate.py", "Training, distillation and evaluation entry points"],
          ["queue_runner.py, queue.yaml", "Dependency-aware job queue used for all experiments"],
          ["tests/", "Unit and integration tests"],
          ["results/, runs/, release/", "Generated tables and figures, run folders, released models"]],
         [5.0, 12.0]),
        ("h2", "Training Configuration"),
        ("tab", "hparams", "Training hyperparameters.",
         ["Setting", "Teachers (C, G, F, D, X)", "Swin-B baselines (B)", "Students (S)"],
         [["Optimiser", "AdamW, weight decay 0.05", "AdamW", "AdamW, weight decay 0.01"],
          ["Learning rates", "LoRA 2e-4; heads and fusion 1e-3; DCT branch 2e-3", "backbone 1e-4; heads 1e-3", "1e-3"],
          ["Schedule", "1 warm-up epoch, then cosine decay", "1 warm-up epoch, cosine", "1 warm-up epoch, cosine"],
          ["Epochs × steps", "12 × 1,500", "10 epochs", "15 × 1,500"],
          ["Batch size", "32", "32", "64"],
          ["Precision, EMA, clipping", "bf16, EMA 0.999, grad-norm 1.0", "bf16, EMA 0.999", "bf16, EMA 0.999, grad-norm 1.0"],
          ["Batch composition", "real 3 : SBI 2 : each FF++ method 1", "real : each method", "real 3 : SBI 3 : each method 1"],
          ["Augmentation", "flip, colour jitter, JPEG 60–100, blur", "same", "degradation only (JPEG 20–60, resampling)"]],
         [3.4, 5.4, 3.8, 4.4]),
        ("h2", "Pseudocode of the Main Algorithms"),
        ("p", "Algorithms {alg:alg_split} to {alg:alg_app} give the main procedures in pseudocode. They follow the "
              "implementation in the repository closely; helper functions are named after the corresponding functions in "
              "the code."),
    ]
    for lab, title, lines in ALGS:
        out.append(("alg", lab, title, lines))
    out += [
        ("h2", "Testing and Validation of the Implementation"),
        ("p", "The implementation was tested at four levels. Unit tests cover the split rules, tensor shapes, the "
              "self-blending engine, the concept and decision layers, the DCT branch (a constant image must produce only "
              "a DC coefficient), the student, the degradations, the video cues and the head-pose filter. Sanity runs "
              "checked that a model can over-fit 32 images, that a reloaded checkpoint reproduces the trainer's "
              "validation AUC exactly, and that random scores give an AUC near 0.5. The distillation cache is checked "
              "against fresh forward passes, and the ONNX export against PyTorch. Finally, the application was tested "
              "through its API and in a real browser."),
        ("tab", "tests", "Automated tests and their results.",
         ["Test group", "What is checked", "Result"],
         [["Split tests (6)", "No shared video, source ID or identity; files exist; no frames/ paths", "Pass"],
          ["SBI tests (204)", "Masks, transforms, label ranges and boundary maps for many seeds", "Pass"],
          ["Model tests (17)", "Shapes, concept head, decision layer, DCT/FFT branches, student", "Pass"],
          ["Data, degradation, explanation, video-cue and pose tests (15)", "Loader, JPEG/resize/H.264, maps, interventions, flicker, pose filter", "Pass"],
          ["Total", "242 automated tests (65 s)", "242 passed"],
          ["Teacher cache check", "Cached vs fresh logit on 100 random crops", "max difference 6.7 × 10⁻⁶"],
          ["ONNX export check", "Exported vs PyTorch student logits", "max difference 2 × 10⁻⁶"],
          ["API test (gradio_client)", "/check, /clear, /rescore, /overlay_concept, /overlay_opacity", "All working"],
          ["Browser test (Playwright, Edge)", "14 checks on 3 videos, both check types", "14 / 14 pass"]],
         [5.0, 8.4, 3.6]),
        ("h2", "Execution of the Experiments"),
        ("p", f"The experiments ran on one NVIDIA RTX 4500 Ada GPU in the department lab in November 2026, "
              f"including the extra seeds and experiments. In total {n_runs} training runs were logged in experiments.csv, "
              f"using about {gpu_h:.0f} GPU hours. A teacher run took about 1.8–1.9 GPU hours, a Swin-B run about 0.8 h "
              f"and a student run about 0.7 h. The job queue ran unattended; failed jobs were diagnosed, "
              f"fixed and re-queued under a new name, so every result can be traced to the exact code that produced "
              f"it."),
    ]
    return out
