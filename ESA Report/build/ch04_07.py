"""Chapter 4 (PRS), 5 (HLD), 6 (LLD), 7 (System Design, detailed)."""
from pathlib import Path

from data import R, REPO

FIG = Path(__file__).resolve().parent / "figs"


def _counts_rows():
    rows = []
    for r in R.counts[list(R.counts)[0]][1:]:
        split, method, label, vids, crops = r
        if "all" in method:
            rows.append([split.replace("_", " "), "**all**", "", vids.strip("*"), crops.strip("*")])
        else:
            rows.append([split.replace("_", " "), method, "fake" if label == "1" else "real", vids, crops])
    return rows


def ch4():
    s3 = R.eff.set_index("model").loc["s3_kd_full"]
    return [
        ("chapter", "Project Requirements Specification"),
        ("p", "This chapter specifies the requirements of the GRACE system in the style of a software requirements "
              "specification. The requirements were fixed at the start of Phase 3 and refined during development; "
              "the last column of each requirement table records how the requirement was verified."),
        ("h2", "Introduction"),
        ("h3", "Purpose"),
        ("p", "The purpose of the system is to detect manipulated faces in images and short videos, to explain each "
              "decision in terms of named forensic concepts, and to make this capability available through a simple web "
              "application that runs either on a workstation GPU or on an ordinary CPU."),
        ("h3", "Intended Audience"),
        ("p", "The specification is written for the project guide and evaluators, for the team members who build and test "
              "the system, and for future students who extend it. The application itself is intended for two kinds of "
              "users: non-expert users such as journalists, students and members of the public who want a clear answer, "
              "and expert users such as forensic analysts and researchers who need the underlying evidence."),
        ("h3", "Product Scope"),
        ("p", "The product consists of (i) a data pipeline that builds leak-free splits and synthetic training forgeries, "
              "(ii) training and distillation programs for the teacher and student models, (iii) an evaluation suite for "
              "accuracy, robustness, efficiency and explanation quality, and (iv) a web application for inference. Audio "
              "analysis, live-stream monitoring and large-scale content moderation are outside the scope of this phase."),
        ("h3", "Definitions, Acronyms and Abbreviations"),
        ("p", "All acronyms used in this report (AUC, CLIP, DCT, DANN, EER, LoRA, SBI and others) are defined in Appendix A."),
        ("h2", "Overall Description"),
        ("h3", "Product Perspective"),
        ("p", "GRACE is a self-contained system that runs on a single workstation. It reads face-crop datasets from local "
              "storage, writes trained models and results into the project repository, and serves the application over "
              "HTTP on the local machine. It replaces the Phase-2 prototype, whose single Swin-B classifier and Grad-CAM "
              "explanations are kept as baselines for comparison."),
        ("h3", "Product Functions"),
        ("bul", [
            "Build training, validation and test splits with no shared videos, source identities or near-duplicates.",
            "Generate self-blended forgeries with a logged recipe and convert the recipe into concept labels.",
            "Train teacher models (CLIP + LoRA, DCT branch, gated fusion, concept bottleneck) and baselines.",
            "Distil students on degraded frames and export them to ONNX for CPU inference.",
            "Evaluate every model on three test sets under ten degradation settings and compute explanation metrics.",
            "Analyse an uploaded image or video and present a calibrated verdict, its clues and a downloadable report.",
        ]),
        ("h3", "User Classes and Characteristics"),
        ("tab", "users", "User classes of the system.",
         ["User class", "Needs", "Interface"],
         [["Non-expert user", "A clear verdict in plain words, what to do next", "Result page (default view)"],
          ["Forensic / expert user", "Probability, timeline, concept maps, re-scoring, evidence", "Technical-details panel, HTML report"],
          ["Researcher / developer", "Re-run training and evaluation, add models", "Command-line scripts, configuration files"],
          ["Evaluator", "Verify claims and reproduce numbers", "experiments.csv, results tables, tests"]],
         [4.0, 7.4, 5.6]),
        ("h3", "Operating Environment"),
        ("p", "Development and training run on Windows 11 with Python 3.12, PyTorch 2.11 and CUDA 12.8 on an NVIDIA RTX "
              "4500 Ada GPU with 24 GB of memory. The application runs in any modern web browser. The Quick check needs "
              "only a CPU and ONNX Runtime; the Careful check needs a CUDA-capable GPU with at least 8 GB of memory."),
        ("h3", "Design and Implementation Constraints"),
        ("bul", [
            "Test data may never influence training or model selection; all choices use FF++ validation data.",
            "Datasets are licensed for research use, are read-only, and are never redistributed or uploaded.",
            "GPU time in the lab was limited, so every run had to be scheduled and logged automatically.",
            "The application must process uploads locally; no media leaves the machine.",
        ]),
        ("h3", "Assumptions and Dependencies"),
        ("bul", [
            "Faces have been (or can be) detected and aligned; one face per frame is analysed.",
            "Pretrained weights for CLIP ViT-L/14 (OpenCLIP), Swin-B and EfficientNet-B0 (timm) are available.",
            "The face-alignment library [@fan] provides 68-point landmarks for self-blending and pose estimation.",
        ]),
        ("h2", "Functional Requirements"),
        ("tab", "fr", "Functional requirements and how each was verified.",
         ["ID", "Requirement", "Verification"],
         [["FR-1", "Build splits by official FF++ source IDs and the Celeb-DF test list; reject any overlap of video ID, source ID or identity", "pytest test_splits (6 tests) on every run"],
          ["FR-2", "Generate self-blended images from real training faces with a recipe of region, blend ratio and transforms", "204 SBI unit tests; visual check sheet"],
          ["FR-3", "Convert each recipe into a 10-dimensional concept label and a 16×16 boundary map", "Unit tests on label ranges"],
          ["FR-4", "Train teacher and baseline models from YAML configurations with a fixed seed", "experiments.csv rows (commit, seed)"],
          ["FR-5", "Select the best epoch and the final model using FF++ validation data only", "Selection log in results/decisions.md"],
          ["FR-6", "Distil students from cached teacher outputs on degraded inputs", "Cache check: cached vs fresh logit < 1e-3"],
          ["FR-7", "Report frame and video AUC, EER, accuracy, precision, recall and F1 per test set", "evaluate.py metrics files"],
          ["FR-8", "Evaluate under JPEG, downscaling and H.264 degradations", "results/robustness.csv"],
          ["FR-9", "Produce concept maps, contributions, interventions, Grad-CAM and attention-rollout maps", "Explanation metrics table"],
          ["FR-10", "Accept an image or video upload, analyse up to 32 frames, and return a calibrated verdict with an inconclusive band", "App check on 6 held-out files"],
          ["FR-11", "Show the clues behind the verdict in plain language and the technical details on demand", "Browser test (14 checks)"],
          ["FR-12", "Provide a downloadable HTML report with the file's SHA-256 hash", "Browser test: download"],
          ["FR-13", "Expose the analysis as HTTP API endpoints", "gradio_client test of all endpoints"]],
         [1.5, 10.3, 5.2]),
        ("h2", "Non-Functional Requirements"),
        ("tab", "nfr", "Non-functional requirements with the measured outcome.",
         ["ID", "Category", "Requirement", "Measured outcome"],
         [["NFR-1", "Performance", "Quick check under 10 ms per face on a CPU", f"{s3['cpu_onnx_latency_ms_bs1']:.1f} ms per face (ONNX, 4 threads)"],
          ["NFR-2", "Performance", "A 10 s video analysed in under 30 s (Quick) and 60 s (Careful)", "3–5 s per 32-frame video in both modes"],
          ["NFR-3", "Accuracy", "Cross-dataset video AUC above the Swin-B baseline", f"F1 {R.auc('F1', 'Cross-dataset')} vs B1 {R.auc('B1', 'Cross-dataset')}"],
          ["NFR-4", "Robustness", "Graceful degradation under JPEG quality 30", f"F1 {R.auc('F1', 'jpeg:30', '3. Robust')} at JPEG 30"],
          ["NFR-5", "Explainability", "Most of the decision explained by named concepts", "Concept share about 0.8 for F1"],
          ["NFR-6", "Reliability", "Invalid input never crashes the app; a clear message is shown", "No-face, wrong type, oversize tested"],
          ["NFR-7", "Usability", "Verdict understandable without technical knowledge; works at phone width", "Plain-language page; no horizontal scroll at 390 px"],
          ["NFR-8", "Privacy", "Media processed locally; nothing uploaded to third parties", "Local server on 127.0.0.1"],
          ["NFR-9", "Reproducibility", "Every number traceable to a run, commit and seed", "experiments.csv; frozen code snapshots"],
          ["NFR-10", "Maintainability", "Modular code with automated tests", "242 automated tests"]],
         [1.6, 2.6, 6.4, 6.4]),
        ("h2", "External Interface Requirements"),
        ("h3", "User Interface"),
        ("p", "A single web page served by Gradio [@gradio]. It contains an upload area, a choice between a Careful and a "
              "Quick check, a verdict card, clue cards, a 'what you can do now' section, and a collapsible technical "
              "panel (Section 7.6, Appendix B)."),
        ("h3", "Software Interfaces"),
        ("p", "The application exposes five HTTP endpoints through Gradio's API: /check, /clear, /rescore, /overlay_concept "
              "and /overlay_opacity (Table {tab:api}). Models are loaded through PyTorch [@pytorch] and ONNX Runtime "
              "[@onnxruntime]; nearest-neighbour evidence uses a FAISS index [@faiss]."),
        ("h3", "Hardware Interfaces"),
        ("p", "The system needs no special hardware beyond a CUDA GPU for training and for the Careful check. Video decoding "
              "and H.264 re-encoding use the FFmpeg binary bundled with imageio-ffmpeg."),
        ("h2", "Data Requirements"),
        ("p", "Table {tab:data} lists the face-crop data used. Each crop is a 224 × 224 RGB image of an aligned face. "
              "FF++ is used at the c23 compression level; on the project machine it contains only part of the original "
              "Face2Face and FaceSwap videos, which the sampler compensates for by balancing batches across method and "
              "class. " + __import__("data").cdf_note()),
        ("tab", "data", "Face-crop data per split (videos and crops).",
         ["Split", "Method", "Class", "Videos", "Crops"], _counts_rows(), [3.2, 4.6, 2.4, 3.0, 3.8]),
        ("h2", "Hardware and Software Requirements"),
        ("tab", "hwsw", "Hardware and software used for development and required for deployment.",
         ["Item", "Development / training", "Deployment (application)"],
         [["Processor", "32-thread Intel workstation CPU", "Any 64-bit CPU (Quick check)"],
          ["GPU", "NVIDIA RTX 4500 Ada, 24 GB", "Optional; ≥ 8 GB for the Careful check"],
          ["Memory / storage", "64 GB RAM; about 200 GB of datasets and caches", "8 GB RAM; about 1.5 GB of model files"],
          ["Operating system", "Windows 11 Pro", "Windows, Linux or macOS"],
          ["Language and frameworks", "Python 3.12, PyTorch 2.11 (CUDA 12.8), timm, OpenCLIP, PEFT", "Python 3.12, PyTorch or ONNX Runtime"],
          ["Libraries", "face-alignment, OpenCV, FAISS, pandas, scikit-learn, matplotlib, pytest", "Gradio 6, OpenCV, face-alignment"],
          ["Tools", "Git, VS Code, FFmpeg (imageio-ffmpeg)", "A web browser"]],
         [3.6, 7.0, 6.4]),
    ]


def ch5():
    return [
        ("chapter", "High-Level Design"),
        ("p", "This chapter describes the architecture of the system at the level of subsystems and their interactions. "
              "Chapter 6 then specifies the modules and interfaces in detail."),
        ("h2", "Design Goals"),
        ("bul", [
            "**Separation of evaluation from training.** The data layer enforces split integrity, and evaluation is one "
            "shared program, so no model can be scored on data it was selected on.",
            "**Explanation by construction.** The decision of every GRACE model is computed from named concept scores, so "
            "the explanation shown to the user is the computation, not an approximation of it.",
            "**One model family, two deployment targets.** The teacher and the student produce the same outputs (logit, "
            "concept scores, concept maps, contributions), so the application treats them interchangeably.",
            "**Reproducibility and automation.** Every experiment is a configuration plus a seed, executed by a queue and "
            "logged with its code commit.",
            "**Extensibility.** New backbones, branches, concepts or modalities can be added without changing the "
            "application or the evaluation suite.",
        ]),
        ("h2", "System Context and Deployment"),
        ("p", "Figure {fig:context} shows the two halves of the system. Offline, the training and evaluation pipeline turns "
              "the datasets into leak-free splits, trains and evaluates models, and packages the selected teacher and "
              "student into a release folder. Online, the web application loads the released models and serves users "
              "through a browser. Both halves run on the same workstation; the online half can also run on a CPU-only "
              "machine with the student alone."),
        ("fig", "context", FIG / "hld_context_deployment.png", "System context and deployment view.", 16.5),
        ("h2", "Layered Architecture"),
        ("p", "The software is organised in five layers (Figure {fig:layers}). Each layer depends only on the layers "
              "below it. The model layer does not know about the application, and the data layer does not know about "
              "the models."),
        ("fig", "layers", FIG / "hld_layers.png", "Layered architecture and the main modules in each layer.", 16.5),
        ("h2", "Major Components"),
        ("tab", "components", "Major components, their responsibilities and their main source files.",
         ["Component", "Responsibility", "Main files"],
         [["Split builder", "Leak-free CSV splits, counts, overlap checks", "data/make_splits.py, tests/test_splits.py"],
          ["Data engine", "Loading, augmentation, self-blending with recipe, balanced batches, degradations", "data/dataset.py, sbi.py, sampler.py, degrade.py"],
          ["Teacher model", "CLIP + LoRA, DCT branch, gated fusion, concept bottleneck", "models/grace.py, clip_lora.py, frequency.py, fusion.py, concepts.py"],
          ["Baselines", "Swin-B (with optional DANN), CLIP linear head", "models/swin.py, domain.py"],
          ["Student model", "EfficientNet-B0 with the same heads", "models/student.py"],
          ["Training", "Training loop, losses, EMA, model selection", "train.py, models/losses.py, models/ema.py"],
          ["Distillation", "Teacher cache and student training on degraded inputs", "scripts/cache_teacher.py, data/teacher_cache.py, distill.py"],
          ["Evaluation", "Metrics, robustness grid, calibration, figures", "evaluate.py, scripts/eval_all.py, calibrate.py"],
          ["Explanation", "Concept maps, interventions, Grad-CAM, rollout, retrieval, video cues", "explain/*.py, scripts/explain_eval.py"],
          ["Application", "Face cropping, pose filter, engines, result page, report, API", "app/pipeline.py, app/app.py"],
          ["Orchestration", "Job queue, code snapshots, tables, release", "queue_runner.py, scripts/run_snapshot.py, make_tables.py"]],
         [3.3, 7.2, 6.5]),
        ("h2", "Data Flow"),
        ("h3", "Training Flow"),
        ("p", "During training (Figure {fig:training}), real FF++ faces, self-blended faces with recipe labels and FF++ "
              "fakes are drawn in balanced proportions. The teacher is trained with a classification loss, concept and "
              "boundary losses on self-blended faces, and penalties that keep the decision sparse and concept-driven. In "
              "a second stage, the teacher's outputs on clean faces are cached, and the student learns to reproduce them "
              "from degraded copies of the same faces."),
        ("fig", "training", FIG / "diagram2_training.png", "Two-stage training: teacher (stage 1) and explanation-preserving distillation (stage 2).", 13.0),
        ("h3", "Inference Flow"),
        ("p", "At inference (Figure {fig:inference}), the application samples up to 32 frames, detects and aligns the face "
              "in each, scores each face with the selected model, discards faces whose head pose lies outside the "
              "training range, averages the remaining probabilities, calibrates them and maps the result to a verdict "
              "band. The concept scores and maps of the same forward pass provide the explanation."),
        ("fig", "inference", FIG / "diagram1_video_to_verdict.png", "From video to verdict: frame sampling, face pipeline, per-frame model, aggregation, calibration and outputs.", 12.0),
        ("h2", "Key Design Decisions"),
        ("tab", "decisions", "Key design decisions and the alternatives considered.",
         ["Decision", "Alternatives considered", "Reason for the choice"],
         [["Frozen CLIP ViT-L/14 + LoRA", "Fully fine-tuned Swin-B (Phase 2); Xception", "Better transfer to unseen data with 1.6 M trainable parameters (H1)"],
          ["Concept bottleneck with recipe labels", "Linear head + post-hoc Grad-CAM", "Explanations become part of the decision; no manual labels"],
          ["Block-DCT frequency branch", "Global FFT branch; no frequency branch", "Matches the 8×8 transform of JPEG/H.264 (H3)"],
          ["Gated fusion", "Concatenation", "Lets the model weight frequency evidence per sample"],
          ["Final model chosen on a hard validation score", "Choose on clean validation AUC", "Clean validation AUC saturates near 1.0 and cannot rank models"],
          ["Distil explanations on degraded inputs", "Plain logit distillation; no teacher", "Transfers concepts and maps to a CPU model (H4)"],
          ["Video score = mean of front-facing frames", "Max or temporal model", "Simple, robust; extreme poses cause false alarms"],
          ["Three-band verdict", "Binary decision", "An explicit 'can't tell' band avoids over-confident answers"]],
         [4.6, 5.4, 7.0]),
        ("h2", "Extensibility and Scalable Multimodal Integration"),
        ("p", "The decision layer is a linear function of named concept scores plus a small residual (Section 8.5). This "
              "makes it the natural integration point for further evidence. An audio or audio-visual branch, for example "
              "a lip-speech synchronisation detector, can contribute additional concepts such as 'mouth movement does not "
              "match speech'. These enter the same bottleneck with their own weights, and the user interface lists them "
              "next to the visual clues without any change to its logic. Because both engines implement the same "
              "interface (a list of faces in, a list of per-face results out), a new modality or a new model can also be "
              "added as a separate engine and combined at the video level. The experiment infrastructure is "
              "model-agnostic: any new configuration is trained, evaluated on the same splits and degradations, and "
              "compared with its parent automatically."),
        ("h2", "Security, Privacy and Ethical Considerations"),
        ("bul", [
            "Uploaded media is processed locally and stored only in a temporary folder; the report records the file's "
            "SHA-256 hash so that a result can be tied to the exact file analysed.",
            "Results are presented as evidence with an explicit 'can't tell' band and a statement of limitations, not as "
            "proof.",
            "Deepfake datasets are demographically skewed, so error rates may differ between groups. This is listed as a "
            "limitation and as future work.",
            "A public detector can be used to tune evasive fakes. The system is intended for research and assisted "
            "verification, not as an automated moderation tool.",
        ]),
    ]


def ch6():
    return [
        ("chapter", "Low-Level Design"),
        ("p", "This chapter specifies the modules introduced in Chapter 5: their classes and functions, inputs and outputs, "
              "data structures, file formats and error handling."),
        ("h2", "Data Module"),
        ("tab", "lld_data", "Data module: main classes and functions.",
         ["Class / function", "Input", "Output / behaviour"],
         [["make_splits.main()", "Face-crop folders, official FF++ JSON splits, Celeb-DF test list", "splits/*.csv (one row per crop), counts.md; fails on any overlap"],
          ["FaceFrameDataset", "Split CSV, augmentation config, SBI function, degradation function", "Dict per item: image, label, concepts, concept_mask, boundary, boundary_mask, domain_id, video_id"],
          ["CellBalancedBatchSampler", "Cell indices (real, sbi, Deepfakes, ...), weights, batch size", "Batches with a fixed quota per cell; small cells are cycled"],
          ["SBIMaker / make_sbi()", "Real face, 68 landmarks, random generator", "Fake image, 10-d concept vector, 16×16 boundary map, recipe dict"],
          ["degrade.ImageDegrade", "Spec 'jpeg:q', 'down:s' or 'h264:crf'", "Degraded image; H.264 cached per video"],
          ["StudentDegrade", "Clean image", "JPEG 20–60 and/or 0.25–0.5× resampling"],
          ["TeacherCache", "cache/teacher shards and index", "Teacher logit, z, concepts and maps per frame path"]],
         [4.2, 5.8, 7.0]),
        ("h2", "Model Module"),
        ("p", "Figure {fig:classes} shows the main classes. GraceDetector composes five parts; the student reuses the "
              "same concept head and decision layer on a different backbone, and the domain head is attached only in "
              "ablation models. Table {tab:shapes} gives the tensor shapes for one 224 × 224 face."),
        ("fig", "classes", FIG / "lld_classes.png", "Class diagram of the model module.", 16.5),
        ("tab", "shapes", "Tensor shapes through the teacher and student (batch dimension omitted).",
         ["Stage", "Teacher (F1)", "Student (S3)"],
         [["Input", "3 × 224 × 224, values in [0, 1]", "3 × 224 × 224 (degraded during training)"],
          ["Backbone", "CLIP ViT-L/14: class token 1024, patch grid 1024 × 16 × 16", "EfficientNet-B0: feature map 1280 × 7 × 7"],
          ["Frequency branch", "Y channel → 64 × 28 × 28 DCT coefficients → 256", "–"],
          ["Fused feature z", "512 (gated fusion + LayerNorm)", "512 (projection, matched to teacher z)"],
          ["Concept maps", "10 × 16 × 16", "10 × 7 × 7"],
          ["Concept scores", "10 (mean of top 10% of each map)", "10"],
          ["Decision", "logit = wᵀσ(c) + b + r(z)", "same form"],
          ["Parameters", "306.9 M in total (1.6 M LoRA trainable in the backbone)", "4.71 M; 0.39 GFLOPs"]],
         [3.0, 7.5, 6.5]),
        ("h2", "Training and Distillation Module"),
        ("p", "train.py builds the model and data loaders from a YAML configuration, runs warm-up followed by cosine "
              "learning-rate decay with AdamW [@adamw] and separate learning rates for each parameter group, keeps an "
              "exponential moving average (EMA) of the weights, evaluates the EMA weights on FF++ validation after every "
              "epoch, and saves the best checkpoint. compute_loss() in models/losses.py combines the terms of Equation "
              "{eq:teacherloss} and masks the concept and boundary terms for samples that have no recipe. For students, "
              "distill.py reuses train.py with the distillation terms of Equation {eq:kdloss} and reads the teacher's "
              "outputs from TeacherCache instead of running the teacher. Before training starts, the cache is checked "
              "against a fresh forward pass on 100 random crops (maximum difference below 10⁻³)."),
        ("h2", "Evaluation Module"),
        ("p", "evaluate.py loads a checkpoint, predicts every crop of a test split (optionally degraded), aggregates crops "
              "into videos by averaging probabilities, and computes frame- and video-level AUC, equal error rate, and "
              "accuracy, precision, recall and F1 at the threshold that maximises balanced accuracy on FF++ validation. "
              "It writes metrics_<split>.json, predictions_<split>.csv, a ROC curve and per-video cues into the run "
              "folder. eval_all.py runs this for every model, seed, test set and degradation and writes "
              "results/robustness.csv; make_tables.py builds every table in Chapter 10 from these files."),
        ("h2", "Explanation Module"),
        ("tab", "lld_explain", "Explanation module functions.",
         ["Function", "What it computes"],
         [["concept_maps(model, x)", "The 10 concept maps, upsampled to 224 × 224 and normalised"],
          ["contributions(model, x)", "Per-concept contribution w_k σ(c_k) and the residual r(z)"],
          ["intervene(model, x, off)", "The logit with selected concepts switched off (their contribution removed)"],
          ["gradcam / SwinGradCAM", "Grad-CAM [@gradcam] for the Swin baseline; Grad-CAM++ [@gradcampp] for the student"],
          ["attention_rollout", "Attention rollout [@rollout] for the CLIP baseline"],
          ["TrainingIndex.nearest(z, k)", "The k most similar training faces in the teacher's feature space (FAISS)"],
          ["video_cues", "Concept flicker over time and identity drift of the CLIP class token"]],
         [5.2, 11.8]),
        ("h2", "Application Module"),
        ("p", "The application is split into a pipeline that has no user-interface code (app/pipeline.py) and a "
              "presentation layer (app/app.py). Figure {fig:sequence} shows the sequence of calls for one analysis."),
        ("fig", "sequence", FIG / "lld_sequence_app.png", "Sequence diagram of one analysis request.", 16.0),
        ("tab", "lld_app", "Application module: main classes and functions.",
         ["Class / function", "Responsibility"],
         [["sample_frames(path, n=32)", "Evenly spaced frames from a video (OpenCV)"],
          ["FaceCropper.crop_with_pose", "SFD face detection, eye-level rotation, 224 × 224 crop (margin 1.02 × box), yaw and pitch from landmarks"],
          ["pose_ok(yaw, pitch)", "True if the pose lies inside the 0.1st–99.9th percentile range of FF++ training faces"],
          ["QuickEngine", "Student via ONNX Runtime on the CPU; concept weights read from a JSON file"],
          ["DeepEngine", "Final teacher on the GPU (PyTorch), with class-token features for retrieval"],
          ["analyse(path, engine, cropper)", "Runs the pipeline and returns an Analysis (faces, probability, verdict, messages)"],
          ["band(p)", "Likely real (p < 0.35), Can't tell (0.35–0.65), Likely fake (≥ 0.65)"],
          ["hero_html, reasons_html", "Plain-language verdict card and the three clue cards"],
          ["build_report", "Self-contained HTML report with hash, timeline, maps and contributions"]],
         [5.0, 12.0]),
        ("tab", "api", "HTTP API endpoints of the application.",
         ["Endpoint", "Inputs", "Outputs"],
         [["/check", "file, check type, concept name, overlay opacity", "Verdict card, clue cards, technical views, report file"],
          ["/clear", "–", "Empty page state"],
          ["/rescore", "check type, concepts to switch off", "Re-scored probability and verdict"],
          ["/overlay_concept", "concept name, opacity", "Gallery of suspicious faces with that concept's map"],
          ["/overlay_opacity", "concept name, opacity", "Same gallery with a new opacity"]],
         [3.4, 6.4, 7.2]),
        ("h2", "Data Structures and File Formats"),
        ("tab", "formats", "Main files and their formats.",
         ["File", "Format and content"],
         [["splits/<split>.csv", "video_id, frame_path, label, dataset, method, source_id, landmarks_path (one row per crop)"],
          ["configs/<model>.yaml", "name, parent, notes, model{...}, data{...}, loss{...}, optim{...}, select{...}"],
          ["runs/<name>_s<seed>_<time>/", "config.yaml, best.pth, train.log, metrics_*.json, predictions_*.csv, roc_*.png"],
          ["experiments.csv", "run_id, config, parent, seed, commit, start, end, gpu_hours, best_epoch, val and test AUCs, notes"],
          ["cache/teacher/", "shard_XX.npz (logit, z, concepts, maps per crop), index.csv, teacher.json"],
          ["results/robustness.csv", "model, seed, run, split, degrade, video_auc, frame_auc, eer, acc, f1, concept_share"],
          ["release/", "Teacher and student checkpoints, ONNX model, temperatures, thresholds, run descriptions"]],
         [4.6, 12.4]),
        ("h2", "Error Handling"),
        ("tab", "errors", "Error conditions and how they are handled.",
         ["Condition", "Handling"],
         [["Two splits share a video, source ID or identity", "Split tests fail and training does not start"],
          ["Teacher cache differs from a fresh forward pass", "Distillation stops with an error"],
          ["A queued job fails", "Marked failed; jobs that depend on it are marked blocked; the queue continues"],
          ["Upload missing, wrong type or larger than 200 MB", "Friendly message card; no analysis is run"],
          ["No face, or fewer than 4 faces in a video", "'We couldn't find a face' message; no verdict"],
          ["Too few front-facing faces", "Verdict forced to 'Can't tell' with an explanation"],
          ["Quick model missing", "Message asking the user to use the Careful check"]],
         [7.0, 10.0]),
    ]


def ch7():
    return [
        ("chapter", "System Design (Detailed)"),
        ("p", "This chapter describes the complete GRACE system in detail: the teacher, the student, the video-level "
              "inference, the explanation subsystem, the user interface and the experiment infrastructure."),
        ("h2", "GRACE Architecture"),
        ("p", "Figure {fig:arch} shows the architecture. Each aligned face passes through two branches. The CLIP branch "
              "provides a global class token and a 16 × 16 grid of patch tokens; the DCT branch provides a compact "
              "description of the face's compression-domain statistics. A gated fusion combines the class token with the "
              "frequency feature into a 512-dimensional vector z. A concept head turns the patch grid into ten concept "
              "maps and scores, and the decision layer combines the concept scores, plus a small residual from z, into "
              "the fake logit."),
        ("fig", "arch", FIG / "grace_architecture_overview.png", "GRACE architecture. The dashed path marks training-time distillation into the student.", 14.0),
        ("h2", "Teacher Components"),
        ("h3", "Backbone"),
        ("p", "The OpenAI CLIP ViT-L/14 image encoder [@clip] (through OpenCLIP [@openclip]) is kept frozen. LoRA adapters "
              "[@lora] of rank 16 and scale 32 are inserted into the query-key-value projections of all 24 transformer "
              "blocks. This adds about 1.6 M trainable parameters. Only the adapters are saved with a checkpoint (12.6 MB)."),
        ("h3", "Frequency Branch"),
        ("p", "The luminance channel is divided into 8 × 8 blocks, and each block is transformed with the orthonormal 2-D "
              "DCT-II, the same transform that JPEG and H.264 quantise. The 64 coefficients of the 28 × 28 blocks form a "
              "64-channel image, which is log-compressed, standardised and encoded by a small convolutional network into "
              "a 256-dimensional vector. An alternative branch based on the global 2-D FFT is used in an ablation."),
        ("h3", "Gated Fusion"),
        ("p", "The class token and the frequency vector are projected to 512 dimensions. A sigmoid gate computed from both "
              "projections decides, per dimension, how much of each to keep (Equation {eq:gate}). The mean gate value "
              "is logged during evaluation as an indicator of how much the model relies on CLIP features."),
        ("h3", "Concept Head and Decision Layer"),
        ("p", "A 1 × 1 convolution maps the 1024-channel patch grid to ten concept maps. Each concept score is the mean of "
              "the top 10% of its map, so a concept is detected when a small region shows strong evidence. The decision "
              "layer computes the logit as a weighted sum of the sigmoid concept scores plus a bias and a residual r(z) "
              "from a small zero-initialised MLP. Penalties on r(z) and on the L1 norm of the weights keep the decision "
              "concept-driven and sparse."),
        ("h2", "Student Design"),
        ("p", "The student is an EfficientNet-B0 [@efficientnet] from timm [@timm] with the same concept head (on its 7 × 7 "
              "feature grid) and decision layer, and a projection that maps its pooled feature to the teacher's z. It "
              "has 4.71 M parameters and needs 0.39 GFLOPs per face, compared with 306.9 M parameters and 81.5 GFLOPs for "
              "the teacher. It is exported to ONNX (opset 18) through the torch.export-based exporter. The exported model "
              "matches PyTorch to within 2 × 10⁻⁶; the legacy exporter differed by about 2 logits and was not used."),
        ("h2", "Video-Level Inference and Calibration"),
        ("p", "For a video, 32 evenly spaced frames are sampled, and the largest face in each frame is detected with the "
              "SFD detector, rotated so that the eyes are level, and cropped at 1.02 times the detection box (the ratio "
              "measured on training crops). Head yaw and pitch are estimated from the 68 landmarks [@fan]. Faces whose "
              "pose lies outside the range covered by 99.8% of FF++ training faces are kept for display but excluded "
              "from the verdict, because such poses produced false alarms on real phone videos. The video probability "
              "is the mean of the remaining face probabilities after temperature scaling [@guo] (Equation "
              "{eq:temp}). The fitted temperatures are 1.04 for the teacher and 1.66 for the student. The verdict "
              "is 'Likely real' below 0.35, 'Can't tell' between 0.35 and 0.65, and 'Likely fake' above 0.65."),
        ("h2", "Explainability Subsystem"),
        ("p", "Because the decision is computed from concept scores, three kinds of explanation come directly from the "
              "forward pass: (i) the contribution of each concept to the logit, (ii) the concept's spatial map, and (iii) "
              "the effect of switching a concept off, which recomputes the logit without that concept's contribution. "
              "Video-level cues add the temporal flicker of the concept scores and the identity drift of the CLIP "
              "class token across frames. For expert users, the three training faces closest to the analysed face in "
              "the teacher's feature space are retrieved with FAISS [@faiss] and shown as evidence. For comparison, "
              "Grad-CAM and attention rollout are computed for the baselines."),
        ("h2", "User Interface Design"),
        ("p", "The interface follows the 'answer first, evidence on demand' principle recommended for explainable systems "
              "aimed at non-experts [@laylens; @dfp2e]. Figure {fig:ui} shows the result page. A verdict card gives "
              "the answer in plain words (for example 'This video is likely fake'), a three-part 'How sure are we?' "
              "scale, and how many views of the face were used. Below it, three clue cards show the concepts that "
              "contributed most, each named in everyday language ('The edge around the face is uneven'), tagged as a "
              "strong or some clue, and drawn on the face as an orange glow from that concept's map. A 'What you can do "
              "now' card gives three practical steps. All technical views (probability, timeline, concept maps, "
              "contribution chart, switch-off test, nearest training faces, report) are in a collapsible panel. Colours "
              "pair blue (real) with orange (fake), which colour-blind users can distinguish, and every state is also "
              "named in words."),
        ("fig", "ui", FIG / "app_result.png", "Result page of the application for a fake test video (Careful check).", 13.0),
        ("h2", "Experiment Infrastructure and Reproducibility"),
        ("p", "All experiments were run by a dependency-aware job queue (Figure {fig:workflow}). Each job freezes a copy "
              "of the code before it starts, so later edits cannot change a running experiment. It trains or evaluates "
              "one configuration with one seed and appends a row to experiments.csv with the commit, seed, GPU hours, "
              "best epoch and test results. A failed job is marked as such, its dependants are blocked, and the queue "
              "continues with independent work, so experiments ran unattended without wasting GPU time. Tables and "
              "figures are generated from the result files by scripts, never typed by hand."),
        ("fig", "workflow", FIG / "sys_experiment_workflow.png", "Experiment workflow from queue to tables and release.", 16.5),
    ]
