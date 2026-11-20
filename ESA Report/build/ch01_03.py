"""Abstract, Chapter 1 (Introduction), Chapter 2 (Problem Definition), Chapter 3 (Literature Review)."""
from data import R


def abstract():
    f1c, f1d = R.val("F1", "Celeb-DF"), R.val("F1", "DFD")
    b1x, f1x = R.val("B1", "Cross-dataset"), R.val("F1", "Cross-dataset")
    return [
        "Deepfake detectors often score almost perfectly on the dataset they were trained on, yet fail on videos from new "
        "sources, lose accuracy on compressed uploads, and rarely explain their decisions. This project presents GRACE, a "
        "face-forgery detector built for generalisation, robustness and built-in explanation. A CLIP ViT-L/14 backbone "
        "adapted with LoRA is fused with a JPEG-aligned DCT frequency branch, and every decision passes through ten named "
        "forensic concepts that are learned from self-blended training images whose synthesis recipe supplies the labels.",
        f"A 4.7-million-parameter student, distilled from the teacher on degraded frames, runs on a CPU in about 4 ms per "
        f"face. Trained on FaceForensics++ and tested on unseen DFD and Celeb-DF v2 under a leak-free protocol, the final "
        f"model reaches video AUCs of {f1d:.3f} and {f1c:.3f} (cross-dataset mean {f1x:.3f} against {b1x:.3f} for a Swin-B "
        f"baseline). A plain-language web application presents each verdict with its evidence.",
    ]


def ch1():
    return [
        ("chapter", "Introduction"),
        ("h2", "Background"),
        ("p", "A deepfake is an image or video in which a person's face has been replaced, re-animated or otherwise edited "
              "by a generative model so that the result looks authentic. Open-source face-swapping tools and modern "
              "generators have made such manipulations cheap to produce and hard to spot by eye. They are now used for "
              "misinformation, impersonation fraud, harassment and the manipulation of public opinion, which makes "
              "automatic detection an important part of media verification and digital forensics."),
        ("p", "Research on detection has progressed quickly on curated benchmarks such as FaceForensics++ [@ffpp] and "
              "Celeb-DF [@celebdf]. Convolutional and transformer classifiers routinely report accuracies close to 100% "
              "when they are trained and tested on the same benchmark. Benchmark studies such as DeepfakeBench "
              "[@deepfakebench] show, however, that the same models lose much of their accuracy when they are tested on "
              "manipulations or datasets they did not see during training. They also tend to rely on fine high-frequency "
              "traces that disappear when a video is re-compressed by a social-media platform."),
        ("h2", "Motivation"),
        ("p", "Three practical gaps motivate this project. First, **generalisation**: a detector is useful only if it works on "
              "videos from sources other than its training set. Second, **robustness**: real videos are resized and "
              "re-encoded at low bitrates, so a detector must tolerate JPEG and H.264 compression. Third, **explanation**: a "
              "single fake probability gives an investigator or an ordinary user no reason to trust or question the "
              "result. Heatmaps produced after training show where a model looked, but not what it found, and they are "
              "not guaranteed to reflect the computation that produced the score [@tsigos; @gradcam]."),
        ("p", "A fourth, methodological motivation emerged during this phase. When the project's earlier pipeline was "
              "audited, its very high validation accuracy turned out to be inflated by overlap between training and "
              "validation data. A detection study is only as good as its evaluation protocol, so building a leak-free, "
              "reproducible protocol became a goal in its own right."),
        ("h2", "Scope of the Work"),
        ("p", "The work covers visual (frame-based) detection of face manipulations in images and short videos. It includes "
              "the data protocol, the detection models, their training and evaluation, the explanation methods, a "
              "compressed model for CPU deployment and a web application. Audio and audio-visual (lip-sync) analysis are "
              "outside the experimental scope of this phase. The design keeps a clear place for an additional modality "
              "(Chapters 5 and 11), which is how the *scalable multimodal integration* in the project title is addressed."),
        ("p", "The models are trained only on FaceForensics++ (FF++) [@ffpp] and evaluated on its test split (in-domain) and "
              "on two datasets never used for training: Google's DeepFakeDetection (DFD) [@dfd] and Celeb-DF v2 "
              "[@celebdf]."),
        ("h2", "Relation to Previous Phases and Present Development"),
        ("p", "In Phase 2 the team trained a Swin Transformer [@swin] classifier with Grad-CAM explanations and reported a "
              "peak validation accuracy of 98.98% on Celeb-DF v2. In this phase the pipeline was audited end to end. The "
              "audit found that frames of the same videos had been placed in both the training and the validation sets, "
              "so the earlier figure measured memorisation rather than detection (Section 8.3). Phase 3 therefore "
              "rebuilt the evaluation protocol from scratch and redesigned the detector. The result is GRACE "
              "(Generalisable, Robust And Concept-Explainable detection), which draws on recent work on foundation-model "
              "features [@univfd; @fa], self-blended training data [@sbi; @facexray], concept bottleneck models "
              "[@cbm] and knowledge distillation [@hinton]."),
        ("h2", "Contributions"),
        ("bul", [
            "**A leak-free, reproducible evaluation protocol**: official FF++ and Celeb-DF splits, automated overlap tests, "
            "model selection on FF++ validation only, and video-level metrics from a single evaluation script, with every "
            "run logged with its code commit and seed.",
            "**A recipe-supervised concept bottleneck**: self-blended forgeries record how they were made, and these "
            "records label ten forensic concepts. The real/fake decision is a weighted sum of the concept scores, so the "
            "explanation is part of the computation and needs no manual annotation.",
            "**A generalisable teacher**: a frozen CLIP ViT-L/14 [@clip] adapted with LoRA [@lora], fused through a learned "
            "gate with a JPEG-aligned DCT frequency branch.",
            "**Explanation-preserving, compression-robust distillation** into a 4.7 M-parameter EfficientNet-B0 [@efficientnet] "
            "student that sees only degraded frames and runs on a CPU.",
            "**A controlled comparison study** of 17 model configurations, most with three random seeds, covering "
            "cross-dataset accuracy, compression robustness, efficiency and explanation quality. Negative results are "
            "reported as well as positive ones.",
            "**A plain-language web application** that shows a verdict, the clues behind it and what the user can do next, "
            "with every technical view available on demand.",
        ]),
        ("h2", "Organisation of the Report"),
        ("p", "Chapter 2 defines the problem and objectives, and Chapter 3 reviews the literature. Chapters 4 to 7 give the "
              "requirements specification, the high-level and low-level designs, and the detailed system design. Chapter 8 "
              "describes the methodology and Chapter 9 the implementation with pseudocode. Chapter 10 presents and "
              "discusses the experimental results, and Chapter 11 concludes with future work. The appendices define the "
              "acronyms and give a user manual for the application."),
    ]


def ch2():
    return [
        ("chapter", "Problem Definition"),
        ("h2", "Problem Statement"),
        ("p", "Given a photograph or a short video that contains a human face, decide whether the face has been manipulated. "
              "The system must work on videos from sources that were not used for training and on compressed or resized "
              "uploads. It must explain its decision in terms that a non-expert can check and an expert can verify, and it "
              "must be usable without specialised hardware."),
        ("p", "Formally, let a video be a sequence of aligned face crops x_{1}, …, x_{T}. The task is to learn a function that "
              "outputs a calibrated probability p ∈ [0, 1] that the video is fake, together with an explanation E. The "
              "function is trained on a source distribution (FF++) and evaluated on target distributions (DFD, Celeb-DF) "
              "that differ in actors, recording conditions and manipulation methods. E consists of named concept scores "
              "and spatial maps that are part of the computation of p, not an approximation added afterwards."),
        ("h2", "Objectives"),
        ("num", [
            "Establish a leak-free training and evaluation protocol with automated checks, and quantify how much the "
            "earlier protocol overstated accuracy.",
            "Build a detector whose cross-dataset video AUC (mean of DFD and Celeb-DF) clearly exceeds that of a Swin-B "
            "baseline trained under the same protocol.",
            "Make every decision pass through named forensic concepts whose labels come from the training-data synthesis "
            "itself, and measure how much of the decision the concepts explain.",
            "Measure robustness to JPEG compression, downscaling and H.264 re-encoding, and design the model to degrade "
            "gracefully.",
            "Produce a model small enough for CPU inference (under 10 ms per face) that keeps as much accuracy and "
            "explanation quality as possible.",
            "Evaluate explanations quantitatively against ground-truth manipulation masks and against standard post-hoc "
            "methods.",
            "Deliver a web application that presents results in plain language, with technical detail on demand.",
        ]),
        ("h2", "Research Questions and Hypotheses"),
        ("bul", [
            "**H1 (backbone).** A frozen CLIP backbone adapted with LoRA narrows the generalisation gap relative to a fully "
            "fine-tuned Swin-B.",
            "**H2 (concepts).** Routing the decision through a concept bottleneck costs little or no cross-dataset accuracy "
            "while making most of the decision explainable by named concepts.",
            "**H3 (frequency).** A block-DCT branch aligned with JPEG/H.264 helps more than a global FFT branch.",
            "**H4 (distillation).** A student distilled from the teacher on degraded inputs keeps more accuracy under "
            "compression, and agrees more closely with the teacher's explanations, than a student trained without a "
            "teacher.",
        ]),
        ("h2", "Scope, Assumptions and Constraints"),
        ("bul", [
            "Visual frames only; one face per frame (the largest detected face).",
            "Training data is the c23 (light compression) version of FF++ available on the project machine, which contains "
            "Deepfakes, Face2Face and FaceSwap manipulations (NeuralTextures and FaceShifter are absent).",
            "All selection decisions (checkpoints, thresholds, the choice of final model) use FF++ validation data only. "
            "Test sets are used once, for reporting.",
            "Compute budget: one workstation GPU (NVIDIA RTX 4500 Ada, 24 GB) over a seven-day experimental sprint.",
            "Licensed datasets are used only as provided to the team and are never redistributed.",
        ]),
        ("h2", "Success Criteria"),
        ("p", "The project is successful if (i) the leak-free protocol is enforced by automated tests, (ii) the final "
              "teacher's cross-dataset video AUC exceeds the Swin-B baseline's, (iii) concepts account for most of the "
              "teacher's decision variance, (iv) a CPU model with latency under 10 ms per face is delivered, and (v) the "
              "application gives correct verdicts on held-out test videos and handles invalid input gracefully."),
    ]


LIT = [
    ("ffpp", "FaceForensics++: Learning to Detect Manipulated Facial Images",
     "Rössler et al. [@ffpp] released FaceForensics++, the most widely used benchmark for face-manipulation detection. "
     "It contains 1,000 YouTube videos, each manipulated with several methods, including the graphics-based Face2Face "
     "and FaceSwap and the learning-based Deepfakes and NeuralTextures. Each video is provided at three compression "
     "levels (raw, c23 and c40).",
     "The authors defined a fixed split by source video and compared hand-crafted and learned detectors on face crops. "
     "An ImageNet-pretrained XceptionNet [@xception], fine-tuned on face crops, was the strongest model and became the "
     "standard baseline.",
     "Accuracy is high on raw and lightly compressed video and drops noticeably at strong (c40) compression, which "
     "shows the sensitivity of learned detectors to re-encoding.",
     "FF++ supplies our training data and in-domain test set, and we use its official split. The benchmark measures only "
     "in-domain performance, which motivates our cross-dataset protocol."),
    ("celebdf", "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics",
     "Li et al. [@celebdf] observed that early deepfake datasets contained visible artefacts that made detection "
     "artificially easy. They built Celeb-DF v2: 590 real celebrity interview videos, 300 additional real YouTube videos, "
     "and 5,639 deepfakes made with an improved synthesis process.",
     "The improved synthesis reduces colour mismatch, temporal flicker and low-resolution blending boundaries. The "
     "dataset defines an official list of 518 test videos.",
     "Detectors that performed well on earlier datasets scored much lower on Celeb-DF, establishing it as a standard "
     "test of generalisation.",
     "We never train on Celeb-DF; it is one of our two unseen test sets. Because all fakes come from one synthesis "
     "method, it tests transfer to one new method rather than to many."),
    ("facexray", "Face X-ray for More General Face Forgery Detection",
     "Li et al. [@facexray] argued that most face manipulations share one step: blending an altered face into a "
     "background image. A detector that finds the blending boundary should therefore generalise across methods.",
     "Training images are created by blending one real face into another, so no real forgeries are needed. The network "
     "predicts a grey-scale boundary image (the 'X-ray') as well as the real/fake label.",
     "Trained only on self-generated blends, the model generalised to unseen manipulations far better than classifiers "
     "trained on the forgeries themselves.",
     "We adopt the idea of synthesising training forgeries and of supervising a boundary map. Face X-ray's limitation "
     "is that it explains only one kind of evidence, the boundary, and compression can erase that boundary."),
    ("sbi", "Detecting Deepfakes with Self-Blended Images",
     "Shiohara and Yamasaki [@sbi] proposed self-blended images (SBI): pseudo-fakes created by blending a face with a "
     "slightly transformed copy of itself. Source and target then share identity, so the only cue left is the blending "
     "artefact.",
     "The source copy is altered by colour, frequency and geometric transforms, and blended through a randomly shaped, "
     "softened landmark mask. An EfficientNet-B4 is trained on real faces and their self-blended versions only.",
     "SBI reported strong cross-dataset results on Celeb-DF and other benchmarks without seeing any real deepfake during "
     "training.",
     "SBI is the basis of our data engine. Our extension is to **record the recipe** of every blend (region, blend "
     "strength, colour shift, blur, resampling, geometry, mask shape) and use it as concept labels. The original SBI "
     "model outputs only a binary score."),
    ("univfd", "Towards Universal Fake Image Detectors that Generalise Across Generative Models",
     "Ojha et al. [@univfd] showed that detectors trained on one generator over-fit to its traces, and that the feature "
     "space of a large pretrained vision-language model, CLIP ViT-L/14 [@clip], separates real and fake images across "
     "many generators.",
     "The CLIP image encoder is kept frozen, and a linear probe or nearest-neighbour classifier is trained on its "
     "features.",
     "Frozen CLIP features generalised to unseen GANs and diffusion models far better than fully trained CNNs.",
     "This motivates our frozen CLIP backbone. UnivFD works on whole images, without adaptation or explanation; we add "
     "low-rank adaptation [@lora], a frequency branch and a concept bottleneck for face forgeries."),
    ("fa", "Forensics Adapter: Adapting CLIP for Generalisable Face Forgery Detection",
     "Cui et al. [@fa] adapt CLIP to face-forgery detection with a small trainable adapter while keeping CLIP's general "
     "knowledge intact.",
     "The adapter interacts with the CLIP encoder and is trained to find blending-boundary traces specific to face "
     "forgery, so that only a small number of parameters change.",
     "The method reported strong cross-dataset results with few trainable parameters.",
     "We follow the same principle of parameter-efficient adaptation, using LoRA on CLIP's attention projections. Like "
     "most CLIP-based detectors, Forensics Adapter outputs a score with no named evidence."),
    ("ganin", "Domain-Adversarial Training of Neural Networks",
     "Ganin et al. [@ganin] introduced domain-adversarial training (DANN) for unsupervised domain adaptation, aiming for "
     "features that are useful for the task but uninformative about the domain.",
     "A domain classifier is attached to the feature extractor through a gradient reversal layer. The extractor is "
     "trained to minimise the task loss while maximising the domain classifier's loss.",
     "DANN improved accuracy on target domains in several adaptation benchmarks.",
     "Our Phase-2 design used DANN, and Phase 3 tests it carefully under the leak-free protocol: with real faces as a "
     "domain, at feature level, at patch level and for unsupervised adaptation to Celeb-DF (Chapter 10). We combine it "
     "with a maximum mean discrepancy term [@mmd]."),
    ("cbm", "Concept Bottleneck Models",
     "Koh et al. [@cbm] proposed models that first predict a set of human-understandable concepts and then predict the "
     "label from those concepts alone.",
     "The concept layer is supervised with concept annotations, and the label layer reads only the concepts. A user can "
     "inspect the concepts and even correct them ('intervene') at test time.",
     "Concept bottlenecks reached accuracy close to standard models while allowing explanation and intervention.",
     "We use a concept bottleneck for deepfake detection. Concept annotations are normally expensive, but in our design "
     "they come for free from the self-blending recipe. We add a small, penalised residual path so that accuracy is not "
     "lost, and we measure the share of the decision that flows through the concepts."),
    ("tsigos", "Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection",
     "Tsigos et al. [@tsigos] noted that explanations for deepfake detectors are usually judged by eye. They proposed a "
     "quantitative way to compare explanation methods.",
     "The regions that an explanation method marks as important are perturbed, and the effect on the detector's "
     "decision is measured. A faithful explanation should mark the regions whose change most affects the decision.",
     "Common post-hoc methods differed considerably in how faithfully they reflected the detector.",
     "This informs our explanation evaluation, which combines localisation (pointing game, mask IoU), faithfulness "
     "(deletion/insertion [@rise]) and intervention tests, with ground-truth masks from held-out self-blended faces."),
    ("dfp2e", "From Prediction to Explanation: An Explainable, Interactive Deepfake Detection Framework for Non-Expert Users",
     "Tariq et al. [@dfp2e] (ACM Multimedia 2025), reviewed in Phase 2, target non-expert users who need to "
     "understand why an image is classified as fake.",
     "A three-stage pipeline combines a deepfake classifier, a visual explanation module based on heatmaps, and a "
     "captioning and language-model module that turns the evidence into a short textual explanation. It is evaluated on "
     "the DF40 benchmark [@df40].",
     "The authors report that layered, plain-language explanations help non-experts. Related work on simplified "
     "explanations [@laylens] reports a similar preference.",
     "We share the goal of plain-language results, which shaped our application (Section 7.6). Unlike this framework, "
     "our explanations come from concepts inside the model rather than from post-hoc heatmaps and a separate language "
     "model, and they run without a large model."),
    ("hinton", "Distilling the Knowledge in a Neural Network",
     "Hinton et al. [@hinton] proposed knowledge distillation: training a small student network to match the softened "
     "output distribution of a large teacher.",
     "The teacher's logits are divided by a temperature T to produce soft targets, and the student minimises a weighted "
     "sum of the usual loss and the divergence from these targets.",
     "Students trained this way recover much of the teacher's accuracy at a fraction of the cost.",
     "We extend distillation in two ways: the student sees degraded frames while the teacher sees clean ones, and the "
     "student also matches the teacher's features, concept scores and concept maps, so that explanations are "
     "transferred as well as decisions."),
    ("deepfakebench", "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection",
     "Yan et al. [@deepfakebench] observed that detectors were compared under inconsistent preprocessing, data and "
     "metrics, which made many claims of generalisation hard to verify.",
     "The benchmark re-implements many detectors with a common pipeline and evaluates them with the same training data "
     "and cross-dataset protocol.",
     "Under the common protocol, the advantage of many specialised detectors over simple baselines shrank, and "
     "cross-dataset performance remained far below in-domain performance.",
     "This supports two of our design choices: comparing every model against its parent under one protocol, and "
     "reporting video-level AUC on FF++, DFD and Celeb-DF with the same script."),
]


def ch3():
    out = [("chapter", "Literature Review"),
           ("p", "This chapter reviews the work that shaped the design of GRACE. Each review covers what the paper "
                 "proposes, how it works, what it showed, and how it relates to this project and the gaps it leaves. "
                 "Section 3.13 notes other related work, Section 3.14 summarises the reviewed work, and Section 3.15 states the "
                 "gaps this project addresses.")]
    for key, title, intro, method, evalu, rel in LIT:
        out += [("h2", title), ("h3", "Introduction"), ("p", intro), ("h3", "Characteristics and Method"), ("p", method),
                ("h3", "Evaluation"), ("p", evalu), ("h3", "Relevance and Gaps"), ("p", rel)]
    out += [
        ("h2", "Other Related Work"),
        ("p", "Several other works reviewed in earlier phases inform specific parts of the design. Temporal methods model "
              "inconsistencies between frames [@ciamarra], and prototype-based multimodal detectors combine audio and "
              "visual evidence [@pudd]; both are relevant to future multimodal extensions. Segmentation-based methods "
              "localise diffusion-based inpainting [@diffseg]. Self-consistency learning [@lu_ssc] and invariant risk "
              "minimisation [@irm] target generalisation. Surveys of deep-learning detection [@rama] and of explainable "
              "detection [@mansoor] stress the difficulty of transfer to real-world data and the need for faithful "
              "explanations. Image-level pipelines such as DeepSight [@deepsight], which combines CNN features with gradient-boosted "
              "classifiers, show that well-tuned conventional classifiers remain strong in-domain baselines."),
        ("h2", "Summary of the Literature"),
        ("tab", "lit", "Summary of the reviewed work and what this project takes from it.",
         ["Work", "Main idea", "Limitation for our goals", "Used in GRACE"],
         [["FF++ [@ffpp]", "Benchmark; Xception baseline", "In-domain only", "Training data, official split"],
          ["Celeb-DF [@celebdf]", "Harder, cleaner deepfakes", "One synthesis method", "Unseen test set"],
          ["Face X-ray [@facexray]", "Blending boundary from synthetic blends", "Only boundary evidence", "Boundary map loss"],
          ["SBI [@sbi]", "Self-blended pseudo-fakes", "Binary output only", "Data engine + recipe labels"],
          ["UnivFD [@univfd]", "Frozen CLIP features", "No adaptation, no explanation", "Frozen CLIP backbone"],
          ["Forensics Adapter [@fa]", "Light adapter on CLIP", "No named evidence", "LoRA adaptation"],
          ["DANN [@ganin]", "Domain-invariant features", "Can remove useful signal", "Tested as ablation"],
          ["CBM [@cbm]", "Concept bottleneck", "Needs concept labels", "Concepts labelled by recipe"],
          ["Tsigos et al. [@tsigos]", "Quantitative XAI evaluation", "Post-hoc methods only", "Explanation metrics"],
          ["DF-P2E [@dfp2e]", "Plain-language explanations", "Post-hoc, large models", "Plain-language app"],
          ["Distillation [@hinton]", "Teacher-student transfer", "Decision only", "Explanation-preserving KD"],
          ["DeepfakeBench [@deepfakebench]", "Common protocol", "Benchmark, not a method", "Single protocol, parents"]],
         [3.4, 4.6, 4.4, 4.6]),
        ("h2", "Research Gaps Addressed"),
        ("bul", [
            "**Generalisation under a verified protocol.** Many reported results mix training and test data or select "
            "models on test data. We enforce separation with automated tests and select only on FF++ validation.",
            "**Explanations that are part of the decision.** Post-hoc heatmaps are not guaranteed to be faithful, and "
            "concept models need costly labels. We obtain concept labels from the forgery synthesis itself.",
            "**Robust and light deployment.** Strong detectors are large and sensitive to compression. We distil both "
            "the decision and the explanation into a small student trained on degraded frames.",
            "**Honest comparison.** Each design element is tested against its parent model, with several seeds where it "
            "matters, and negative results are reported.",
        ]),
    ]
    return out
