"""Chapter 10 (Experimentation Results and Discussion) and Chapter 11 (Conclusion and Future Work)."""
from pathlib import Path

import pandas as pd

from data import R, REPO, RES, cdf_note

FIG = Path(__file__).resolve().parent / "figs"


def _clean(rows):
    return [[c.replace("nan", "–") if c.strip() == "nan" else c for c in r] for r in rows]


def _x1_robustness():
    extra = RES / "robustness_extra.csv"  # official Celeb-DF list (written by eval_extras_official)
    x1 = pd.read_csv(extra) if extra.exists() else pd.read_csv(RES / "robustness_x1.csv")
    x1 = x1[x1["model"].str.startswith("x1_ffpp_dfd")]
    rob = pd.read_csv(RES / "robustness.csv")
    f1 = rob[rob["model"].str.startswith("f1_grace_dct")]
    rows = []
    for d, name in [("clean", "Clean"), ("jpeg:30", "JPEG quality 30"), ("down:0.25", "Downscale 0.25"),
                    ("h264:28", "H.264 CRF 28"), ("h264:34", "H.264 CRF 34"), ("h264:40", "H.264 CRF 40")]:
        a = x1[(x1.split == "celebdf_test") & (x1.degrade == d)].video_auc.mean()
        b = f1[(f1.split == "celebdf_test") & (f1.degrade == d)].video_auc.mean()
        rows.append([name, f"{b:.3f}", f"{a:.3f}", f"{a - b:+.3f}"])
    return rows


def _extra_rows():
    spec = [("f1_grace_dct", "F1 teacher (FF++)", False), ("x1_ffpp_dfd", "X1 teacher (FF++ + DFD)", True),
            ("d2_unsup_adapt", "D2 = F1 + unlabelled Celeb-DF adaptation", False),
            ("s0_student", "S0 student, no teacher (FF++ data)", False), ("s3_kd_full", "S3 student, F1 teacher", False),
            ("s3x_kd_from_x1", "S3x student, X1 teacher (FF++ data)", False),
            ("s0x_student_dfd", "S0x student, no teacher, FF++ + DFD data", True),
            ("s3xx_kd_x1_dfd", "S3xx student, X1 teacher, FF++ + DFD data", True)]
    rows = []
    for cfg, name, dfd_in_train in spec:
        n = R.nseeds(cfg)
        if not n:
            continue
        rows.append([name, R.fmt(cfg, "ffpp_test_video_auc"),
                     "in training" if dfd_in_train else R.fmt(cfg, "dfd_test_video_auc"),
                     R.fmt(cfg, "celebdf_test_video_auc"), str(n)])
    return rows


def _concept_auc_rows():
    tab = R.table("5. Explanations (2)")
    d = {}
    for m, metric, v in tab[1:]:
        if metric.startswith("concept_auc_"):
            d.setdefault(m, {})[metric.split("_")[-1]] = v
    names = ["Blend strength", "Colour mismatch", "Blur in a region", "Over-sharpened region", "Resampling traces",
             "Mouth region edited", "Eye region edited", "Whole face swapped", "Misaligned geometry", "Irregular boundary"]
    return [[f"c{k}", names[k - 1]] + [d.get(m, {}).get(f"c{k}", "–") for m in ("F1", "S0", "S3")] for k in range(1, 11)]


def _metric(model, metric):
    for m, met, v in R.table("5. Explanations (2)")[1:]:
        if m == model and met == metric:
            return float(v)
    return float("nan")


def ch10():
    v = R.val
    b1x, c1x, c2x, g1x, g2x, gfx = (v(m, "Cross-dataset") for m in ("B1", "C1", "C2", "G1", "G2", "G2-fft"))
    g3x, a3x, f1x, f2x = (v(m, "Cross-dataset") for m in ("G3", "A3", "F1", "F2"))
    s0x, s2x, s3x = (v(m, "Cross-dataset") for m in ("S0", "S2", "S3"))
    rob = lambda m, c: v(m, c, "3. Robust")  # noqa: E731
    expl = {r[0]: r[1:] for r in R.table("5. Explanations")[1:]}
    pg = {k: float(e[3]) for k, e in expl.items()}
    eff = R.eff.set_index("model")
    x1c, f1c = R.seed_mean("x1_ffpp_dfd", "celebdf_test_video_auc")[0], R.seed_mean("f1_grace_dct", "celebdf_test_video_auc")[0]
    d2c = R.seed_mean("d2_unsup_adapt", "celebdf_test_video_auc")[0]
    s0c, s3c = R.seed_mean("s0_student", "celebdf_test_video_auc")[0], R.seed_mean("s3_kd_full", "celebdf_test_video_auc")[0]
    s3xc = R.seed_mean("s3x_kd_from_x1", "celebdf_test_video_auc")[0]
    s0xc = R.seed_mean("s0x_student_dfd", "celebdf_test_video_auc")[0]
    main = _clean(R.table("1. Main"))
    frame = _clean(R.table("Frame AUC"))
    abl = R.table("2. Ablation")
    robt = _clean(R.table("3. Robust"))
    stud = _clean(R.table("4. Students"))
    leak = R.table("6. Leakage")
    app = R.app[list(R.app)[0]]
    return [
        ("chapter", "Experimentation Results and Discussion"),
        ("h2", "Experimental Setup"),
        ("p", "All models were trained on FF++ (c23) under the protocol of Section 8.2 and evaluated on the FF++ test "
              "split (in-domain) and on DFD and Celeb-DF v2 (cross-dataset). The headline metric is video-level AUC; "
              "'cross-dataset' is the mean of the DFD and Celeb-DF AUCs. Where a model was trained with three seeds, "
              "tables show the mean ± standard deviation over seeds. " + cdf_note()),
        ("h2", "Main Results"),
        ("p", "Table {tab:main} gives the video AUC of every model. All models are close to perfect in-domain (FF++ "
              "test), so the differences that matter are in the cross-dataset columns. Table {tab:frame} gives the "
              "corresponding frame-level AUCs, which are lower because single frames are noisier than video averages."),
        ("tab", "main", "Video AUC of all models (mean ± std over seeds where available).", main[0], main[1:],
         [6.8, 2.5, 2.5, 2.6, 2.6]),
        ("p", f"Replacing the fully fine-tuned Swin-B (B1, cross-dataset {b1x:.3f}) with a frozen CLIP backbone adapted with "
              f"LoRA (C1, {c1x:.3f}) gives the largest single improvement. Adding the concept bottleneck (G1, {g1x:.3f}) "
              f"and the DCT branch with gated fusion (G2, {g2x:.3f}) improves it further. G2 is the best single model "
              f"on this metric. The final teacher F1 ({f1x:.3f}) trades part of that gain for a decision that is about "
              f"80% explained by concepts (Section 10.3), and it remains well ahead of the baseline. Patch-level "
              f"domain alignment (F2, {f2x:.3f}) failed: the domain loss removed the patch information the concepts "
              f"depend on, and the concept share fell to zero."),
        ("tab", "frame", "Frame-level AUC of all models.", frame[0], frame[1:], [6.8, 2.5, 2.5, 2.6, 2.6]),
        ("fig", "roc", RES / "fig_roc_celebdf_test.png", "Video-level ROC curves on Celeb-DF v2 (seed 42 runs).", 10.5),
        ("h2", "Ablation: Effect of Each Component"),
        ("p", "Table {tab:ablation} isolates each design step; every row changes one component of the row above (G2-fft "
              "is compared with G1, A3 with G3, F1 with G2 and F2 with F1)."),
        ("tab", "ablation", "Ablation of the teacher (cross-dataset video AUC).", abl[0], abl[1:], [7.6, 2.8, 3.0, 3.6]),
        ("bul", [
            f"**Self-blended images** (C1 → C2) lowered the cross-dataset AUC slightly ({c1x:.3f} → {c2x:.3f}). DFD "
            f"improved but Celeb-DF dropped, so SBI on its own did not help this backbone. SBI is still needed, because "
            f"it provides the concept labels.",
            f"**The concept bottleneck** (C2 → G1) raised it ({c2x:.3f} → {g1x:.3f}). Interpretability did not cost "
            f"accuracy; it acted as a useful structural prior.",
            f"**The DCT branch** (G1 → G2) helped ({g1x:.3f} → {g2x:.3f}), while the global **FFT branch** hurt "
            f"({g1x:.3f} → {gfx:.3f}). A transform aligned with the codec is the better choice (H3).",
            f"**DANN on z** (G2/G2-fft → G3) did not help ({g3x:.3f}), and removing the concepts from G3 (A3, "
            f"{a3x:.3f}) lowered the AUC, again showing that the bottleneck is not a cost.",
            f"**Residual penalty 10⁻²** (G2 → F1): G1 and G2 explained only about a quarter of their decisions through "
            f"concepts (concept share 0.25–0.29), because the residual path carried most of the evidence. Increasing "
            f"the penalty raised the share to about 0.8 for a cross-dataset AUC of {f1x:.3f}.",
        ]),
        ("h2", "Robustness to Compression and Resizing"),
        ("p", "Table {tab:robust} and Figure {fig:robfig} show cross-dataset AUC under degradations."),
        ("tab", "robust", "Cross-dataset video AUC under degradations.", robt[0], robt[1:], [6.6, 2.6, 2.6, 2.6, 2.6]),
        ("fig", "robfig", RES / "fig_robustness_celebdf_test.png", "Celeb-DF video AUC against JPEG quality, downscaling factor and H.264 CRF.", 16.5),
        ("p", f"The CLIP-based teacher with the DCT branch is much more robust than the Swin-B baseline. At JPEG quality "
              f"30, F1 keeps {rob('F1', 'jpeg:30'):.3f}, against {rob('B1', 'jpeg:30'):.3f} for B1 and "
              f"{rob('C1', 'jpeg:30'):.3f} for C1 without the frequency branch. Under downscaling to a quarter the "
              f"picture is the same. Strong H.264 re-encoding (CRF 40) remains hard for every model "
              f"({rob('F1', 'h264:40'):.3f} for F1, {rob('B1', 'h264:40'):.3f} for B1), because it removes most "
              f"fine detail and adds blocking artefacts of its own. This is the main robustness limitation."),
        ("h2", "Final Model Selection and Seed Variability"),
        ("p", f"The selection rule of Section 8.8 chose F1 over F2 on validation data alone (score 0.961 against 0.817). "
              f"Over three seeds, F1 reached {R.auc('F1', 'FF++ test')} on FF++, {R.auc('F1', 'DFD')} on DFD and "
              f"{R.auc('F1', 'Celeb-DF')} on Celeb-DF. B1, also with three seeds, reached {R.auc('B1', 'DFD')} and "
              f"{R.auc('B1', 'Celeb-DF')}. F1 is clearly ahead on DFD. On Celeb-DF the two are within about one standard "
              f"deviation, which shows why several seeds are needed before drawing conclusions about Celeb-DF."),
        ("h2", "Students and Efficiency"),
        ("tab", "students", "Efficiency and cross-dataset accuracy of the teacher, the baseline and the students.", stud[0],
         stud[1:], [3.4, 1.4, 1.4, 1.6, 1.6, 2.0, 2.0, 1.8, 1.8]),
        ("p", f"The student has 65 times fewer parameters and 200 times fewer operations than the teacher. It processes "
              f"{eff.loc['s3_kd_full', 'gpu_fps_bs64']:.0f} faces per second on the GPU and takes "
              f"{eff.loc['s3_kd_full', 'cpu_onnx_latency_ms_bs1']:.1f} ms per face on the CPU, which meets requirement NFR-1."),
        ("p", f"The hypothesis that distillation improves accuracy (H4) was **not confirmed** for clean inputs. With "
              f"three seeds each, the student trained without a teacher (S0, {s0x:.3f}) is slightly ahead of the "
              f"logit-distilled (S2, {s2x:.3f}) and fully distilled (S3, {s3x:.3f}) students, within seed variation. An "
              f"earlier single-seed comparison had suggested the opposite and was corrected. Under JPEG quality 30 the "
              f"distilled students are slightly better than S0 ({rob('S3', 'jpeg:30'):.3f} and "
              f"{rob('S2', 'jpeg:30'):.3f} against {rob('S0', 'jpeg:30'):.3f}). Distillation also reduced the variation "
              f"between seeds. What distillation clearly transfers is the explanation. S3's concept scores correlate "
              f"more strongly with the teacher's than S0's (Spearman 0.77 against 0.68), its concepts carry about half "
              f"of its decision, and its maps point at the manipulated region more often than the teacher's own c1 map "
              f"(Section 10.7)."),
        ("h2", "Explanation Quality"),
        ("tab", "explain", "Explanation metrics on 2,000 held-out self-blended faces (pointing game and mask IoU: higher is better; deletion: lower is better).",
         ["Method", "Deletion AUC", "Insertion AUC", "Mask IoU", "Pointing game"],
         [[k] + ["–" if x == "nan" else x for x in e] for k, e in expl.items()], [4.4, 3.0, 3.0, 3.0, 3.6]),
        ("p", f"The concept maps of the GRACE teacher localise the manipulated region far better than post-hoc methods. "
              f"Its region-concept map points inside the region {pg['F1 region']:.0%} of the time, and its blend-strength "
              f"map {pg['F1 c1']:.0%}. Grad-CAM on Swin-B scores {pg['B1 Grad-CAM']:.0%}, attention rollout on CLIP "
              f"{pg['C1 rollout']:.0%}, and a random map {pg['random']:.0%}. The student's c1 map ({pg['S3 c1']:.0%}) "
              f"also beats the post-hoc methods. Switching off the top concept lowers the teacher's fake probability by "
              f"{_metric('F1', 'intervention_drop_top'):.3f} on average, against "
              f"{_metric('F1', 'intervention_drop_random'):.3f} for a random concept, so the named concepts do drive "
              f"the decision. The deletion test is a weakness: removing the marked pixels does not lower the score "
              f"more than removing random pixels. This suggests that the teacher also uses evidence spread over the "
              f"whole face, and it is reported as a limitation."),
        ("tab", "conceptauc", "Concept AUC against the recipe labels on held-out self-blended faces.",
         ["ID", "Concept", "F1 (teacher)", "S0", "S3"], _concept_auc_rows(), [1.4, 5.6, 3.4, 3.2, 3.4]),
        ("p", "Region concepts (mouth, eyes, whole face) and blend strength are recognised almost perfectly (AUC above "
              "0.95). Colour, blur, sharpening, resampling and geometry are recognised well (0.76–0.88). The irregular-"
              "boundary concept stays at chance level (about 0.51) for every model, so the elastic warp of the mask "
              "leaves no learnable trace at this resolution. This concept should be removed or redesigned."),
        ("fig", "explfig", FIG / "fig_explanations_top.png", "Explanation examples. Columns: input, final-teacher top-concept map, concept scores, Swin-B Grad-CAM, CLIP attention rollout, student map on a JPEG-30 input.", 16.5),
        ("h2", "Feature-Space Visualisation"),
        ("p", "Figure {fig:tsne} shows t-SNE [@tsne] projections of the features of test faces from the three datasets "
              "for four models. In the Swin-B baseline the datasets occupy largely separate regions: its features "
              "encode which dataset a face comes from, a shortcut that does not transfer. In the CLIP-based models (G2 "
              "and F1) faces from the three datasets are much more intermixed, consistent with their better "
              "cross-dataset accuracy. The features of G3, trained with domain-adversarial alignment, break up into "
              "many thin strands, a visible side effect of the adversarial loss that matches its lack of benefit in "
              "Table {tab:ablation}."),
        ("fig", "tsne", RES / "fig_tsne.png", "t-SNE of test-face features for B1, G2, G3 and F1 (colour: dataset; dot: real, cross: fake).", 16.5),
        ("h2", "Extra Experiments: Data, Adaptation or Teacher?"),
        ("p", "With GPU time left after the planned experiments, three further questions were tested (Table {tab:extra}). "
              "Could unlabelled target-domain data help (D2)? Could a second labelled source help (X1)? And does a "
              "better teacher or more data help the student (S3x, S0x, S3xx)?"),
        ("tab", "extra", "Extra experiments (video AUC; mean ± std over seeds).",
         ["Model", "FF++ test", "DFD", "Celeb-DF", "Seeds"], _extra_rows(), [7.0, 2.6, 2.6, 2.8, 2.0]),
        ("bul", [
            f"**More diverse labelled data helps most.** Training the F1 recipe on FF++ and DFD (X1) raised Celeb-DF "
            f"AUC from {f1c:.3f} to {x1c:.3f}, the largest improvement in the study. Table {{tab:x1rob}} shows that "
            f"this gain holds under every degradation.",
            f"**Unsupervised adaptation to the target did not help.** D2 adapted to unlabelled Celeb-DF videos that "
            f"were not in the test set and reached {d2c:.3f} on Celeb-DF, below F1.",
            f"**A better teacher alone does not make a better student.** A student taught by X1 but trained on FF++ "
            f"data only (S3x, {s3xc:.3f}) is no better than one taught by F1 (S3, {s3c:.3f}). Giving the student "
            f"the DFD data itself (S0x, {s0xc:.3f}) does help, compared with S0 ({s0c:.3f}).",
        ]),
        ("tab", "x1rob", "Celeb-DF video AUC of F1 and X1 under degradations (three seeds each).",
         ["Condition", "F1 (FF++)", "X1 (FF++ + DFD)", "Difference"], _x1_robustness(), [5.0, 4.0, 4.0, 4.0]),
        ("h2", "Effect of the Leakage Fix"),
        ("p", "Table {tab:leak} re-evaluates a Phase-2 checkpoint on the new splits. That checkpoint had been trained "
              "on a random video split of all FF++ folders, including the DFD videos, so its FF++ and DFD columns are "
              "still contaminated; only its Celeb-DF column is a fair comparison. The leaky validation figures of Phase "
              "2 (97–99%) cannot be compared with any leak-free number, and the Phase-2 claim of 98.98% accuracy should "
              "be read in this light."),
        ("tab", "leak", "A Phase-2 checkpoint: its reported (leaky) validation accuracy and its scores on the leak-free splits.",
         leak[0], leak[1:], [2.0, 3.0, 3.0, 2.8, 2.6, 3.6]),
        ("h2", "Application Evaluation"),
        ("p", "The application was checked on six held-out files: three fakes and two reals assembled from test-split "
              "face crops of FF++, DFD and Celeb-DF, and one image without a face (Table {tab:appcheck}). The Careful "
              "check gave the correct verdict on all five videos in 4–5 seconds each. The Quick check found all three "
              "fakes but called one real FF++ video 'inconclusive' and one real Celeb-DF video 'likely fake'. The "
              "student scores that Celeb-DF video as fake on the dataset's own crops as well, so this is a weakness of "
              "the student, not of the application. For this reason the Careful check is the default. The image "
              "without a face was rejected with a clear message. Figure {fig:apptech} shows the technical panel."),
        ("tab", "appcheck", "Application check on held-out files.", app[0][:5] + [app[0][-1]],
         [r[:5] + [r[-1]] for r in app[1:]], [6.2, 1.6, 2.6, 2.2, 1.6, 2.0]),
        ("fig", "apptech", FIG / "app_tech.png", "Technical-details panel: probability, timeline, concept maps, contributions, switch-off test and nearest training faces.", 14.5),
        ("h2", "Discussion"),
        ("tab", "hyp", "Outcome of the hypotheses of Section 2.3.",
         ["Hypothesis", "Evidence", "Outcome"],
         [["H1: CLIP + LoRA narrows the gap", f"Cross-dataset B1 {b1x:.3f} → C1 {c1x:.3f}", "Confirmed"],
          ["H2: Concepts cost little", f"C2 {c2x:.3f} → G1 {g1x:.3f}; F1 concept share ≈ 0.8", "Confirmed"],
          ["H3: DCT better than FFT", f"G2 {g2x:.3f} vs G2-fft {gfx:.3f}; F1 robust at JPEG 30", "Confirmed"],
          ["H4: Distillation helps the student", f"Clean: S0 {s0x:.3f}, S3 {s3x:.3f}; explanation agreement S3 > S0", "Partly: explanations yes, accuracy no"]],
         [5.2, 7.8, 4.0]),
        ("p", "Four lessons stand out. First, the evaluation protocol mattered more than any model change: the Phase-2 "
              "numbers were inflated by leakage, and only a strict protocol makes the comparisons above meaningful. "
              "Second, the strongest generalisation came from the backbone (CLIP with LoRA) and from data diversity "
              "(X1), not from adversarial domain alignment, which was neutral or harmful in all five variants tried. "
              "Third, a concept bottleneck can be made both accurate and genuinely explanatory, provided the residual "
              "path is penalised strongly enough. Fourth, the claims that did not survive (distillation improving "
              "accuracy, patch-level DANN, unsupervised adaptation) were found only because every important comparison "
              "was repeated with several seeds."),
        ("h2", "Comparison with Published Methods"),
        ("p", "Published results use different training data and preprocessing (for example the full FF++ with all "
              "manipulation methods), so their numbers are not directly comparable with ours. Table {tab:published} "
              "compares the methods by design."),
        ("tab", "published", "Design comparison with published detectors.",
         ["Method", "Backbone", "Main idea", "Explanation", "Frequency", "Light student"],
         [["Xception [@ffpp]", "Xception", "Spatial CNN", "–", "–", "–"],
          ["Face X-ray [@facexray]", "HRNet", "Blending-boundary map", "Map", "–", "–"],
          ["SBI [@sbi]", "EfficientNet-B4", "Self-blended data", "–", "–", "–"],
          ["UnivFD [@univfd]", "CLIP ViT-L/14", "Frozen CLIP + probe", "–", "–", "–"],
          ["Forensics Adapter [@fa]", "CLIP ViT-B/16", "Adapter on CLIP", "–", "–", "–"],
          ["Effort [@effort]", "CLIP ViT-L/14", "Orthogonal-subspace tuning", "–", "–", "–"],
          ["LNCLIP-DF / GenD [@lnclip]", "CLIP ViT-L/14", "LayerNorm-only tuning", "–", "–", "–"],
          ["Why Fake? [@whyfake]", "Xception", "Post-hoc concept discovery", "Concepts", "–", "–"],
          ["**GRACE (ours)**", "CLIP ViT-L/14", "Recipe-supervised concepts", "Concepts + maps", "DCT", "Yes"]],
         [3.6, 2.8, 4.0, 2.6, 1.8, 2.2]),
    ]


def ch11():
    f1x, b1x = R.val("F1", "Cross-dataset"), R.val("B1", "Cross-dataset")
    x1c = R.seed_mean("x1_ffpp_dfd", "celebdf_test_video_auc")[0]
    return [
        ("chapter", "Conclusion and Future Work"),
        ("h2", "Summary of the Work"),
        ("p", "This phase turned a single-dataset prototype into a generalisable, robust and explainable deepfake "
              "detection system with a verified evaluation. The work began with an audit, which showed that the earlier "
              "high accuracy came from train/validation leakage, and with a strict protocol that enforces separation by "
              "automated tests. On this foundation GRACE was built. It combines a frozen CLIP ViT-L/14 adapted with "
              "LoRA, a JPEG-aligned DCT branch with gated fusion, and a concept bottleneck whose ten concepts are labelled "
              "automatically by the recipe of self-blended training forgeries. Its decision is a weighted sum of named "
              "concepts, so the explanation is the computation."),
        ("p", f"Trained only on FF++, the final teacher reached a cross-dataset video AUC of {f1x:.3f} against {b1x:.3f} "
              f"for the Swin-B baseline, kept its accuracy under strong JPEG compression and downscaling, and localised "
              f"manipulated regions far better than Grad-CAM or attention rollout. A 4.7 M-parameter student runs in "
              f"about 4 ms per face on a CPU and inherits the teacher's explanations. A further experiment showed that "
              f"adding a second labelled source (DFD) raises Celeb-DF AUC to {x1c:.3f}, more than any architectural "
              f"change. The system is delivered as a plain-language web application with the full technical evidence "
              f"available on demand, backed by 242 automated tests and browser and API checks."),
        ("h2", "Utility of the Work"),
        ("bul", [
            "**Media verification and journalism:** a quick, explained first opinion on a suspicious clip, with a "
            "report that records the exact file analysed.",
            "**Digital forensics:** concept scores, maps, the switch-off test and nearest training examples give an "
            "analyst evidence to examine, not just a score.",
            "**Education and awareness:** the plain-language clues teach users what face manipulations look like.",
            "**Research:** the leak-free protocol, the experiment queue and the result scripts form a reusable "
            "test bench for new detectors.",
        ]),
        ("h2", "Limitations"),
        ("bul", [
            "The FF++ copy available lacks NeuralTextures and FaceShifter and has few Face2Face and FaceSwap videos. "
            "Only two unseen datasets were tested, and both predate diffusion-based face synthesis.",
            "Strong H.264 re-encoding (CRF 40) still reduces cross-dataset AUC to about 0.64 for every model.",
            "Concept labels come from a synthetic recipe. One concept (irregular boundary) is not learnable, and "
            "deletion tests show that the teacher also uses evidence outside the marked region.",
            "The system is visual and frame-based. Audio and temporal modelling, part of the original multimodal "
            "vision, were not implemented in this phase.",
            "Detection is limited to near-frontal faces; extreme head poses are excluded from the verdict.",
            "Deepfake datasets are demographically skewed, and error rates across groups were not measured.",
        ]),
        ("h2", "Future Work"),
        ("bul", [
            "**Multimodal integration:** add an audio-visual branch, for example lip-speech synchronisation, whose "
            "evidence enters the concept bottleneck as additional named concepts (Section 5.7).",
            "**Temporal modelling:** learn from sequences of frames to capture flicker and identity drift directly.",
            "**Broader and newer data:** the full FF++ (all methods, c23 and c40), DFDC [@dfdc] and DF40 [@df40], "
            "including diffusion-based forgeries; train students with diverse labelled data, which the extra "
            "experiments showed to be the most effective lever.",
            "**Better robustness to video codecs:** train with H.264 re-encoding and codec-aware frequency features.",
            "**Explanations:** replace the unlearnable concept, add concepts for diffusion artefacts, and run a user "
            "study of the plain-language interface.",
            "**Fairness and security:** measure error rates across demographic groups and test robustness to "
            "adversarial perturbations.",
            "**Publication:** complete the IEEE paper draft with the final results on the official Celeb-DF list.",
        ]),
    ]
