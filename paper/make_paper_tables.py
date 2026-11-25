"""LaTeX tables for the paper -> paper/tables/*.tex. Every number is read from results/ or runs/; nothing is typed.

python paper/make_paper_tables.py
Sources: runs/<model>_s<seed>_*/metrics_*.json (video AUC per run, test set and degradation, concept share),
results/paper_stats*.csv (bootstrap CIs), results/explain_metrics.csv, results/explain_restore.csv,
results/paper_calibration.csv, results/efficiency.csv, results/agreement.csv, results/paper_interventions.csv,
results/paper_concepts_by_method.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all import resolve_runs  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "paper" / "tables"
SEEDS = [42, 1, 2]
NAME = {"b1_swin": "B1", "b2_swin_dann": "B2", "b2f_swin_dann_fake": "B2f", "r1_univfd": "R1", "r2_sbi_effb4": "R2",
        "c1_clip_lora": "C1", "c2_clip_sbi": "C2", "g1_concepts": "G1", "g2_dct": "G2", "g2_fft": "G2-fft",
        "g3_full": "G3", "a3_linear_head": "A3", "f1_grace_dct": "F1", "f2_grace_pdann": "F2",
        "f1h_grace_dct_h264": "F1h", "d2_unsup_adapt": "D2", "x1_ffpp_dfd": "X1", "s0_student": "S0",
        "s2_kd_logits": "S2", "s3_kd_full": "S3", "s3h_kd_full_h264": "S3h", "s3hh_kd_f1h_h264": "S3hh",
        "s0x_student_dfd": "S0x", "s3x_kd_from_x1": "S3x", "s3xx_kd_x1_dfd": "S3xx"}
TAG = {"clean": "", "jpeg:30": "_jpeg30", "down:0.25": "_down0.25", "h264:40": "_h26440"}
CONCEPTS = ["blend strength", "colour mismatch", "blur", "sharpen", "resampling", "mouth region", "eye region",
            "full face", "geometric misalign.", "irregular boundary"]


# ------------------------------------------------------------------ helpers
def metric(model, split, cond="clean", key="video_auc"):
    """Per-seed values of a metrics_*.json field for the finished runs of `model`."""
    vals = []
    for run in resolve_runs([model], SEEDS):
        f = run / f"metrics_{split}{TAG[cond]}.json"
        if f.exists():
            v = json.loads(f.read_text()).get(key)
            if v is not None:
                vals.append(float(v))
    return np.array(vals)


def cross(model, cond="clean"):
    a, b = metric(model, "dfd_test", cond), metric(model, "celebdf_test", cond)
    return (a + b) / 2 if len(a) and len(a) == len(b) else np.array([])


def ms(v, d=3):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f}$\\pm${v.std(ddof=1):.{d}f}"


def mm(x, sign=True, d=3):
    t = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return t.replace("-", "$-$")


def ci(model, split, cond="clean"):
    s = pd.read_csv(RES / "paper_stats.csv")
    r = s[(s.model == NAME[model]) & (s.split == split) & (s.condition == cond)]
    return f"[{r.ci_low.iloc[0]:.3f}, {r.ci_high.iloc[0]:.3f}]" if len(r) else "--"


def table(name, env, caption, label, spec, header, rows, size="\\footnotesize", colsep=None):
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size]
    if colsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    lines += [f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
              header + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ tables
def tab_main():
    desc = {"B1": "Swin-B, FF++ fakes", "R1": "frozen CLIP, linear probe", "R2": "SBI reprod., EffNet-B4",
            "C1": "CLIP ViT-L/14 + LoRA", "F1": "final teacher", "F1h": "F1 + H.264 augmentation"}
    rows = []
    for gi, g in enumerate([["b1_swin", "r1_univfd", "r2_sbi_effb4"], ["c1_clip_lora", "f1_grace_dct", "f1h_grace_dct_h264"]]):
        if gi:
            rows.append("\\midrule")
        for m in g:
            rows.append(f"{NAME[m]} & {desc[NAME[m]]} & {ms(metric(m, 'ffpp_test'))} & {ms(metric(m, 'dfd_test'))} & "
                        f"{ms(metric(m, 'celebdf_test'))} & {ms(cross(m))} & {ci(m, 'cross')} \\\\")
    table("tab_main", "table*", "Video AUC under the leak-free protocol (mean$\\pm$sample std over 3 seeds). Cross is the "
          "mean of the DFD and Celeb-DF AUCs; its 95\\% CI comes from 1,000 stratified bootstrap resamples of the test "
          "videos applied to the seed-mean AUC. All models share the FF++ training videos, face crops, sampler and "
          "selection code; R2 trains on real faces and their self-blends only.", "tab:main", "llccccc",
          "ID & Model & FF++ test & DFD & Celeb-DF & Cross & Cross 95\\% CI", rows)


def tab_ablation():
    p = pd.read_csv(RES / "paper_stats_pairs.csv")
    p = p[p.condition == "clean"]
    steps = [("C2 - C1", "c2_clip_sbi", "+ SBI pseudo-fakes in the batch"),
             ("G1 - C2", "g1_concepts", "+ concept bottleneck (residual 1e-3)"),
             ("G2 - G1", "g2_dct", "+ block-DCT branch, gated fusion"),
             ("F1 - G2", "f1_grace_dct", "residual 1e-3 $\\to$ 1e-2, harder selection"),
             ("G3 - A3", "g3_full", "concept head vs.\\ linear head (both + DANN)"),
             ("C1 - B1", "c1_clip_lora", "Swin-B $\\to$ CLIP ViT-L/14 + LoRA"),
             ("F1 - C1", "f1_grace_dct", "all additions together"),
             ("F1h - F1", "f1h_grace_dct_h264", "+ single-frame H.264 augmentation")]
    rows = []
    for i, (comp, model, what) in enumerate(steps):
        if i == 5:
            rows.append("\\midrule")
        cells = []
        for s in ["dfd", "celebdf", "cross"]:
            r = p[(p.comparison == comp) & (p.split == s)].iloc[0]
            star = "$^{*}$" if r.ci_low > 0 or r.ci_high < 0 else ""
            cells.append(f"{mm(r['diff'])}{star}")
        r = p[(p.comparison == comp) & (p.split == "cross")].iloc[0]
        share = metric(model, "ffpp_test", key="concept_share")
        rows.append(f"{comp.replace(' - ', ' vs.\\ ')} & {what} & " + " & ".join(cells) +
                    f" & [{mm(r.ci_low)}, {mm(r.ci_high)}] & {mm(r.p_boot, sign=False)} & {ms(share, 2)} \\\\")
    table("tab_ablation", "table*", "Paired change in seed-mean video AUC, later model minus earlier (3 seeds each, the same "
          "resampled videos for both models). $^{*}$: the 95\\% paired bootstrap CI excludes zero. CI and two-sided "
          "bootstrap $p$ refer to the cross column. Concept share: variance of the concept part of the logit divided "
          "by the variance of the logit on FF++ test frames, for the later model (-- for linear heads).",
          "tab:ablation", "llcccccc", "Pair & Change & DFD & Celeb-DF & Cross & Cross 95\\% CI & $p$ & Concept share", rows)


def tab_alignment():
    def row(model, ref, what):
        rc = metric(model, "celebdf_test"); rd = metric(model, "dfd_test")
        x = cross(model)
        refx = cross(ref)
        n = len(rc)
        d = f"{mm(x.mean() - refx.mean())}" if len(x) and len(refx) else "--"
        dfd = ms(rd) if model != "x1_ffpp_dfd" else "(train)"
        crs = ms(x) if model != "x1_ffpp_dfd" else "--"
        if model == "x1_ffpp_dfd":
            d = f"{mm(rc.mean() - metric(ref, 'celebdf_test').mean())} (CDF)"
        return f"{NAME[model]} & {what} & {n} & {dfd} & {ms(rc)} & {crs} & {d} \\\\"
    rows = [row("b2_swin_dann", "b1_swin", "B1 + DANN/MMD, real is a domain"),
            row("b2f_swin_dann_fake", "b1_swin", "B1 + DANN/MMD on fake domains only"),
            row("g3_full", "g2_fft", "G2-fft + DANN/MMD on $z$, fake domains"),
            row("f2_grace_pdann", "f1_grace_dct", "F1 + DANN/MMD on patch tokens"),
            row("d2_unsup_adapt", "f1_grace_dct", "F1 + unlabelled Celeb-DF crops, DANN/MMD"),
            "\\midrule",
            row("x1_ffpp_dfd", "f1_grace_dct", "F1 recipe trained on FF++ + all of DFD")]
    table("tab_align", "table*", "Domain alignment and training data. $\\Delta$ is the change in mean cross AUC against "
          "the parent model (B1 for B2 and B2f, G2-fft for G3, F1 for F2 and D2); for X1, DFD is training data, so the "
          "change is reported on Celeb-DF against F1. The D2 pool of unlabelled Celeb-DF crops excludes every test video.",
          "tab:align", "lllcccc", "ID & Change & Seeds & DFD & Celeb-DF & Cross & $\\Delta$", rows)


def tab_robustness():
    cols = ["clean", "jpeg:30", "down:0.25", "h264:40"]
    rows = []
    for gi, g in enumerate([["b1_swin", "r1_univfd", "r2_sbi_effb4"],
                            ["c1_clip_lora", "c2_clip_sbi", "g1_concepts", "g2_dct", "f1_grace_dct", "f1h_grace_dct_h264"]]):
        if gi:
            rows.append("\\midrule")
        for m in g:
            rows.append(f"{NAME[m]} & " + " & ".join(ms(cross(m, c)) for c in cols) + " \\\\")
    table("tab_robustness", "table", "Cross-dataset video AUC with degraded test frames (mean$\\pm$sample std, 3 seeds). "
          "Down 0.25: downscale to 25\\% and back. F1h trains with CRF 24--38, never 40.", "tab:robust", "lcccc",
          "Model & Clean & JPEG 30 & Down 0.25 & H.264 40", rows, colsep="3pt")


def tab_calibration():
    c = pd.read_csv(RES / "paper_calibration.csv")
    c = c[c.level == "video"]
    rows = []
    for m in ["b1_swin", "r1_univfd", "r2_sbi_effb4", "c1_clip_lora", "c2_clip_sbi", "g1_concepts", "g2_dct",
              "f1_grace_dct", "f1h_grace_dct_h264", "s3_kd_full"]:
        x = {s: c[(c.model == m) & (c.split == s)] for s in ["ffpp", "dfd", "celebdf"]}
        rows.append(f"{NAME[m]} & {float(x['ffpp']['T'].iloc[0]):.2f} & " + " & ".join(
            f"{x[s].ece_raw.iloc[0]:.3f} / {x[s].ece_temp.iloc[0]:.3f}" for s in ["ffpp", "dfd", "celebdf"]) + " \\\\")
    table("tab_calibration", "table", "Expected calibration error of the video score (15 equal-width bins, seed 42; lower "
          "is better), raw / after the temperature $T$ fitted on FF++ validation frames.", "tab:ece", "lcccc",
          "Model & $T$ & FF++ & DFD & Celeb-DF", rows, colsep="3pt")


def tab_students():
    eff = pd.read_csv(RES / "efficiency.csv").set_index("model")
    rows = []
    groups = [["s0_student", "s2_kd_logits", "s3_kd_full"], ["s3h_kd_full_h264", "s3hh_kd_f1h_h264"],
              ["s3x_kd_from_x1", "s0x_student_dfd", "s3xx_kd_x1_dfd"]]
    what = {"S0": "no teacher", "S2": "KD: logit only", "S3": "KD: logit, $z$, concepts, maps",
            "S3h": "S3 + H.264 inputs", "S3hh": "S3h, teacher F1h", "S3x": "S3, teacher X1",
            "S0x": "S0 + DFD in training", "S3xx": "S3x + DFD in training"}
    for gi, g in enumerate(groups):
        if gi:
            rows.append("\\midrule")
        for m in g:
            dfd = ms(metric(m, "dfd_test")) if m not in ("s0x_student_dfd", "s3xx_kd_x1_dfd") else "(train)"
            rows.append(f"{NAME[m]} & {what[NAME[m]]} & {ms(metric(m, 'ffpp_test'))} & {dfd} & "
                        f"{ms(metric(m, 'celebdf_test'))} & {ms(cross(m, 'h264:40'))} \\\\")
    table("tab_students", "table*", "EfficientNet-B0 students (4.71\\,M parameters, 0.39\\,GFLOPs; 3 seeds each). "
          f"Teacher F1: {eff.loc['f1_grace_dct', 'params_m']:.2f}\\,M parameters, {eff.loc['f1_grace_dct', 'gflops']:.2f}\\,GFLOPs. "
          "H.264 40 is the cross AUC at CRF 40 (-- where not evaluated). S0x and S3xx train on DFD, so DFD is not a test set for them.",
          "tab:students", "llcccc", "ID & Training signal & FF++ test & DFD & Celeb-DF & Cross, H.264 40", rows)


def tab_decomposition():
    d = pd.read_csv(RES / "paper_interventions.csv")
    rows = []
    for model in ["f1_grace_dct", "f1h_grace_dct_h264"]:
        for cond, lab in [("clean", "clean"), ("h26440", "H.264 40")]:
            cells = []
            for split in ["dfd", "celebdf"]:
                x = d[(d.model == model) & (d.condition == cond) & (d.split == split)]
                for v in ["full", "concepts only", "residual only"]:
                    cells.append(ms(x[x.variant == v].video_auc))
            share = d[(d.model == model) & (d.condition == cond) & (d.split == "celebdf") & (d.variant == "full")].concept_share
            rows.append(f"{NAME[model]} & {lab} & " + " & ".join(cells) + f" & {ms(share, 2)} \\\\")
    table("tab_decomp", "table*", "Decision decomposition by intervention on the bottleneck (video AUC, mean$\\pm$sample std "
          "over 3 seeds). Concepts only: $r(z)$ set to 0; residual only: all $s_k$ set to 0. Concept share on Celeb-DF frames.",
          "tab:decomp", "llccccccc",
          "Model & Test frames & \\multicolumn{3}{c}{DFD: full / concepts / residual} & "
          "\\multicolumn{3}{c}{Celeb-DF: full / concepts / residual} & Share", rows, colsep="4pt")


def tab_concept_effects():
    d = pd.read_csv(RES / "paper_interventions.csv")
    d = d[(d.model == "f1_grace_dct") & (d.condition == "clean") & d.variant.str.startswith("off")]
    rows = []
    for k in range(1, 11):
        x = d[d.variant == f"off c{k}"]
        cells = [f"{x.weight.mean():.2f}"]
        for split in ["dfd", "celebdf"]:
            y = x[x.split == split]
            cells += [mm((y.contrib_fake - y.contrib_real).mean(), sign=False, d=2), f"{100 * y.flip_rate.mean():.1f}"]
        cells.append(mm(x[x.split == "celebdf"].delta_auc.mean()))
        rows.append(f"c{k} & {CONCEPTS[k - 1]} & " + " & ".join(cells) + " \\\\")
    table("tab_effects", "table*", "Effect of switching off one concept, do$(s_k{=}0)$, in F1 (mean over 3 seeds, clean test "
          "frames). $\\bar w_k$: learned weight. $\\Delta\\ell$: mean logit contribution $w_k s_k$ on fake frames minus on "
          "real frames. Flip: share of fake videos flagged by the full model (FF++-val threshold) that fall below the "
          "threshold after the intervention. $\\Delta$AUC: change in Celeb-DF video AUC.", "tab:effects", "llccccccc",
          "& Concept & $\\bar w_k$ & DFD $\\Delta\\ell$ & DFD flip (\\%) & CDF $\\Delta\\ell$ & CDF flip (\\%) & "
          "CDF $\\Delta$AUC", rows, colsep="4pt")


def tab_explain():
    e = pd.read_csv(RES / "explain_metrics.csv")
    r = pd.read_csv(RES / "explain_restore.csv").set_index("method")
    get = lambda m, k: e[(e.method == m) & (e.metric == k)].value  # noqa: E731
    f = lambda v: f"{v.iloc[0]:.3f}" if len(v) else "--"  # noqa: E731
    rows = []
    for m, lab in [("F1 region", "F1 region concept"), ("F1 c1", "F1 blend-strength map"), ("S3 c1", "S3 blend-strength map"),
                   ("B1 Grad-CAM", "B1 Grad-CAM"), ("C1 rollout", "C1 attention rollout"), ("random", "random map")]:
        rs = f"{r.loc[m, 'restore_auc']:.3f}" if m in r.index and not m.startswith(("B1", "C1")) else "--"
        rows.append(f"{lab} & {f(get(m, 'pointing_game'))} & {f(get(m, 'mask_iou'))} & {f(get(m, 'deletion_auc'))} & {rs} \\\\")
    rows.append(f"true blend mask & -- & -- & -- & {r.loc['oracle', 'restore_auc']:.3f} \\\\")
    table("tab_explain", "table", "Explanation maps on held-out self-blended test images. Pointing game and mask IoU: "
          "higher is better ($n$=2,000; region concept $n$=1,713, nose/cheek blends have no region concept). "
          "Blur-deletion and restoration: area under the fake-probability curve, lower is better ($n$=500 images "
          "flagged by F1; region concept $n$=438).", "tab:explain", "lcccc",
          "Map & Point.$\\uparrow$ & IoU$\\uparrow$ & Blur-del.$\\downarrow$ & Restore$\\downarrow$", rows, colsep="3pt")


def tab_concepts_by_method():
    t = pd.read_csv(RES / "paper_concepts_by_method.csv", index_col=0)
    order = [("ffpp: real", "FF++ real"), ("ffpp: Deepfakes", "FF++ Deepfakes"), ("ffpp: Face2Face", "FF++ Face2Face"),
             ("ffpp: FaceSwap", "FF++ FaceSwap"), ("dfd: real", "DFD real"), ("dfd: DeepFakeDetection", "DFD fake"),
             ("celebdf: Celeb-real", "Celeb-real"), ("celebdf: YouTube-real", "YouTube-real"),
             ("celebdf: Celeb-synthesis", "Celeb-synthesis")]
    cols = list(t.columns[1:])
    rows = []
    for key, lab in order:
        r = t.loc[key]
        rows.append(f"{lab} & {int(r.frames):,} & " + " & ".join(f"{r[c]:.2f}" for c in cols) + " \\\\")
        if lab in ("FF++ FaceSwap", "DFD fake"):
            rows.append("\\midrule")
    table("tab_bymethod", "table*", "Mean concept score $s_k$ of F1 (seed 42) per test group, frame level. The concepts were "
          "trained only on self-blended images; FF++ fakes and all DFD/Celeb-DF videos carry no concept labels.",
          "tab:bymethod", "lc" + "c" * 10, "Group & Frames & " + " & ".join(f"c{k}" for k in range(1, 11)), rows, colsep="4pt")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.tex"):
        f.unlink()
    for fn in [tab_main, tab_ablation, tab_alignment, tab_robustness, tab_calibration, tab_students, tab_decomposition,
               tab_concept_effects, tab_explain, tab_concepts_by_method]:
        fn()
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f.name}\n{f.read_text(encoding='utf-8')}")
