"""LaTeX tables for the paper -> paper/tables/*.tex. Numbers come only from results/ files (no typing by hand).

python paper/make_paper_tables.py
Sources: results/robustness.csv (per run, per test set, per degradation), results/paper_stats.csv and
paper_stats_pairs.csv (bootstrap CIs), results/explain_metrics.csv, results/explain_restore.csv,
results/paper_calibration.csv, results/efficiency.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "tables"
NAME = {"b1_swin": "B1", "r1_univfd": "R1", "r2_sbi_effb4": "R2", "c1_clip_lora": "C1", "c2_clip_sbi": "C2",
        "g1_concepts": "G1", "g2_dct": "G2", "g3_full": "G3", "a3_linear_head": "A3", "f1_grace_dct": "F1",
        "f1h_grace_dct_h264": "F1h", "s0_student": "S0", "s2_kd_logits": "S2", "s3_kd_full": "S3",
        "s3h_kd_full_h264": "S3h", "x1_ffpp_dfd": "X1", "b2_swin_dann": "B2", "f2_grace_pdann": "F2",
        "d2_unsup_adapt": "D2", "g2_fft": "G2-fft"}
DESC = {"B1": "Swin-B, FF++ fakes", "R1": "frozen CLIP, linear probe", "R2": "SBI reprod., EffNet-B4",
        "C1": "CLIP ViT-L/14 + LoRA", "F1": "GRACE (final teacher)", "F1h": "F1 + H.264 augment.",
        "S0": "EffNet-B0, no teacher", "S3": "EffNet-B0, full KD", "S3h": "S3 + H.264 inputs"}


def rob():
    d = pd.read_csv(RES / "robustness.csv")
    return d[d.model.isin(NAME)]


def ms(v, digits=3):
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f}$\\pm${v.std(ddof=1):.{digits}f}"


def per_seed(d, model, degrade):
    x = d[(d.model == model) & (d.degrade == degrade)].pivot_table(index="seed", columns="split", values="video_auc")
    x["cross"] = (x["dfd_test"] + x["celebdf_test"]) / 2
    return x


def ci(model, split, cond="clean"):
    s = pd.read_csv(RES / "paper_stats.csv")
    r = s[(s.model == NAME[model]) & (s.split == split) & (s.condition == cond)]
    return f"[{r.ci_low.iloc[0]:.3f}, {r.ci_high.iloc[0]:.3f}]" if len(r) else "--"


def mm(x: float, sign: bool = True) -> str:
    """Signed number with a math minus for LaTeX."""
    t = f"{x:+.3f}" if sign else f"{x:.3f}"
    return t.replace("-", "$-$")


def write(name, body):
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")


def main_table(d):
    rows = []
    groups = [["b1_swin", "r1_univfd", "r2_sbi_effb4"], ["c1_clip_lora", "f1_grace_dct", "f1h_grace_dct_h264"],
              ["s0_student", "s3_kd_full", "s3h_kd_full_h264"]]
    for gi, g in enumerate(groups):
        if gi:
            rows.append("\\midrule")
        for m in g:
            x = per_seed(d, m, "clean")
            rows.append(f"{NAME[m]} & {DESC[NAME[m]]} & {ms(x.ffpp_test)} & {ms(x.dfd_test)} & {ms(x.celebdf_test)} & "
                        f"{ms(x.cross)} & {ci(m, 'cross')} \\\\")
    write("tab_main", "\n".join([
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Video AUC under the leak-free protocol (mean$\\pm$sample std over 3 seeds). Cross = mean of DFD and Celeb-DF; "
        "the 95\\% CI is a stratified bootstrap over videos (1,000 resamples) of the seed-mean cross AUC. All models "
        "see the same FF++ training identities; R2 sees only real faces and their self-blends.}",
        "\\label{tab:main}", "\\begin{tabular}{llccccc}", "\\toprule",
        "ID & Model & FF++ test & DFD & Celeb-DF & Cross & Cross 95\\% CI \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))


def ablation():
    p = pd.read_csv(RES / "paper_stats_pairs.csv")
    p = p[p.condition == "clean"]
    steps = [("C2 - C1", "+ SBI pseudo-fakes"), ("G1 - C2", "+ concept bottleneck"), ("G2 - G1", "+ DCT branch, gated fusion"),
             ("F1 - G2", "+ residual penalty, val selection"), ("G3 - A3", "concept head vs.\\ linear head"),
             ("C1 - B1", "Swin-B $\\to$ CLIP+LoRA"), ("F1 - C1", "all GRACE additions"), ("F1h - F1", "+ H.264 augmentation")]
    rows = []
    for comp, what in steps:
        cells = []
        for s in ["dfd", "celebdf", "cross"]:
            r = p[(p.comparison == comp) & (p.split == s)].iloc[0]
            star = "$^{*}$" if r.ci_low > 0 or r.ci_high < 0 else ""
            cells.append(f"{mm(r['diff'])}{star}")
        r = p[(p.comparison == comp) & (p.split == "cross")].iloc[0]
        rows.append(f"{comp.replace(' - ', ' vs.\\ ')} & {what} & " + " & ".join(cells) + f" & [{mm(r.ci_low)}, {mm(r.ci_high)}] \\\\")
    write("tab_ablation", "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\footnotesize",
        "\\caption{Paired change in seed-mean video AUC (later minus earlier model, 3 seeds each). $^{*}$: the 95\\% "
        "paired bootstrap CI excludes zero. CI shown for the cross-dataset column.}",
        "\\label{tab:ablation}", "\\begin{tabular}{llcccc}", "\\toprule",
        "Pair & Change & DFD & Celeb-DF & Cross & Cross 95\\% CI \\\\", "\\midrule", *rows[:5], "\\midrule", *rows[5:],
        "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))


def robustness(d):
    cols = [("clean", "Clean"), ("jpeg:30", "JPEG 30"), ("down:0.25", "Down 0.25"), ("h264:40", "H.264 40")]
    rows = []
    for gi, g in enumerate([["b1_swin", "r1_univfd", "r2_sbi_effb4", "c1_clip_lora"], ["f1_grace_dct", "f1h_grace_dct_h264"],
                            ["s0_student", "s3_kd_full", "s3h_kd_full_h264"]]):
        if gi:
            rows.append("\\midrule")
        for m in g:
            vals = [per_seed(d, m, c).cross for c, _ in cols]
            rows.append(f"{NAME[m]} & " + " & ".join(ms(v) for v in vals) + " \\\\")
    write("tab_robustness", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Cross-dataset video AUC under degradation of the test frames (mean$\\pm$sample std, 3 seeds). "
        "Down 0.25 = downscale to 25\\% and back. H.264 training (F1h, S3h) uses CRF 24--38, never 40.}",
        "\\label{tab:robust}", "\\begin{tabular}{lcccc}", "\\toprule",
        "Model & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


def explanations():
    e = pd.read_csv(RES / "explain_metrics.csv")
    r = pd.read_csv(RES / "explain_restore.csv").set_index("method")
    get = lambda m, k: e[(e.method == m) & (e.metric == k)].value  # noqa: E731
    rows = []
    for m, lab in [("F1 region", "F1 region concept"), ("F1 c1", "F1 blend-strength map"), ("S3 c1", "S3 blend-strength map"),
                   ("B1 Grad-CAM", "B1 Grad-CAM"), ("C1 rollout", "C1 attention rollout"), ("random", "Random map")]:
        pg, iou, dl = get(m, "pointing_game"), get(m, "mask_iou"), get(m, "deletion_auc")
        rs = r.loc[m, "restore_auc"] if m in r.index and not m.startswith(("B1", "C1")) else None
        f = lambda v: f"{v.iloc[0]:.3f}" if len(v) else "--"  # noqa: E731
        rows.append(f"{lab} & {f(pg)} & {f(iou)} & {f(dl)} & " + (f"{rs:.3f}" if rs is not None else "--") + " \\\\")
    rows.append(f"Oracle (true mask) & -- & -- & -- & {r.loc['oracle', 'restore_auc']:.3f} \\\\")
    write("tab_explain", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Explanation quality on held-out self-blended test images with known blend masks. Pointing game and "
        "mask IoU: higher is better ($n$=2,000; region concept $n$=1,713). Blur-deletion and restoration: area under the "
        "fake-probability curve as patches are blurred or restored to the real source pixels, lower is better ($n$=500 "
        "images the teacher flags; region concept $n$=438). B1 and C1 do not flag these images (mean start probability "
        "0.15 and 0.24), so their restoration curves are not comparable and are omitted.}",
        "\\label{tab:explain}", "\\begin{tabular}{lcccc}", "\\toprule",
        "Map & Pointing$\\uparrow$ & IoU$\\uparrow$ & Blur-del.$\\downarrow$ & Restore$\\downarrow$ \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


def calibration():
    c = pd.read_csv(RES / "paper_calibration.csv")
    c = c[c.level == "video"]
    rows = []
    for m in ["b1_swin", "r1_univfd", "r2_sbi_effb4", "c1_clip_lora", "f1_grace_dct", "f1h_grace_dct_h264", "s3_kd_full"]:
        v = {s: c[(c.model == m) & (c.split == s)].ece_raw for s in ["ffpp", "dfd", "celebdf"]}
        rows.append(f"{NAME[m]} & " + " & ".join(f"{v[s].iloc[0]:.3f}" for s in v) + " \\\\")
    write("tab_calibration", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Expected calibration error (15 bins, video level, seed 42; lower is better) of the raw fake "
        "probability. ECE does not measure ranking: R2 is well calibrated but ranks poorly (Table~\\ref{tab:main}).}",
        "\\label{tab:ece}", "\\begin{tabular}{lccc}", "\\toprule", "Model & FF++ & DFD & Celeb-DF \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = rob()
    main_table(d); ablation(); robustness(d); explanations(); calibration()
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f.name}\n{f.read_text(encoding='utf-8')}")
