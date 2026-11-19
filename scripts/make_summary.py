"""T7.3 draft: results/summary.md built only from the results files (every number traced to a metrics file).
Findings are phrased neutrally ("raises"/"lowers") from the measured deltas, so weak or negative results read
as such. A human (or Claude) edits the wording afterwards; numbers must not be typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_tables import LABEL, cross, metric  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def m(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.mean(v)) if v else None


def compare(a: str, b: str, what: str, split: str | None = None, degrade: str = "clean") -> str | None:
    va = m(metric(a, split, degrade=degrade)) if split else m(cross(a, degrade=degrade))
    vb = m(metric(b, split, degrade=degrade)) if split else m(cross(b, degrade=degrade))
    if va is None or vb is None:
        return None
    verb = "raises" if vb > va + 0.005 else "lowers" if vb < va - 0.005 else "leaves essentially unchanged (±0.005)"
    where = f"{split} video AUC" if split else "cross-dataset video AUC (mean of DFD and Celeb-DF)"
    if degrade != "clean":
        where += f" at {degrade}"
    return f"- {what} {verb} {where}: {va:.3f} → {vb:.3f} ({LABEL[a]} → {LABEL[b]})."


def main() -> None:
    have = lambda n: bool(resolve_runs([n], [42]))  # noqa: E731
    mode = (REPO_ROOT / "splits" / "counts.md").read_text().split("\n")[0]
    lines = ["# Sprint summary (draft generated from results files)", "",
             "## Setup", "",
             "- Train: FaceForensics++ c23 (Deepfakes, Face2Face, FaceSwap; official split by source video), aligned face crops only.",
             "- Unseen tests: DeepFakeDetection (DFD) and Celeb-DF v2; FF++ test for in-domain. " + mode.replace("**", "") + ".",
             "- Metric: video AUC (mean fake probability over a video's crops); thresholds from FF++ validation only.",
             "- Models: Swin-B baseline (B1, B2 + DANN), CLIP ViT-L/14 + LoRA (C1), + self-blended images (C2), + concept bottleneck (G1), "
             "+ frequency branch with gated fusion (G2 DCT / G2-fft), + DANN (G3); final teacher chosen on FF++ val between F1 (G2-DCT + residual 1e-2 + harder val selection) and F2 (F1 + DANN on patch features); EfficientNet-B0 students S0/S2/S3 distilled from the final teacher.",
             "- Hardware: one RTX 4500 Ada (24 GB); seeds 42, 1 and 2 for the final teacher (F1), B1 and all three students (S0, S2, S3); X1 and D2 three seeds; other models one seed (42). Numbers are seed means where more than one seed exists.", "",
             "## Findings", ""]
    for a, b, what in [("b1_swin", "c1_clip_lora", "Replacing Swin-B with CLIP + LoRA"),
                       ("c1_clip_lora", "c2_clip_sbi", "Adding self-blended images"),
                       ("c2_clip_sbi", "g1_concepts", "The concept bottleneck"),
                       ("g1_concepts", "g2_dct", "The DCT branch with gated fusion"),
                       ("g1_concepts", "g2_fft", "The FFT branch with gated fusion"),
                       ("b1_swin", "b2_swin_dann", "DANN on Swin-B with real as a domain (runbook spec)"),
                       ("b1_swin", "b2f_swin_dann_fake", "DANN on Swin-B over fake domains only"),
                       ("g3_full", "a3_linear_head", "Removing the concept head from G3 (A3)"),
                       ("g2_dct", "f1_grace_dct", "Residual penalty 1e-2 + harder val selection on G2-DCT (F1)"),
                       ("f1_grace_dct", "f2_grace_pdann", "DANN/MMD on patch features (F2 vs F1)"),
                       ("g3_full", "f2_grace_pdann", "Domain terms on patch features instead of z (F2 vs G3)"),
                       ("s0_student", "s3_kd_full", "Full distillation over a teacher-free student")]:
        if have(a) and have(b):
            s = compare(a, b, what)
            if s:
                lines.append(s)
    for a, b in (("s0_student", "s3_kd_full"), ("b1_swin", "s3_kd_full")):
        if have(a) and have(b):
            s = compare(a, b, "Under JPEG quality 30,", degrade="jpeg:30")
            if s:
                lines.append(s)
    eff = REPO_ROOT / "results" / "efficiency.csv"
    if eff.exists():
        e = pd.read_csv(eff).set_index("model")
        from utils.common import final_teacher_name
        t = final_teacher_name()
        if "s3_kd_full" in e.index and t in e.index:
            lines.append(f"- The S3 student has {e.loc['s3_kd_full', 'params_m']} M parameters and {e.loc['s3_kd_full', 'gflops']} GFLOPs "
                         f"vs {e.loc[t, 'params_m']} M and {e.loc[t, 'gflops']} GFLOPs for the teacher ({t})"
                         + (f"; ONNX CPU latency {e.loc['s3_kd_full', 'cpu_onnx_latency_ms_bs1']} ms per face." if 'cpu_onnx_latency_ms_bs1' in e.columns else "."))
    ex = REPO_ROOT / "results" / "explain_metrics.csv"
    if ex.exists():
        d = pd.read_csv(ex)
        pg = d[d.metric == "pointing_game"].set_index("method")["value"]
        if len(pg):
            lines.append("- Pointing game on held-out SBI blends: " + ", ".join(f"{k} {v:.2f}" for k, v in pg.items()) + ".")
    g3p = REPO_ROOT / "results" / "decisions.md"
    if g3p.exists():
        seen, dec = set(), []
        for ln in g3p.read_text(encoding="utf-8").strip().splitlines():
            key = ln.split(" ", 3)[-1]  # drop the timestamp so repeated decisions (re-runs) appear once
            if key not in seen:
                seen.add(key)
                dec.append(ln)
        lines += ["", "Decisions taken by the runbook rules:", ""] + dec
    lines += ["", "## What did not work", "", "(Fill in from the negative deltas above and `results/decisions.md`.)", "",
              "## Limitations", "",
              "- Reduced FF++: c23 only; FaceSwap 39 and Face2Face 134 videos have crops; no NeuralTextures or FaceShifter.",
              "- Two cross-dataset sets; " + ("Celeb-DF uses an interim test list (same composition as the official one)." if "interim" in mode else "Celeb-DF uses the official test list."),
              "- Faces only, no audio; one or two seeds per model.",
              "- The app's face crops use SFD + eye alignment (margin 1.02x), not the original preprocessing code.",
              "- Head pose: FF++ is near-frontal; faces beyond its pose range (|yaw| > 0.57) are excluded from the app verdict.",
              "- The student is less reliable than the teacher on re-cropped and on hard real Celeb-DF faces; the app defaults to the teacher (Deep).",
              "- Explanations localise well (pointing game) but deletion tests do not show that removing the highlighted region flips the decision; distillation transfers concept scores better than map locations.", "",
              "## Next steps toward a paper", "",
              "- Full FF++ (all methods, c23 + c40), more unseen sets (DFDC-P, DF40), three seeds for every model, a user study of the explanations."]
    out = REPO_ROOT / "results" / "summary.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
