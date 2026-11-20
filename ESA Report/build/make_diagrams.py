"""Design diagrams for the Phase-3 report (HLD / LLD / system design). Writes PNGs (200 dpi) to build/figs/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
INK, MUTED, LINE = "#16181D", "#545A64", "#9AA0A8"
FILL = {"blue": "#E3ECFA", "orange": "#FCEBDD", "grey": "#F1F2F4", "white": "#FFFFFF", "green": "#E4F2EA"}
EDGE = {"blue": "#1F55B8", "orange": "#B4410E", "grey": "#6B7280", "white": "#6B7280", "green": "#2F7D4F"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), tone="white", size=9, title_size=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=FILL[tone], ec=EDGE[tone], lw=1.2))
    ts = title_size or size + 0.5
    if lines:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=ts, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 2.2 - ts * 0.55 - 1.2, "\n".join(lines), ha="center", va="top", fontsize=size - 0.5,
                color=MUTED, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts, weight="bold", color=INK)
    return (x, y, w, h)


def arrow(ax, p, q, text="", style="-|>", dashed=False, color=INK, rad=0.0, fs=8):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.1, color=color,
                        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.2, text, ha="center", va="bottom", fontsize=fs, color=MUTED)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def context_deployment():
    fig, ax = canvas(16, 9.4)
    ax.text(3, 91, "Online: inference (one workstation)", fontsize=10, weight="bold", color=EDGE["blue"])
    ax.text(3, 43, "Offline: training and evaluation (same workstation, RTX 4500 Ada 24 GB)", fontsize=10, weight="bold",
            color=EDGE["orange"])
    u = box(ax, 3, 66, 22, 18, "User", ["web browser", "uploads a photo or", "short video"], "grey")
    g = box(ax, 36, 60, 34, 28, "GRACE web app (Gradio)", ["app/app.py: plain-language result page",
                                                            "app/pipeline.py: frames, face crop,", "head-pose filter, scoring, report",
                                                            "REST endpoints: /check /clear /rescore", "/overlay_concept /overlay_opacity"], "blue")
    c = box(ax, 82, 72, 36, 16, "Careful check (GPU)", ["final teacher F1, PyTorch", "CLIP ViT-L/14 + LoRA + DCT + concepts"], "blue")
    q = box(ax, 82, 52, 36, 16, "Quick check (CPU)", ["S3 student, ONNX Runtime", "EfficientNet-B0, 4.7 M params"], "blue")
    r = box(ax, 128, 60, 30, 26, "release/ and cache/", ["teacher checkpoint + temperature", "student ONNX + thresholds",
                                                         "teacher outputs index", "(nearest training faces)"], "grey")
    arrow(ax, (u[0] + u[2], 78), (g[0], 78), "HTTP :7860")
    arrow(ax, (g[0], 69), (u[0] + u[2], 69))
    ax.text((u[0] + u[2] + g[0]) / 2, 66.5, "verdict,\nclues, report", ha="center", va="top", fontsize=8, color=MUTED)
    arrow(ax, (g[0] + g[2], 80), (c[0], 80))
    arrow(ax, (g[0] + g[2], 64), (q[0], 60))
    arrow(ax, (r[0], 80), (c[0] + c[2], 80), dashed=True)
    arrow(ax, (r[0], 66), (q[0] + q[2], 60), dashed=True)

    d = box(ax, 3, 14, 24, 22, "Datasets (read-only)", ["FaceForensics++ c23", "DeepFakeDetection (DFD)", "Celeb-DF v2",
                                                         "aligned 224x224 face crops"], "grey")
    s = box(ax, 36, 14, 24, 22, "Leak-free splits", ["official FF++ split", "official Celeb-DF test list", "DFD test only",
                                                      "overlap tests (pytest)"], "orange")
    t = box(ax, 69, 14, 26, 22, "Training", ["train.py (teachers)", "distill.py (students)", "self-blended images",
                                             "queue_runner.py, 24/7"], "orange")
    e = box(ax, 104, 14, 24, 22, "Evaluation", ["evaluate.py, eval_all.py", "3 test sets x 10", "degradations, XAI metrics",
                                                "calibration"], "orange")
    rel = box(ax, 136, 14, 22, 22, "Reporting", ["tables.md, summary.md", "figures", "release package",
                                                 "experiments.csv"], "orange")
    for a_, b_ in [(d, s), (s, t), (t, e), (e, rel)]:
        arrow(ax, (a_[0] + a_[2], 24), (b_[0], 24))
    arrow(ax, (rel[0] + rel[2] / 2, rel[1] + rel[3]), (r[0] + r[2] / 2, r[1]), "make_release.py", dashed=True)
    save(fig, "hld_context_deployment.png")


def layered():
    fig, ax = canvas(16, 10.5)
    layers = [
        ("Presentation layer", "blue", ["Gradio page: upload card, verdict card, clue cards, next steps",
                                        "technical panel (timeline, concept maps, re-score)", "downloadable HTML report"]),
        ("Application layer", "blue", ["app/pipeline.py: sample_frames, FaceCropper, head_pose / pose_ok",
                                       "QuickEngine (ONNX) / DeepEngine (PyTorch), analyse(), band()",
                                       "explain/: concept_maps, contributions, intervene, retrieval, video_cues"]),
        ("Model layer", "orange", ["GraceDetector: ClipLoraBackbone + DCTBranch + GatedFusion + ConceptHead + ConceptDecision",
                                   "StudentDetector (EfficientNet-B0), SwinDetector (baseline), DomainHead (ablations)",
                                   "losses.py, EMA, calibration (temperature scaling)"]),
        ("Data layer", "orange", ["make_splits.py (leak-free CSV splits), FaceFrameDataset, CellBalancedBatchSampler",
                                  "SBIMaker (self-blended images + recipe labels), degrade.py (JPEG / resize / H.264)",
                                  "TeacherCache (cached teacher outputs for distillation)"]),
        ("Infrastructure layer", "grey", ["configs/*.yaml + utils/config.py, queue.yaml + queue_runner.py, run_snapshot.py",
                                          "evaluate.py, eval_all.py, make_tables.py, make_release.py, experiments.csv",
                                          "pytest suite (splits, shapes, SBI, student, degrade, video cues, app pose)"]),
    ]
    y = 86
    for name, tone, lines in layers:
        box(ax, 2, y, 26, 16, name, (), tone, title_size=10)
        box(ax, 32, y, 126, 16, "", (), "white")
        ax.text(36, y + 8, "\n".join(lines), ha="left", va="center", fontsize=9, color=INK, linespacing=1.5)
        y -= 20
    for yy in (86, 66, 46, 26):
        arrow(ax, (15, yy), (15, yy - 3.6), style="<|-|>")
    save(fig, "hld_layers.png")


def class_diagram():
    fig, ax = canvas(16, 11)

    def cls(x, y, w, name, attrs, methods, tone="white"):
        h = 4.5 + 2.4 * (len(attrs) + len(methods)) + 2.5
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="square,pad=0", fc=FILL[tone], ec=EDGE[tone], lw=1.2))
        ax.text(x + w / 2, y - 2.4, name, ha="center", va="center", weight="bold", fontsize=9.5, color=INK)
        ax.plot([x, x + w], [y - 4.6, y - 4.6], color=EDGE[tone], lw=0.8)
        yy = y - 6.6
        for a in attrs:
            ax.text(x + 1.2, yy, a, ha="left", va="center", fontsize=7.8, color=INK)
            yy -= 2.4
        ax.plot([x, x + w], [yy + 1.0, yy + 1.0], color=EDGE[tone], lw=0.8)
        yy -= 0.6
        for m in methods:
            ax.text(x + 1.2, yy, m, ha="left", va="center", fontsize=7.8, color=MUTED)
            yy -= 2.4
        return (x, y - h, w, h)

    # composition tree: teacher on top, its five parts in a row, student and domain head on the bottom row
    g = cls(52, 108, 56, "GraceDetector (nn.Module)", ["backbone: ClipLoraBackbone   freq: DCTBranch | FFTBranch",
                                                      "fusion: GatedFusion   concept_head: ConceptHead",
                                                      "decision: ConceptDecision   domain: DomainHead (optional)"],
            ["forward(x) -> {logit, z, concepts, maps, contributions, residual, cls}"], "orange")
    parts = [("ClipLoraBackbone", ["CLIP ViT-L/14 (frozen)", "LoRA r=16, a=32 on qkv"], ["forward(x) -> cls[1024],", "  patches[1024x16x16]"]),
             ("DCTBranch", ["8x8 orthonormal DCT-II on Y", "CNN encoder -> 256-d"], ["forward(x) -> h_f[256]"]),
             ("GatedFusion", ["pc, pf: Linear -> 512", "gate + LayerNorm"], ["forward(cls, h_f)", "  -> z[512], gate"]),
             ("ConceptHead", ["conv1x1: C -> 10 maps", "score = top-10% mean"], ["forward(patches)", "  -> scores[10], maps"]),
             ("ConceptDecision", ["w[10], b", "residual MLP r(z)"], ["forward(c, z) -> logit,", "  concept_part, residual"])]
    boxes = []
    for i, (n, a, m) in enumerate(parts):
        boxes.append(cls(1 + i * 32, 76, 30, n, a, m))
    for bx in boxes:
        arrow(ax, (g[0] + g[2] / 2, g[1]), (bx[0] + bx[2] / 2, bx[1] + bx[3]), style="-|>")
    st = cls(100, 36, 50, "StudentDetector (nn.Module)", ["backbone: EfficientNet-B0 (timm)", "concept_head, decision, proj(z)"],
             ["forward(x) -> same output dict as teacher"], "blue")
    dh = cls(22, 36, 50, "DomainHead (ablations only)", ["GradientReversal(alpha)", "classifier over D domains"],
             ["forward(z, alpha) -> domain logits", "mmd_loss(z_a, z_b)"], "grey")
    for bx in boxes[3:]:
        arrow(ax, (st[0] + st[2] / 2, st[1] + st[3]), (bx[0] + bx[2] / 2, bx[1]), dashed=True)
    ax.text(st[0] + 2, st[1] + st[3] + 6, "reuses the same heads", fontsize=8, color=MUTED)
    arrow(ax, (dh[0] + dh[2] / 2, dh[1] + dh[3]), (boxes[2][0] + boxes[2][2] / 2 - 6, boxes[2][1]), dashed=True)
    ax.text(dh[0] + dh[2] / 2 + 12, dh[1] + dh[3] + 6, "takes z from GatedFusion", fontsize=8, color=MUTED)
    save(fig, "lld_classes.png")


def sequence():
    fig, ax = canvas(16, 11.5)
    actors = ["User", "Gradio UI", "run_analysis", "FaceCropper", "Engine\n(Deep / Quick)", "Result builder"]
    xs = [13, 40, 67, 94, 121, 148]
    for x, a in zip(xs, actors):
        box(ax, x - 11, 104, 22, 9, a, (), "blue" if a != "User" else "grey", title_size=9)
        ax.plot([x, x], [104, 4], color=LINE, lw=0.9, linestyle=(0, (3, 3)))
    msgs = [
        (0, 1, "upload file, choose check type, click Check"),
        (1, 2, "/check(file, mode)"),
        (2, 2, "validate type and size; sample 32 frames"),
        (2, 3, "crop_with_pose(frame) x 32"),
        (3, 2, "224x224 aligned faces + yaw, pitch"),
        (2, 4, "run(faces)"),
        (4, 2, "prob, concepts, maps, contributions per face"),
        (2, 2, "drop extreme poses; mean prob; band()"),
        (2, 5, "hero_html, reasons_html, build_report"),
        (5, 1, "verdict card, 3 clue cards, report file"),
        (1, 0, "page updates; technical panel on demand"),
    ]
    y = 98
    for a, b, t in msgs:
        if a == b:
            ax.add_patch(FancyArrowPatch((xs[a], y), (xs[a], y - 4), arrowstyle="-|>", mutation_scale=10,
                                         connectionstyle="arc3,rad=-1.6", color=INK, lw=1))
            ax.text(xs[a] + 5, y - 2, t, ha="left", va="center", fontsize=8, color=MUTED)
            y -= 8.5
            continue
        arrow(ax, (xs[a], y), (xs[b], y), dashed=b < a)
        ax.text((xs[a] + xs[b]) / 2, y + 0.8, t, ha="center", va="bottom", fontsize=8, color=INK)
        y -= 8.5
    save(fig, "lld_sequence_app.png")


def workflow():
    fig, ax = canvas(16, 6.4)
    steps = [("queue.yaml", ["ordered jobs,", "one dependency each"], "grey"),
             ("queue_runner.py", ["runs next ready job,", "marks done / failed"], "orange"),
             ("run_snapshot.py", ["freezes the code", "for the run"], "orange"),
             ("train.py / distill.py", ["config + seed;", "EMA, bf16, best on val"], "orange"),
             ("runs/<run>/", ["best.pth, metrics_*.json,", "predictions_*.csv"], "grey"),
             ("eval_all.py", ["3 test sets x 10", "degradations"], "orange"),
             ("make_tables.py\nmake_summary.py", ["tables.md, summary.md,", "figures, release/"], "blue")]
    x = 1
    prev = None
    for i, (t, lines, tone) in enumerate(steps):
        bx = box(ax, x, 22 if i % 2 == 0 else 6, 20, 30, t, lines, tone, size=8.5, title_size=8.8)
        if prev:
            arrow(ax, (prev[0] + prev[2], prev[1] + prev[3] / 2), (bx[0], bx[1] + bx[3] / 2))
        prev = bx
        x += 22.6
    ax.text(80, 60, "Every result in the report is produced by this chain; experiments.csv logs each run "
            "(commit, seed, GPU hours, best epoch, test AUCs).", ha="center", fontsize=9, color=MUTED)
    save(fig, "sys_experiment_workflow.png")


if __name__ == "__main__":
    context_deployment()
    layered()
    class_diagram()
    sequence()
    workflow()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
