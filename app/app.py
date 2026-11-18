"""GRACE demo app (T6.4). Run: python app/app.py  ->  http://localhost:7860

Quick mode: S3 student via ONNX Runtime on the CPU. Deep mode: G3 teacher on the GPU.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
# data (runs/, cache/, release/) live in the real repo; under scripts/run_snapshot.py the code runs from a frozen copy
REPO = Path(os.environ.get("GRACE_REPO_ROOT") or CODE).resolve()

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from app.pipeline import (CONCEPT_LABELS, LIMITATION, Analysis, DeepEngine, FaceCropper, QuickEngine,  # noqa: E402
                          analyse, band, sha256_file)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_MB = 200

_engines: dict = {}
_cropper = None
_index = None


QUICK_ONNX = REPO / "release" / "grace_student.onnx"


def latest_run(name: str):
    runs = sorted((REPO / "runs").glob(f"{name}_s42_*"))
    runs = [r for r in runs if (r / "best.pth").exists()]
    return runs[-1] if runs else None


def temperature(run) -> float:
    p = run / "temperature.json" if run else None
    return json.loads(p.read_text())["temperature"] if p and p.exists() else 1.0


def engine(mode: str):
    if mode not in _engines:
        if mode == "Quick":
            _engines[mode] = QuickEngine(QUICK_ONNX, QUICK_ONNX.with_suffix(".json"), temperature(latest_run("s3_kd_full")))
        else:
            from utils.common import final_teacher_name
            run = latest_run(final_teacher_name())
            _engines[mode] = DeepEngine(run, temperature(run))
    return _engines[mode]


def cropper():
    global _cropper
    if _cropper is None:
        _cropper = FaceCropper()
    return _cropper


def training_index():
    global _index
    if _index is None:
        from explain.retrieval import TrainingIndex
        _index = TrainingIndex(REPO / "cache" / "teacher", REPO / "cache" / "sbi_pool" / "student_train.csv")
    return _index


def overlay(face_crop: np.ndarray, m: np.ndarray, opacity: float) -> np.ndarray:
    heat = (matplotlib.colormaps["jet"](m)[..., :3] * 255).astype(np.uint8)
    return ((1 - opacity) * face_crop + opacity * heat).astype(np.uint8)


def top_faces(a: Analysis, k: int = 4):
    """Most suspicious faces, front-facing ones first (extreme poses are shown only if nothing else is left)."""
    return sorted(range(len(a.faces)), key=lambda i: (not a.faces[i].reliable, -a.faces[i].prob))[:k]


def timeline_fig(a: Analysis):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    probs = [f.prob for f in a.faces]
    ax.plot(probs, color="tab:blue", alpha=0.5)
    ok = [i for i, f in enumerate(a.faces) if f.reliable]
    bad = [i for i, f in enumerate(a.faces) if not f.reliable]
    ax.scatter(ok, [probs[i] for i in ok], color="tab:blue", zorder=3, label="used for the verdict")
    if bad:
        ax.scatter(bad, [probs[i] for i in bad], color="lightgray", edgecolor="gray", zorder=3,
                   label="not used: head turned or tilted too far")
        ax.legend(fontsize=7, loc="upper left")
    ax.axhspan(0.35, 0.65, color="orange", alpha=0.12)
    ax.set(ylim=(0, 1), xlabel="Face frame (in time order)", ylabel="Fake probability")
    extra = []
    if a.flicker is not None:
        extra.append(f"concept flicker {a.flicker:.3f}")
    if a.drift is not None:
        extra.append(f"identity drift {a.drift:.3f}")
    ax.set_title(" | ".join(extra) or "Per-frame score", fontsize=9)
    fig.tight_layout()
    return fig


def concept_fig(contrib: np.ndarray, residual: float):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = np.argsort(-contrib)
    ax.barh([CONCEPT_LABELS[i] for i in order], contrib[order], color="tab:red")
    ax.barh(["Other evidence (residual)"], [residual], color="tab:gray")
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to the fake score (logit)")
    fig.tight_layout()
    return fig


def gallery_items(a: Analysis, concept_idx: int, opacity: float):
    return [(overlay(a.faces[i].crop, a.faces[i].maps[concept_idx], opacity),
             f"frame {i}: p={a.faces[i].prob:.2f}" + ("" if a.faces[i].reliable else " (extreme pose, not used)"))
            for i in top_faces(a)]


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_to_b64(img: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def build_report(path: str, a: Analysis, eng, concept_idx: int) -> str:
    mean_contrib = np.mean([f.contributions for f in a.faces], axis=0)
    rows = "".join(f"<tr><td>{CONCEPT_LABELS[k]}</td><td>{mean_contrib[k]:.3f}</td></tr>" for k in np.argsort(-mean_contrib))
    tops = "".join(f'<figure><img src="data:image/png;base64,{img_to_b64(img)}" width="224"><figcaption>{cap}</figcaption></figure>'
                   for img, cap in gallery_items(a, concept_idx, 0.45))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>GRACE report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 16px}} figure{{display:inline-block;margin:4px}}
table{{border-collapse:collapse}} td{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>
<h1>Deepfake analysis report</h1>
<p><b>File:</b> {Path(path).name}<br><b>SHA-256:</b> {sha256_file(path)}<br><b>Time:</b> {datetime.now():%Y-%m-%d %H:%M:%S}<br>
<b>Mode / model:</b> {eng.name} ({eng.version})</p>
<h2>Verdict: {a.verdict} (calibrated probability {a.prob:.2f})</h2>
<p>Faces analysed: {len(a.faces)} of {a.n_frames} sampled frames; {a.n_used} front-facing faces used for the verdict.</p>
<h3>Timeline</h3><img src="data:image/png;base64,{fig_to_b64(timeline_fig(a))}">
<h3>Most suspicious frames (map: {CONCEPT_LABELS[concept_idx]})</h3>{tops}
<h3>Concept contributions (mean over faces)</h3><table>{rows}</table>
<p><i>{LIMITATION}</i></p></body></html>"""
    out = Path(tempfile.gettempdir()) / f"grace_report_{datetime.now():%Y%m%d_%H%M%S}.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


def run_analysis(file, mode, concept_name, opacity):
    if file is None:
        return ("Upload an image or a video first.",) + (None,) * 6
    path = file if isinstance(file, str) else file.name
    ext = Path(path).suffix.lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        return (f"Unsupported file type {ext}.",) + (None,) * 6
    if Path(path).stat().st_size > MAX_MB * 1e6:
        return (f"File is larger than {MAX_MB} MB.",) + (None,) * 6
    if mode == "Quick" and not QUICK_ONNX.exists():
        return ("**Quick mode is not available yet**: the compressed student model has not been trained and exported "
                "(`release/grace_student.onnx`). Choose **Deep**.",) + (None,) * 6
    eng = engine(mode)
    a = analyse(path, eng, cropper(), is_video=ext in VIDEO_EXT)
    if not a.faces:
        return (f"**{a.message}**",) + (None,) * 6
    ci = CONCEPT_LABELS.index(concept_name)
    skipped = len(a.faces) - a.n_used
    used_txt = (f"verdict from {a.n_used} of {len(a.faces)} faces ({skipped} skipped: head turned or tilted beyond "
                "the training range)" if skipped else f"{len(a.faces)} faces from {a.n_frames} frames")
    verdict = (f"## {a.verdict}\nCalibrated fake probability **{a.prob:.2f}** · mode: {eng.name} · {used_txt}"
               + (f"\n\n**{a.message}**" if a.message else "") + f"\n\n_{LIMITATION}_")
    top = a.faces[top_faces(a)[0]]
    evidence = []
    if top.z is not None:
        try:
            from data.dataset import load_rgb
            from utils.config import load_paths
            root = load_paths()["data_root"]
            evidence = [(load_rgb(root / n["frame_path"]), f"{n['method']} (sim {n['similarity']:.2f})")
                        for n in training_index().nearest(top.z, 3)]
        except Exception as e:  # noqa: BLE001
            evidence = []
            if not (REPO / "cache" / "teacher").exists():
                verdict += "\n\n(Nearest training faces appear once the teacher cache is built, after the final teacher is chosen.)"
            else:
                verdict += f"\n\n(Evidence unavailable: {type(e).__name__})"
    else:
        verdict += "\n\n(Nearest training faces are shown in Deep mode.)"
    report = build_report(path, a, eng, ci)
    return (verdict, timeline_fig(a) if len(a.faces) > 1 else None, gallery_items(a, ci, opacity),
            concept_fig(np.mean([f.contributions for f in a.faces], 0), float(np.mean([f.residual for f in a.faces]))),
            evidence, report, a)


def update_overlay(a, concept_name, opacity):
    if a is None or not a.faces:
        return None
    return gallery_items(a, CONCEPT_LABELS.index(concept_name), opacity)


def rescore(a, mode, off_names):
    if a is None or not a.faces:
        return "Run an analysis first."
    off = [CONCEPT_LABELS.index(n) for n in off_names or []]
    eng = engine(mode)
    p = float(np.mean([eng.rescore(f, off) for f in a.faces]))
    return (f"With {', '.join(off_names) if off_names else 'no concepts'} switched off: "
            f"probability **{p:.2f}** → **{band(p)}** (was {a.prob:.2f}, {a.verdict}).")


def build_ui():
    with gr.Blocks(title="GRACE deepfake detector") as demo:
        gr.Markdown("# GRACE deepfake detector\nUpload a face image or a short video. "
                    "*Quick* runs the compressed student on the CPU; *Deep* runs the full teacher on the GPU.")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Image or video (MP4, MOV, AVI; up to 200 MB)",
                               file_types=sorted(VIDEO_EXT | IMAGE_EXT))
                mode = gr.Radio(["Quick", "Deep"], value="Quick" if QUICK_ONNX.exists() else "Deep", label="Mode")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                timeline = gr.Plot(label="Timeline")
        with gr.Row():
            with gr.Column():
                concept_dd = gr.Dropdown(CONCEPT_LABELS, value=CONCEPT_LABELS[0], label="Concept map to overlay")
                opacity = gr.Slider(0, 1, value=0.45, step=0.05, label="Overlay opacity")
                gallery = gr.Gallery(label="Most suspicious faces", columns=4, height=260)
            with gr.Column():
                concepts = gr.Plot(label="Concept contributions")
                off = gr.CheckboxGroup(CONCEPT_LABELS, label="Switch concepts off")
                rescore_btn = gr.Button("Re-score")
                rescore_out = gr.Markdown()
        evidence = gr.Gallery(label="Nearest training faces (evidence)", columns=3, height=240)
        report = gr.File(label="Download report (HTML)")
        go.click(run_analysis, [file, mode, concept_dd, opacity], [verdict, timeline, gallery, concepts, evidence, report, state])
        concept_dd.change(update_overlay, [state, concept_dd, opacity], gallery)
        opacity.change(update_overlay, [state, concept_dd, opacity], gallery)
        rescore_btn.click(rescore, [state, mode, off], rescore_out)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
