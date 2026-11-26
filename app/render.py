"""Shared, UI-free parts of the GRACE app: settings, model loading, plain-language text and image helpers.

Used by the web server (app/server.py) and the old Gradio app (app/app.py). No Gradio import here.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import List

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
# data (runs/, cache/, release/) live in the real repo; under scripts/run_snapshot.py the code runs from a frozen copy
REPO = Path(os.environ.get("GRACE_REPO_ROOT") or CODE).resolve()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.pipeline import CONCEPT_LABELS, Analysis, DeepEngine, FaceCropper, QuickEngine  # noqa: E402,F401

PLOT_LOCK = threading.Lock()  # pyplot is not thread-safe
_engine_lock = threading.Lock()

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------- deployment settings (environment variables; the lab defaults are unchanged) ----------
# GRACE_ONLINE=1         public deployment: smaller uploads, no dataset faces shown, privacy note
# GRACE_MODEL_REPO=u/r   download the release files from this Hugging Face model repo (else use ./release)
# GRACE_MODEL_REVISION   pin a model-repo commit (default: latest)
# GRACE_FRAMES           frames sampled per video (default 32; 16 online keeps CPU checks around 30 s)
# GRACE_MAX_MB           upload limit in MB (default 200; 50 online)
ONLINE = os.environ.get("GRACE_ONLINE") == "1"
MAX_MB = int(os.environ.get("GRACE_MAX_MB") or (50 if ONLINE else 200))
N_FRAMES = int(os.environ.get("GRACE_FRAMES") or (16 if ONLINE else 32))
SHOW_EVIDENCE = not ONLINE  # nearest training faces are dataset images we may not publish

_engines: dict = {}
_cropper = None
_index = None
_release = None


def release_dir() -> Path:
    """Folder with the released model files: a Hugging Face model repo when GRACE_MODEL_REPO is set, else ./release."""
    global _release
    if _release is None:
        repo_id = os.environ.get("GRACE_MODEL_REPO")
        if repo_id:
            from huggingface_hub import snapshot_download
            _release = Path(snapshot_download(repo_id, revision=os.environ.get("GRACE_MODEL_REVISION") or None,
                                              token=os.environ.get("HF_TOKEN") or None))
        else:
            _release = REPO / "release"
    return _release


def latest_run(name: str):
    runs = sorted((REPO / "runs").glob(f"{name}_s42_*"))
    runs = [r for r in runs if (r / "best.pth").exists()]
    return runs[-1] if runs else None


def temperature(run) -> float:
    p = run / "temperature.json" if run else None
    return json.loads(p.read_text())["temperature"] if p and p.exists() else 1.0


def released_temperature(stem: str) -> float:
    p = release_dir() / f"{stem}_temperature.json"
    return json.loads(p.read_text())["temperature"] if p.exists() else 1.0


CAREFUL, QUICK = "Careful check (recommended)", "Quick check"
MODE_OF = {CAREFUL: "Deep", QUICK: "Quick", "Deep": "Deep", "Quick": "Quick"}


def quick_onnx() -> Path:
    return release_dir() / "grace_student.onnx"


def engine(mode: str):
    """Lab: the latest run folders under runs/. Elsewhere (no runs/): the released files. Loaded once, shared."""
    mode = MODE_OF.get(mode, mode)
    with _engine_lock:
        return _load_engine(mode)


def _load_engine(mode: str):
    if mode not in _engines:
        if mode == "Quick":
            run = latest_run("s3_kd_full")
            t = temperature(run) if run else released_temperature("grace_student_s3")
            _engines[mode] = QuickEngine(quick_onnx(), quick_onnx().with_suffix(".json"), t)
            _engines[mode].name = "Quick check (small student model, CPU)"
        else:
            from utils.common import final_teacher_name
            name = final_teacher_name()
            run = latest_run(name)
            if run is None:
                run = release_dir() / "grace_teacher.pth"
                t = released_temperature("grace_teacher")
            else:
                t = temperature(run)
            e = DeepEngine(run, t)
            _engines[mode] = e
            _engines[mode].name = f"Careful check (final teacher {name.split('_')[0].upper()}, {e.device.type.upper()})"
    return _engines[mode]


# ---------- plain-language layer ----------
# Concept scores (sigmoid) on FF++ test faces, 20 Nov: reals 0.00-0.07 on every concept, fakes 0.3-0.9 on most.
STRONG, SOME = 0.6, 0.25
PLAIN = [  # (title when found, one-line explanation, title when not found)
    ("The face looks pasted on", "The face seems blended onto the head, like a sticker.", "The face sits naturally on the head"),
    ("Skin colour does not quite match", "The face is a slightly different tone from the neck and ears.", "Skin colour matches"),
    ("Part of the face is unusually blurry", "One area is softer than the rest, which editing often leaves behind.", "No unusual blur"),
    ("Part of the face looks over-sharpened", "Some edges look harsher than a normal camera would make them.", "No harsh sharpening"),
    ("Signs of resizing", "Parts of the face look stretched or re-drawn.", "No signs of resizing"),
    ("The mouth area looks edited", "Lips and teeth look different from the rest of the face, a common sign of a swap.", "The mouth looks natural"),
    ("The eye area looks edited", "Eyes or eyebrows look different from the rest of the face.", "The eyes look natural"),
    ("The whole face may be replaced", "Clues cover the entire face, not just one part.", "No sign of a full face swap"),
    ("The face is slightly out of place", "Features do not line up perfectly with the head.", "Features line up with the head"),
    ("The edge around the face is uneven", "There is a visible seam along the jaw, cheeks or hairline.", "No seam around the face"),
]
VERDICT_UI = {  # band -> label, headline template, explanation, colour, tint, icon
    "Likely fake": ("Likely fake", "This {n} is likely fake",
                    "We found clear signs that the face in this {n} was edited or swapped onto someone else.",
                    "#B4410E", "#FCEBDD", "warn"),
    "Inconclusive": ("Can't tell", "We can't tell for sure",
                     "Some clues point to editing, others don't. Treat this {n} with care and check where it came from.",
                     "#4E5560", "#ECEDEF", "question"),
    "Likely real": ("Looks real", "This {n} looks real",
                    "We didn't find signs that the face was edited. That is not a guarantee, but it is a good sign.",
                    "#1F55B8", "#E3ECFA", "check"),
}
ICONS = {
    "warn": '<path d="M12 3l9 16H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "question": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7"/><path d="M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 10"/>',
}
FRIENDLY_LIMITS = ("No detector is perfect. GRACE learned from face-swap videos and can be fooled by new editing tools, "
                   "heavy compression, or faces seen from the side. Use this as one piece of evidence, not the final word.")


def icon(name: str, size: int = 18) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


def glow(face_crop: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Orange glow where the concept map is high (plain view; the jet overlay stays in the technical view)."""
    # soft glow: weak map values fade out (m**2), strongest areas reach 42% orange, so the face stays visible
    a = (0.42 * np.clip(np.asarray(m, np.float32), 0, 1) ** 2)[..., None]
    orange = np.array([232, 106, 26], np.float32)
    return (face_crop * (1 - a) + orange * a).astype(np.uint8)


def used_faces(a: Analysis):
    used = [f for f in a.faces if f.reliable]
    return used or a.faces


def preview(path: str, is_video: bool):
    """A still of the uploaded file for the verdict card (middle frame of a video) and its length as m:ss."""
    import cv2
    length = ""
    if is_video:
        cap = cv2.VideoCapture(path)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
        ok, bgr = cap.read()
        cap.release()
        if fps > 0 and n > 0:
            s = int(round(n / fps))
            length = f"{s // 60}:{s % 60:02d}"
    else:
        bgr = cv2.imread(path)
        ok = bgr is not None
    if not ok:
        return None, length
    h, w = bgr.shape[:2]
    if w > 720:
        bgr = cv2.resize(bgr, (720, int(h * 720 / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), length


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
    heat = (matplotlib.colormaps["jet"](np.asarray(m, np.float32))[..., :3] * 255).astype(np.uint8)
    return ((1 - opacity) * face_crop + opacity * heat).astype(np.uint8)


def top_faces(a: Analysis, k: int = 4):
    """Most suspicious faces, front-facing ones first (extreme poses are shown only if nothing else is left)."""
    return sorted(range(len(a.faces)), key=lambda i: (not a.faces[i].reliable, -a.faces[i].prob))[:k]


def timeline_fig(a: Analysis):
    """Matplotlib figure; callers hold PLOT_LOCK when threads are involved."""
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


# ---------- JSON result for the web page (app/server.py); same wording and rules as the Gradio cards ----------
def jpg_b64(img: np.ndarray, quality: int = 88) -> str:
    """data: URI of an RGB image as JPEG (about 5x smaller than PNG for face crops)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def verdict_data(a: Analysis, path: str, is_video: bool, name: str) -> dict:
    noun = "video" if is_video else "image"
    label, head, sub, colour, tint, ic = VERDICT_UI.get(a.verdict, VERDICT_UI["Inconclusive"])
    faces = used_faces(a)
    face = max(faces, key=lambda f: f.prob) if a.verdict != "Likely real" else faces[0]
    still, length = preview(path, is_video)
    skipped = len(a.faces) - a.n_used
    if is_video:
        basis = f"Based on {a.n_used} clear view{'s' if a.n_used != 1 else ''} of the face"
        basis += (f". {skipped} {'was' if skipped == 1 else 'were'} skipped because the head was turned away." if skipped else ".")
    else:
        basis = "Based on the face in this image." if a.n_used else "The face is turned too far for a reliable answer."
    return {"band": a.verdict, "label": label, "headline": head.format(n=noun), "explanation": sub.format(n=noun),
            "colour": colour, "tint": tint, "icon": ic, "prob": round(a.prob, 4),
            "knob": round(min(max(a.prob, 0.03), 0.97) * 100, 1), "basis": basis, "note": a.message,
            "media": jpg_b64(still if still is not None else face.crop),
            "pill": f"Your video · {length}" if is_video and length else ("Your video" if is_video else "Your photo"),
            "file": name, "kind": noun}


def clues_data(a: Analysis) -> dict:
    faces = used_faces(a)
    score = np.mean([f.concepts for f in faces], axis=0)
    contrib = np.mean([f.contributions for f in faces], axis=0)
    found = [k for k in np.argsort(-contrib) if score[k] >= SOME][:3]
    if a.verdict == "Likely real" or not found:
        title, picks = "What we checked", [9, 1, 5]
    else:
        title = "Why we think so" if a.verdict == "Likely fake" else "What we noticed"
        normal = [int(k) for k in np.argsort(score) if k not in found]
        picks = [int(k) for k in found] + normal[:3 - len(found)]
    cards = []
    for k in picks:
        hit = score[k] >= SOME and a.verdict != "Likely real"
        if hit:
            best = max(faces, key=lambda f: f.concepts[k])
            strong = score[k] >= STRONG
            cards.append({"concept": int(k), "img": jpg_b64(glow(best.crop, best.maps[k])),
                          "tag": "Strong clue" if strong else "Some clue", "level": "strong" if strong else "some",
                          "title": PLAIN[k][0], "body": PLAIN[k][1]})
        else:
            cards.append({"concept": int(k), "img": jpg_b64(faces[0].crop), "tag": "Looks normal", "level": "ok",
                          "title": PLAIN[k][2], "body": "We checked this and found nothing unusual."})
    return {"title": title, "cards": cards, "hint": a.verdict != "Likely real"}


def gallery_data(a: Analysis, concept_idx: int, opacity: float) -> List[dict]:
    return [{"img": jpg_b64(img), "caption": cap} for img, cap in gallery_items(a, concept_idx, opacity)]


def technical_data(a: Analysis, eng, concept_idx: int = 0, opacity: float = 0.45) -> dict:
    skipped = len(a.faces) - a.n_used
    used_txt = (f"verdict from {a.n_used} of {len(a.faces)} faces ({skipped} skipped: head turned or tilted beyond "
                "the training range)" if skipped else f"{len(a.faces)} faces from {a.n_frames} frames")
    contrib = np.mean([f.contributions for f in a.faces], 0)
    with PLOT_LOCK:
        timeline = "data:image/png;base64," + fig_to_b64(timeline_fig(a)) if len(a.faces) > 1 else None
        concepts = "data:image/png;base64," + fig_to_b64(concept_fig(contrib, float(np.mean([f.residual for f in a.faces]))))
    return {"verdict": a.verdict, "prob": round(a.prob, 4), "model": eng.name, "version": str(eng.version),
            "used": used_txt, "flicker": a.flicker, "drift": a.drift, "timeline": timeline, "concepts": concepts,
            "contributions": [{"label": CONCEPT_LABELS[k], "value": round(float(contrib[k]), 4)} for k in np.argsort(-contrib)],
            "gallery": gallery_data(a, concept_idx, opacity), "evidence": evidence_data(a)}


def evidence_data(a: Analysis) -> List[dict]:
    """Nearest training faces (lab only: dataset images are not shown online). Empty when unavailable."""
    top = a.faces[top_faces(a)[0]]
    if not SHOW_EVIDENCE or top.z is None:
        return []
    try:
        from data.dataset import load_rgb
        from utils.config import load_paths
        root = load_paths()["data_root"]
        return [{"img": jpg_b64(load_rgb(root / n["frame_path"])), "caption": f"{n['method']} (sim {n['similarity']:.2f})"}
                for n in training_index().nearest(top.z, 3)]
    except Exception:  # noqa: BLE001
        return []


def compact(a: Analysis) -> Analysis:
    """Smaller copy for keeping in memory: concept maps as float16 (32 faces: 64 MB -> 32 MB)."""
    for f in a.faces:
        f.maps = f.maps.astype(np.float16)
    return a


def report_html(a: Analysis, eng, name: str, sha256: str, when: str, concept_idx: int = 0) -> str:
    import html as htmllib
    from app.pipeline import LIMITATION
    contrib = np.mean([f.contributions for f in a.faces], axis=0)
    rows = "".join(f"<tr><td>{CONCEPT_LABELS[k]}</td><td>{contrib[k]:.3f}</td></tr>" for k in np.argsort(-contrib))
    tops = "".join(f'<figure><img src="{jpg_b64(img)}" width="224"><figcaption>{cap}</figcaption></figure>'
                   for img, cap in gallery_items(a, concept_idx, 0.45))
    with PLOT_LOCK:
        timeline = f'<h3>Timeline</h3><img src="data:image/png;base64,{fig_to_b64(timeline_fig(a))}">' if len(a.faces) > 1 else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>GRACE report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 16px}} figure{{display:inline-block;margin:4px}}
table{{border-collapse:collapse}} td{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>
<h1>Deepfake analysis report</h1>
<p><b>File:</b> {htmllib.escape(name)}<br><b>SHA-256:</b> {sha256}<br><b>Time:</b> {when}<br>
<b>Mode / model:</b> {htmllib.escape(eng.name)} ({htmllib.escape(str(eng.version))})</p>
<h2>Verdict: {a.verdict} (calibrated probability {a.prob:.2f})</h2>
<p>Faces analysed: {len(a.faces)} of {a.n_frames} sampled frames; {a.n_used} front-facing faces used for the verdict.</p>
{timeline}
<h3>Most suspicious frames (map: {CONCEPT_LABELS[concept_idx]})</h3>{tops}
<h3>Concept contributions (mean over faces)</h3><table>{rows}</table>
<p><i>{LIMITATION}</i></p></body></html>"""
