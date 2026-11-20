"""GRACE demo app (T6.4). Run: python app/app.py  ->  http://localhost:7860

Layout "A · Simple answer" (chosen 20 Nov 2026): a plain-language verdict card, the three clues that drove it with
the face region highlighted, what to do next, and every technical view (probability, timeline, concept maps and
contributions, switch-off test, nearest training faces, HTML report) inside "Show technical details".

Careful check (Deep): final teacher on the GPU. Quick check: S3 student via ONNX Runtime on the CPU.
"""
from __future__ import annotations

import base64
import html as htmllib
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


CAREFUL, QUICK = "Careful check (recommended)", "Quick check"
MODE_OF = {CAREFUL: "Deep", QUICK: "Quick", "Deep": "Deep", "Quick": "Quick"}


def engine(mode: str):
    mode = MODE_OF.get(mode, mode)
    if mode not in _engines:
        if mode == "Quick":
            _engines[mode] = QuickEngine(QUICK_ONNX, QUICK_ONNX.with_suffix(".json"), temperature(latest_run("s3_kd_full")))
            _engines[mode].name = "Quick check (S3 student, ONNX on CPU)"
        else:
            from utils.common import final_teacher_name
            name = final_teacher_name()
            run = latest_run(name)
            _engines[mode] = DeepEngine(run, temperature(run))
            _engines[mode].name = f"Careful check (final teacher {name.split('_')[0].upper()}, GPU)"
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
    a = (0.42 * np.clip(m, 0, 1) ** 2)[..., None]
    orange = np.array([232, 106, 26], np.float32)
    return (face_crop * (1 - a) + orange * a).astype(np.uint8)


def used_faces(a: Analysis):
    used = [f for f in a.faces if f.reliable]
    return used or a.faces


def message_card(title: str, body: str) -> str:
    return (f'<div class="g-card g-msg"><h2>{htmllib.escape(title)}</h2>'
            f'<p>{htmllib.escape(body)}</p></div>')


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


def hero_html(a: Analysis, path: str, is_video: bool) -> str:
    noun = "video" if is_video else "image"
    label, head, sub, colour, tint, ic = VERDICT_UI.get(a.verdict, VERDICT_UI["Inconclusive"])
    faces = used_faces(a)
    face = max(faces, key=lambda f: f.prob) if a.verdict != "Likely real" else faces[0]
    still, length = preview(path, is_video)
    media = still if still is not None else face.crop
    pill = f"Your video · {length}" if is_video and length else ("Your video" if is_video else "Your photo")
    pos = min(max(a.prob, 0.03), 0.97) * 100
    skipped = len(a.faces) - a.n_used
    if is_video:
        basis = f"Based on {a.n_used} clear view{'s' if a.n_used != 1 else ''} of the face"
        basis += (f". {skipped} {'was' if skipped == 1 else 'were'} skipped because the head was turned away." if skipped else ".")
    else:
        basis = "Based on the face in this image." if a.n_used else "The face is turned too far for a reliable answer."
    note = f'<p class="g-note">{htmllib.escape(a.message)}</p>' if a.message else ""
    return f"""
<section class="g-card g-hero">
  <div class="g-hero-media">
    <div class="g-thumb"><img src="data:image/png;base64,{img_to_b64(media)}" alt="Your {noun}"><span class="g-pill">{pill}</span></div>
    <div class="g-caption">{htmllib.escape(Path(path).name)} · checked {datetime.now():%H:%M}</div>
  </div>
  <div class="g-hero-text">
    <div class="g-badge" style="background:{tint};color:{colour}">{icon(ic)}<span>{label}</span></div>
    <h1>{head.format(n=noun)}</h1>
    <p class="g-sub">{sub.format(n=noun)}</p>
    <div class="g-meter-label">How sure are we?</div>
    <div class="g-meter" role="img" aria-label="Position on the scale from looks real to likely fake: {label}">
      <div class="g-seg g-seg-real"></div><div class="g-seg g-seg-mid"></div><div class="g-seg g-seg-fake"></div>
      <div class="g-knob" style="left:{pos:.0f}%"></div>
    </div>
    <div class="g-scale"><span>Looks real</span><span>Can't tell</span><span>Likely fake</span></div>
    <p class="g-basis">{basis}</p>{note}
  </div>
</section>"""


def reasons_html(a: Analysis) -> str:
    faces = used_faces(a)
    score = np.mean([f.concepts for f in faces], axis=0)
    contrib = np.mean([f.contributions for f in faces], axis=0)
    found = [k for k in np.argsort(-contrib) if score[k] >= SOME][:3]
    if a.verdict == "Likely real" or not found:
        title, picks = "What we checked", [9, 1, 5]
    else:
        title = "Why we think so" if a.verdict == "Likely fake" else "What we noticed"
        normal = [k for k in np.argsort(score) if k not in found]
        picks = found + normal[:3 - len(found)]
    cards = []
    for k in picks:
        hit = score[k] >= SOME and a.verdict != "Likely real"
        if hit:
            best = max(faces, key=lambda f: f.concepts[k])
            img = glow(best.crop, best.maps[k])
            strong = score[k] >= STRONG
            tag = ("Strong clue", "g-tag-strong") if strong else ("Some clue", "g-tag-some")
            head, body = PLAIN[k][0], PLAIN[k][1]
        else:
            img, tag = faces[0].crop, ("Looks normal", "g-tag-ok")
            head, body = PLAIN[k][2], "We checked this and found nothing unusual."
        cards.append(f"""<article class="g-card g-reason">
  <img src="data:image/png;base64,{img_to_b64(img)}" alt="">
  <span class="g-tag {tag[1]}">{tag[0]}</span>
  <h3>{head}</h3><p>{body}</p></article>""")
    hint = '<p class="g-hint">Orange glow = the part of the face where we found the clue.</p>' if a.verdict != "Likely real" else ""
    return f'<section class="g-reasons"><h2>{title}</h2><div class="g-grid">{"".join(cards)}</div>{hint}</section>'


NEXT_STEPS_HTML = """
<h2 class="g-h2">What you can do now</h2>
<div class="g-steps">
  <div><span>1</span><p>Check where the file came from. Is the account or website trustworthy?</p></div>
  <div><span>2</span><p>Look for the same clip from a news outlet or the person's official page.</p></div>
  <div><span>3</span><p>Don't share it as genuine until you're sure.</p></div>
</div>"""

HEADER_HTML = """
<header class="g-header">
  <div class="g-header-in">
    <div class="g-brand">
      <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true"><rect x="2" y="2" width="28" height="28" rx="8" fill="#16181D"/><circle cx="16" cy="14" r="6" fill="none" stroke="#fff" stroke-width="2"/><path d="M9 25c2-3 4.5-4.5 7-4.5s5 1.5 7 4.5" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>
      <div>GRACE <span>Deepfake check</span></div>
    </div>
    <nav><a href="#how">How it works</a><a href="#limits">What it can't do</a></nav>
  </div>
</header>
<p class="g-intro">Upload a photo or a short video of a face. We'll tell you, in plain words, whether it looks edited.</p>"""

HOW_HTML = """
<section id="how" class="g-how">
  <h2 class="g-h2">How it works</h2>
  <div class="g-steps">
    <div><span>1</span><p>We find the face in up to 32 moments of your video (or in your photo) and skip moments where the head is turned away.</p></div>
    <div><span>2</span><p>Our model looks for 10 tell-tale signs of face swapping, such as a seam around the face or a mouth that looks edited.</p></div>
    <div><span>3</span><p>We combine them into one answer and show you the strongest clues, with the place on the face where we found them.</p></div>
  </div>
</section>"""

# Selectors start with .gradio-container so they win over Gradio's own .prose / block styles.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
body, gradio-app, .gradio-container{background:#F6F6F3 !important}
.gradio-container{max-width:1120px !important;margin:0 auto !important;padding:0 24px 48px !important;font-family:Inter,system-ui,sans-serif !important;color:#16181D}
footer{display:none !important}
.gradio-container .g-header{margin:0 calc(50% - 50vw);background:#fff;border-bottom:1px solid #E2E2DC}
.gradio-container .g-header-in{max-width:1120px;margin:0 auto;padding:16px 24px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:16px;box-sizing:border-box}
.gradio-container .g-brand{display:flex;align-items:center;gap:12px;font-family:'Space Grotesk',Inter,sans-serif;font-weight:700;font-size:20px;color:#16181D}
.gradio-container .g-brand span{color:#545A64;font-weight:500}
.gradio-container .g-header nav{display:flex;flex-wrap:wrap;gap:24px;font-size:15px}
.gradio-container .g-header nav a{color:#1F55B8;text-decoration:none}.gradio-container .g-header nav a:hover{color:#163F8A;text-decoration:underline}
.gradio-container .g-intro{margin:24px 0 0;font-size:16px;color:#3A3F47}
.gradio-container .g-upload{background:#fff;border:1px solid #E2E2DC;border-radius:20px;padding:20px;gap:20px}
.gradio-container .g-upload .block{border-color:#E2E2DC}
.gradio-container .g-card,.gradio-container .g-col-card{background:#fff;color:#16181D;border:1px solid #E2E2DC;border-radius:20px;padding:28px}
.gradio-container .g-col-card{gap:20px}
.gradio-container .g-h2,.gradio-container .g-reasons h2,.gradio-container .g-msg h2{margin:0 0 16px;font-family:'Space Grotesk',Inter,sans-serif;font-size:26px;font-weight:700;color:#16181D}
.gradio-container .g-msg p{margin:0;font-size:16px;color:#3A3F47}
.gradio-container .g-hero{display:flex;flex-wrap:wrap;gap:32px;align-items:stretch}
.gradio-container .g-hero-media{flex:1 1 320px;display:flex;flex-direction:column;gap:12px}
.gradio-container .g-thumb{position:relative;aspect-ratio:16/10;border-radius:14px;background:#2B2F36;overflow:hidden}
.gradio-container .g-thumb img{width:100%;height:100%;object-fit:cover;display:block;margin:0}
.gradio-container .g-pill{position:absolute;left:12px;bottom:12px;background:rgba(0,0,0,.6);color:#fff;font-size:13px;padding:4px 10px;border-radius:999px}
.gradio-container .g-caption{font-size:14px;color:#545A64;word-break:break-all}
.gradio-container .g-hero-text{flex:1.3 1 420px;display:flex;flex-direction:column;gap:20px;justify-content:center}
.gradio-container .g-badge{display:inline-flex;align-self:flex-start;align-items:center;gap:8px;font-weight:600;font-size:15px;padding:8px 14px;border-radius:999px}
.gradio-container .g-hero h1{margin:0;font-family:'Space Grotesk',Inter,sans-serif;font-size:44px;line-height:1.1;font-weight:700;color:#16181D}
.gradio-container .g-sub{margin:0;font-size:18px;line-height:1.55;color:#3A3F47}
.gradio-container .g-meter-label{font-size:14px;font-weight:600;color:#3A3F47;margin-bottom:-12px}
.gradio-container .g-meter{position:relative;height:14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}
.gradio-container .g-seg-real{background:#C9D8F2;border-radius:999px 0 0 999px}.gradio-container .g-seg-mid{background:#DADCE0}.gradio-container .g-seg-fake{background:#F4C9A9;border-radius:0 999px 999px 0}
.gradio-container .g-knob{position:absolute;top:-6px;width:26px;height:26px;margin-left:-13px;border-radius:999px;background:#fff;border:3px solid #16181D;box-sizing:border-box}
.gradio-container .g-scale{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));font-size:13px;color:#545A64;margin-top:-12px}
.gradio-container .g-scale span:nth-child(2){text-align:center}.gradio-container .g-scale span:nth-child(3){text-align:right}
.gradio-container .g-basis,.gradio-container .g-note{margin:0;font-size:14px;color:#545A64}.gradio-container .g-note{color:#8A4B12}
.gradio-container .g-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.gradio-container .g-reason{padding:20px;display:flex;flex-direction:column;gap:14px;border-radius:16px}
.gradio-container .g-reason img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px;margin:0}
.gradio-container .g-reason h3{margin:0;font-size:18px;line-height:1.35;font-weight:600;color:#16181D}
.gradio-container .g-reason p{margin:0;font-size:15px;line-height:1.55;color:#3A3F47}
.gradio-container .g-tag{align-self:flex-start;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px}
.gradio-container .g-tag-strong{background:#FCEBDD;color:#9A3A0C}.gradio-container .g-tag-some{background:#FDF3E7;color:#8A4B12}.gradio-container .g-tag-ok{background:#E3ECFA;color:#1F55B8}
.gradio-container .g-hint{font-size:14px;color:#545A64;margin:12px 0 0}
.gradio-container .g-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.gradio-container .g-steps div{display:flex;gap:12px}
.gradio-container .g-steps p{margin:0;font-size:15px;line-height:1.5;color:#16181D}
.gradio-container .g-steps span{flex:none;width:32px;height:32px;border-radius:999px;background:#E3ECFA;color:#1F55B8;font-weight:600;display:flex;align-items:center;justify-content:center}
.gradio-container .g-actions{gap:12px;justify-content:flex-start}
.gradio-container .g-actions button,.gradio-container .g-actions a{flex:0 0 auto !important;min-width:180px;min-height:44px;border-radius:12px !important;font-weight:600}
.gradio-container .g-actions .secondary,.gradio-container #g-save{border:1px solid #C9CBC4 !important;background:#fff !important;color:#16181D !important}
.gradio-container .g-tech{background:#fff;border:1px solid #E2E2DC !important;border-radius:16px !important}
.gradio-container .g-how{padding:8px 0 0}
.gradio-container .g-limits{margin:0;font-size:14px;line-height:1.6;color:#545A64;max-width:760px}
@media (max-width:640px){.gradio-container .g-hero h1{font-size:32px}.gradio-container .g-card,.gradio-container .g-col-card{padding:20px}}
"""

FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); document.querySelector('gradio-app')?.classList.remove('dark'); }"


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


def no_result(title: str, body: str, tech: str = ""):
    """Outputs when there is no verdict: a friendly message card, everything else empty."""
    return (message_card(title, body), "", gr.Column(visible=False), gr.DownloadButton(visible=False),
            tech, None, None, None, None, None, None)


def run_analysis(file, mode, concept_name, opacity):
    if file is None:
        return no_result("Choose a file first", "Upload a photo or a short video that shows a face, then press Check.")
    path = file if isinstance(file, str) else file.name
    ext = Path(path).suffix.lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        return no_result("We can't read this file type",
                         f"Please upload a photo (JPG, PNG, WEBP) or a video (MP4, MOV, AVI, MKV, WEBM), not {ext or 'this file'}.")
    if Path(path).stat().st_size > MAX_MB * 1e6:
        return no_result("This file is too large", f"Please upload a file smaller than {MAX_MB} MB, or a shorter clip.")
    if MODE_OF.get(mode) == "Quick" and not QUICK_ONNX.exists():
        return no_result("Quick check is not available", "Please use the Careful check instead.")
    is_video = ext in VIDEO_EXT
    eng = engine(mode)
    a = analyse(path, eng, cropper(), is_video=is_video)
    if not a.faces:
        return no_result("We couldn't find a face",
                         f"We need a clear, front-facing face to check. {'Try a clip where the face is visible for a few seconds.' if is_video else 'Try a photo where the face is clearly visible.'}",
                         f"**{a.message}**")
    friendly = (hero_html(a, path, is_video), reasons_html(a))
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
    return friendly + (gr.Column(visible=True), gr.DownloadButton(value=report, visible=True),
                       verdict, timeline_fig(a) if len(a.faces) > 1 else None, gallery_items(a, ci, opacity),
                       concept_fig(np.mean([f.contributions for f in a.faces], 0), float(np.mean([f.residual for f in a.faces]))),
                       evidence, report, a)


def clear_all():
    return (None, "", "", gr.Column(visible=False), gr.DownloadButton(visible=False),
            "", None, None, None, None, None, None)


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
    with gr.Blocks(title="GRACE · Deepfake check") as demo:
        gr.HTML(HEADER_HTML)
        state = gr.State()
        with gr.Row(equal_height=False, elem_classes="g-upload"):
            with gr.Column(scale=3):
                file = gr.File(label="Photo or video (up to 200 MB)", file_types=sorted(VIDEO_EXT | IMAGE_EXT))
            with gr.Column(scale=2):
                # Careful (teacher) by default: on re-cropped real faces the student drifts towards 0.5 (app check, 19 Nov)
                mode = gr.Radio([CAREFUL, QUICK], value=CAREFUL, label="Type of check",
                                info="Careful: our most accurate model, a few seconds per video. "
                                     "Quick: a small model that runs on any computer; faster, but makes more mistakes.")
                go = gr.Button("Check this file", variant="primary", size="lg")
        hero = gr.HTML()
        reasons = gr.HTML()
        with gr.Column(visible=False, elem_classes="g-col-card") as after:
            gr.HTML(NEXT_STEPS_HTML)
            with gr.Row(elem_classes="g-actions"):
                again = gr.Button("Check another file", variant="primary")
                save = gr.DownloadButton("Save this result", variant="secondary", visible=False, elem_id="g-save")
        with gr.Accordion("Show technical details (for experts)", open=False, elem_classes="g-tech"):
            verdict = gr.Markdown()
            timeline = gr.Plot(label="Score for each face over time")
            with gr.Row():
                with gr.Column():
                    concept_dd = gr.Dropdown(CONCEPT_LABELS, value=CONCEPT_LABELS[0], label="Concept map to overlay")
                    opacity = gr.Slider(0, 1, value=0.45, step=0.05, label="Overlay opacity")
                    gallery = gr.Gallery(label="Most suspicious faces", columns=4, height=260, elem_id="g-gallery")
                with gr.Column():
                    concepts = gr.Plot(label="Concept contributions")
                    off = gr.CheckboxGroup(CONCEPT_LABELS, label="Switch concepts off")
                    rescore_btn = gr.Button("Re-score")
                    rescore_out = gr.Markdown(elem_id="g-rescore")
            evidence = gr.Gallery(label="Nearest training faces (evidence)", columns=3, height=240)
            report = gr.File(label="Full report (HTML)")
        gr.HTML(HOW_HTML)
        gr.HTML(f'<p id="limits" class="g-limits">{FRIENDLY_LIMITS}</p>')

        outs = [hero, reasons, after, save, verdict, timeline, gallery, concepts, evidence, report, state]
        go.click(run_analysis, [file, mode, concept_dd, opacity], outs, api_name="check", scroll_to_output=True)
        again.click(clear_all, None, [file] + outs, api_name="clear")
        concept_dd.change(update_overlay, [state, concept_dd, opacity], gallery, api_name="overlay_concept")
        opacity.change(update_overlay, [state, concept_dd, opacity], gallery, api_name="overlay_opacity")
        rescore_btn.click(rescore, [state, mode, off], rescore_out, api_name="rescore")
    return demo


THEME = gr.themes.Default(primary_hue="blue", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    body_background_fill="#F6F6F3", body_background_fill_dark="#F6F6F3",
    button_primary_background_fill="#16181D", button_primary_background_fill_hover="#2B2F36",
    button_primary_background_fill_dark="#16181D", button_primary_background_fill_hover_dark="#2B2F36",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_text_color="#16181D", button_secondary_text_color_dark="#16181D",
    button_secondary_border_color="#C9CBC4", button_secondary_border_color_dark="#C9CBC4")

if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, css=CSS, theme=THEME, js=FORCE_LIGHT_JS,
                      footer_links=[])
