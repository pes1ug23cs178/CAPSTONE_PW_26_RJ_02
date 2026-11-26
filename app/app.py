"""GRACE demo app (T6.4). Run: python app/app.py  ->  http://localhost:7860

Layout "A · Simple answer" (chosen 20 Nov 2026): a plain-language verdict card, the three clues that drove it with
the face region highlighted, what to do next, and every technical view (probability, timeline, concept maps and
contributions, switch-off test, nearest training faces, HTML report) inside "Show technical details".

Careful check (Deep): final teacher on the GPU. Quick check: S3 student via ONNX Runtime on the CPU.
"""
from __future__ import annotations

import html as htmllib
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr  # noqa: E402

from app.pipeline import CONCEPT_LABELS, LIMITATION, Analysis, analyse, band, sha256_file  # noqa: E402
from app.render import (  # noqa: E402,F401
    REPO, CODE, VIDEO_EXT, IMAGE_EXT, ONLINE, MAX_MB, N_FRAMES, SHOW_EVIDENCE, release_dir, latest_run, temperature, released_temperature, CAREFUL, QUICK, MODE_OF, quick_onnx, engine, STRONG, SOME, PLAIN, VERDICT_UI, ICONS, FRIENDLY_LIMITS, icon, glow, used_faces, preview, cropper, training_index, overlay, top_faces, timeline_fig, concept_fig, gallery_items, fig_to_b64, img_to_b64)


def message_card(title: str, body: str) -> str:
    return (f'<div class="g-card g-msg"><h2>{htmllib.escape(title)}</h2>'
            f'<p>{htmllib.escape(body)}</p></div>')


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
    if MODE_OF.get(mode) == "Quick" and not quick_onnx().exists():
        return no_result("Quick check is not available", "Please use the Careful check instead.")
    is_video = ext in VIDEO_EXT
    eng = engine(mode)
    a = analyse(path, eng, cropper(), is_video=is_video, n_frames=N_FRAMES)
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
    if not SHOW_EVIDENCE:
        pass  # online: dataset faces are not shown (licence)
    elif top.z is not None:
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
                file = gr.File(label=f"Photo or video (up to {MAX_MB} MB)", file_types=sorted(VIDEO_EXT | IMAGE_EXT))
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
            evidence = gr.Gallery(label="Nearest training faces (evidence)", columns=3, height=240, visible=SHOW_EVIDENCE)
            report = gr.File(label="Full report (HTML)")
        gr.HTML(HOW_HTML)
        gr.HTML(f'<p id="limits" class="g-limits">{FRIENDLY_LIMITS}</p>')
        if ONLINE:
            gr.HTML('<p class="g-limits">Privacy: your file is analysed on this server only for this check. It is not '
                    'shared, and it is deleted when the app restarts. Please do not upload media you have no right to use.</p>')

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

def launch():
    """Lab: 127.0.0.1 only. Online (GRACE_ONLINE=1): listen on all interfaces, one analysis at a time (CPU host)."""
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=20)
    # GRACE_SHARE=1: also open a temporary public https link (*.gradio.live, valid up to 72 h) for laptop demos
    share = os.environ.get("GRACE_SHARE") == "1"
    demo.launch(server_name="0.0.0.0" if ONLINE else "127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                css=CSS, theme=THEME, js=FORCE_LIGHT_JS, footer_links=[], max_file_size=f"{MAX_MB}mb", share=share)


if __name__ == "__main__":
    launch()
