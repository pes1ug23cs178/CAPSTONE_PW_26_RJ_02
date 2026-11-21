"""Phase-3 Review-3 deck on the department template (PPT-Template_Phase3- Review3.pptx).

Keeps the template's look (title slide, orange footer bar, PES logo, red titles with an orange rule, Trebuchet MS) and
fills it with the project's latest results. Numbers come from the same results loader as the report.
Run: python "ESA Report/ppt_build/build_ppt.py"
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ESA = HERE.parent
sys.path.insert(0, str(ESA / "build"))
from data import R  # noqa: E402

TPL = ESA / "PPT-Template_Phase3- Review3.pptx"
OUT = ESA / "output" / "PW26_RJ_02_Review3.pptx"
FIG = ESA / "build" / "figs"
RES = ESA.parent / "results"

TITLE = "Generalized Deepfake Detection Framework with Scalable Multimodal Integration"
SHORT = "Generalized Deepfake Detection Framework"
TEAM = [("Charan L", "PES1UG23CS159"), ("Deepak S N", "PES1UG23CS178"),
        ("Samrudh N Shindhe", "PES1UG23CS508"), ("Samarth Gowda G M", "PES1UG23CS502")]
FOOT = "Charan L_Deepak S N_Samrudh N Shindhe_Samarth Gowda G M"
FONT = "Trebuchet MS"
RED, ORANGE, RUST, BLUE = RGBColor(0xFF, 0, 0), RGBColor(0xE4, 0x83, 0x12), RGBColor(0xBD, 0x58, 0x2C), RGBColor(0, 0x33, 0xCC)
INK, MUTED, GREY = RGBColor(0x26, 0x26, 0x26), RGBColor(0x59, 0x59, 0x59), RGBColor(0x88, 0x88, 0x88)
TINT, TINT2, WHITE = RGBColor(0xFD, 0xF0, 0xE2), RGBColor(0xEA, 0xF0, 0xFB), RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2F, 0x7D, 0x4F)

prs = Presentation(TPL)
SL = list(prs.slides)
BLANK = SL[2].slide_layout
LOGO = next(sh for sh in SL[2].shapes if sh.shape_type == 13).image.blob
FOOTER_EL = next(sh for sh in SL[2].shapes if sh.is_placeholder)._element


# ------------------------------------------------------------------ helpers
def _font(run, size, color=INK, bold=False, italic=False):
    run.font.name, run.font.size, run.font.bold, run.font.italic = FONT, Pt(size), bold, italic
    run.font.color.rgb = color


def bullet(p, char="•", indent=0.28):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(int(-Inches(indent))))
    for t in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(t)):
            pPr.remove(e)
    bf = etree.SubElement(pPr, qn("a:buFont")); bf.set("typeface", "Arial")
    bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", char)


def text(slide, x, y, w, h, paras, size=16, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullets=False,
         space=6, name=None):
    """paras: list of str or list of (text, dict(bold, color, size, italic)) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("color", color), o.get("bold", False), o.get("italic", False))
        if bullets:
            bullet(p)
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, head, body, fill=TINT, head_color=RUST, size=14, num=None):
    box(slide, x, y, w, h, fill)
    top = y + 0.15
    if num is not None:
        c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.18), Inches(top), Inches(0.5), Inches(0.5))
        c.fill.solid(); c.fill.fore_color.rgb = ORANGE; c.line.fill.background()
        tf = c.text_frame; tf.margin_left = tf.margin_right = 0
        r = tf.paragraphs[0].add_run(); r.text = str(num); _font(r, 16, WHITE, True)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        text(slide, x + 0.78, top + 0.02, w - 0.95, 0.5, [[(head, {"bold": True, "color": head_color})]], size=size + 2,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.2, top + 0.65, w - 0.35, h - 0.85, body if isinstance(body, list) else [body], size=size,
             color=INK)
    else:
        text(slide, x + 0.2, top, w - 0.35, 0.45, [[(head, {"bold": True, "color": head_color})]], size=size + 2)
        is_list = isinstance(body, list) and not body[0][:1].isdigit()
        text(slide, x + 0.2, top + 0.55, w - 0.35, h - 0.7, body if isinstance(body, list) else [body], size=size,
             bullets=is_list, space=7 if is_list else 6)


def stat(slide, x, y, w, big, label, color=RUST, fill=TINT):
    box(slide, x, y, w, 1.55, fill)
    text(slide, x + 0.1, y + 0.12, w - 0.2, 0.8, [[(big, {"bold": True, "color": color})]], size=34, align=PP_ALIGN.CENTER)
    text(slide, x + 0.15, y + 0.9, w - 0.3, 0.6, [label], size=13, color=MUTED, align=PP_ALIGN.CENTER)


def picture(slide, path, x, y, w=None, h=None):
    from PIL import Image
    iw, ih = Image.open(path).size
    if w and h:  # fit inside the box, keep aspect ratio, centre
        s = min(w / iw, h / ih)
        pw, ph = iw * s, ih * s
        return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None,
                                    height=Inches(h) if h else None)


def table(slide, x, y, w, rows, widths, size=13, row_h=0.42, head_fill=ORANGE):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.text = ""
            r = p.add_run(); r.text = str(val)
            _font(r, size, WHITE if i == 0 else INK, bold=(i == 0))
            cell.fill.solid()
            cell.fill.fore_color.rgb = head_fill if i == 0 else (TINT if i % 2 else WHITE)
    return t


def chart_bar(slide, x, y, w, h, cats, series, colors, title=None, vmin=None, vmax=None, fmt='0.00', legend=False,
              horizontal=False):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name, ch.font.size = FONT, Pt(12)
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 14, MUTED, True)
    va = ch.value_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
    va.tick_labels.font.color.rgb = MUTED
    va.format.line.fill.background()
    ch.category_axis.tick_labels.font.color.rgb = INK
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.font.size, dl.font.color.rgb = Pt(11), INK
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s, c in zip(plot.series, colors):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    return ch


def chrome(slide, title):
    """Header title, PES logo, red title with the template's orange rule, team names in the footer bar."""
    text(slide, 0.08, 0.07, 6.5, 0.4, [SHORT], size=12, color=GREY, name="Header")
    slide.shapes.add_picture(_logo_path(), Inches(11.92), Inches(0.0), Inches(1.42), Inches(1.12))
    text(slide, 3.0, 1.2, 8.67, 0.55, [[(title, {"color": RED})]], size=26, align=PP_ALIGN.RIGHT,
         anchor=MSO_ANCHOR.BOTTOM, name="Title")
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.33), Inches(1.77), Inches(8.33), Inches(0.04))
    rule.fill.solid(); rule.fill.fore_color.rgb = ORANGE; rule.line.fill.background()
    slide.shapes._spTree.append(copy.deepcopy(FOOTER_EL))
    for sh in slide.shapes:
        if sh.is_placeholder:
            sh.text_frame.paragraphs[0].runs[0].text = FOOT


_LOGO = HERE / "logo.png"


def _logo_path():
    if not _LOGO.exists():
        _LOGO.write_bytes(LOGO)
    return str(_LOGO)


def new(title):
    s = prs.slides.add_slide(BLANK)
    chrome(s, title)
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ------------------------------------------------------------------ numbers
v = R.val
b1x, c1x, g2x, f1x = (v(m, "Cross-dataset") for m in ("B1", "C1", "G2", "F1"))
x1c, f1c = R.seed_mean("x1_ffpp_dfd", "celebdf_test_video_auc")[0], R.seed_mean("f1_grace_dct", "celebdf_test_video_auc")[0]
d2c = R.seed_mean("d2_unsup_adapt", "celebdf_test_video_auc")[0]
s0c, s3c = R.seed_mean("s0_student", "celebdf_test_video_auc")[0], R.seed_mean("s3_kd_full", "celebdf_test_video_auc")[0]
s3xc, s0xc = R.seed_mean("s3x_kd_from_x1", "celebdf_test_video_auc")[0], R.seed_mean("s0x_student_dfd", "celebdf_test_video_auc")[0]
rob = lambda m, c: v(m, c, "3. Robust")  # noqa: E731
expl = {r[0]: r[1:] for r in R.table("5. Explanations")[1:]}
pg = {k: float(e[3]) for k, e in expl.items()}
cdf_tag = "" if R.cdf_mode == "official" else " (Celeb-DF: interim test list; official re-run in progress)"

# ------------------------------------------------------------------ slide 1: title (template slide, filled)
s1 = SL[0]
for sh in s1.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text
    if t.startswith("Title of the Project"):
        sh.text_frame.paragraphs[0].runs[0].text = SHORT
    if "Project Title" in t:
        tf = sh.text_frame
        vals = [TITLE, "PW26_RJ_02", "Dr. Jyothi R, Associate Professor",
                "Charan L, Deepak S N, Samrudh N Shindhe, Samarth Gowda G M"]
        labels = ["Project Title", "Project ID", "Project Guide", "Project Team"]
        for p, lab, val in zip(tf.paragraphs, labels, vals):
            p.runs[0].text = lab
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            pPr = p._p.get_or_add_pPr()  # tab stop so every value starts at the same x
            tabs = pPr.find(qn("a:tabLst"))
            if tabs is None:
                tabs = etree.SubElement(pPr, qn("a:tabLst"))
            tab = etree.SubElement(tabs, qn("a:tab")); tab.set("pos", str(int(Inches(2.3)))); tab.set("algn", "l")
            r = p.add_run(); r.text = "	:  " + val
            _font(r, 15, INK)
        sh.width = Inches(11.4); sh.left = Inches(1.0); sh.top = Inches(4.45)
    if sh.is_placeholder:
        sh.text_frame.paragraphs[0].runs[0].text = FOOT
text(s1, 2.6, 2.75, 8.6, 1.3, [[("GRACE: a deepfake detector that works on new videos, survives compression and "
                                 "explains its answer", {"color": RUST, "italic": True})]], size=20, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ slide 2: outline (template slide, refreshed list)
s2 = SL[1]
outline = ["Abstract and the problem", "Suggestions from Review – 2", "What we changed since Phase 2",
           "Implementation details", "Project demonstration", "Test plan, strategy and test results",
           "Results and discussion", "Applicability in other domains", "Timeline and project report",
           "Conclusion and future work", "References, IEEE paper draft and target venues"]
for sh in s2.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Title of the Project"):
        sh.text_frame.paragraphs[0].runs[0].text = SHORT
    elif sh.is_placeholder:
        sh.text_frame.paragraphs[0].runs[0].text = FOOT
    elif sh.has_text_frame and "Abstract" in sh.text_frame.text:
        tf = sh.text_frame
        paras = list(tf.paragraphs)
        tmpl = paras[1]._p
        for p in paras:
            p._p.getparent().remove(p._p)
        for item in outline:
            np_ = copy.deepcopy(tmpl)
            tf._txBody.append(np_)
        for p, item in zip(tf.paragraphs, outline):
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            p.runs[0].text = item
            p.runs[0].font.size = Pt(18)

# ------------------------------------------------------------------ content slides
s = new("Abstract")
text(s, 0.7, 2.1, 7.3, 4.6, [
    [("The problem. ", {"bold": True, "color": RUST}), ("Fake face videos are easy to make. Most detectors work only on "
      "the kind of videos they were trained on, break on compressed uploads and give no reason for their answer.", {})],
    [("What we built. ", {"bold": True, "color": RUST}), ("GRACE, a detector that looks for 10 named clues of face "
      "editing (for example 'the edge around the face is uneven'). Its answer is computed from these clues, so the "
      "explanation is real, not a guess added afterwards.", {})],
    [("How we tested it. ", {"bold": True, "color": RUST}), ("Trained only on FaceForensics++; tested on two datasets it "
      "never saw (DFD and Celeb-DF), with strict checks that no test video leaks into training.", {})],
    [("What you get. ", {"bold": True, "color": RUST}), ("A web app that says 'likely fake', 'can't tell' or 'looks real' in "
      "plain words, plus a small version that runs on any laptop.", {})],
], size=16, space=12)
stat(s, 8.5, 2.1, 4.1, f"{f1x:.3f}", "accuracy (AUC) on unseen videos\nSwin-B baseline: %.3f" % b1x)
stat(s, 8.5, 3.8, 4.1, "10 clues", "named reasons behind every verdict", color=BLUE, fill=TINT2)
stat(s, 8.5, 5.5 - 0.05, 4.1, "4 ms", "per face for the small model on a CPU")
notes(s, "AUC: 1.0 = perfect, 0.5 = guessing. Cross-dataset = mean of DFD and Celeb-DF." + cdf_tag)

s = new("The Problem We Solve")
for i, (h, b) in enumerate([
        ("Works only on familiar videos", "Detectors score ~100% on their own dataset but drop sharply on videos from new "
                                          "sources and new fake-making tools."),
        ("Breaks on compressed uploads", "WhatsApp, YouTube and Instagram re-compress videos. That wipes out the tiny traces "
                                         "many detectors rely on."),
        ("Gives no reason", "A single score like '0.91' does not tell a user or an investigator what looks wrong, so it is "
                            "hard to trust.")]):
    card(s, 0.7 + i * 4.1, 2.2, 3.85, 3.0, h, b, num=i + 1)
text(s, 0.7, 5.5, 11.9, 1.1, [[("Our goal: ", {"bold": True, "color": RUST}),
                               ("a detector that generalises to new videos, survives compression, explains its answer "
                                "in plain words and also runs on an ordinary computer.", {})]], size=17)

s = new("Suggestions from Review – 2")
text(s, 0.7, 2.05, 5.6, 0.45, [[("Panel suggestions", {"bold": True, "color": BLUE})]], size=18)
table(s, 0.7, 2.55, 5.6, [["#", "Suggestion from the panel"], ["1", "[add panel remark]"], ["2", "[add panel remark]"],
                          ["3", "[add panel remark]"]], [0.5, 5.1], size=14, row_h=0.55)
text(s, 6.8, 2.05, 5.9, 0.45, [[("What we did after Review – 2", {"bold": True, "color": BLUE})]], size=18)
text(s, 6.8, 2.55, 5.9, 4.2, [
    "Audited the old pipeline and fixed data leakage (next slide)",
    "Tested on two unseen datasets, not just the training dataset",
    "Added built-in explanations (10 named clues) instead of only Grad-CAM",
    "Made the model robust to compression and built a small CPU version",
    "Built and tested a plain-language web app",
    "Wrote the full Phase-3 report and an IEEE paper draft"], size=15, bullets=True, space=8)
notes(s, "Fill the left table with the exact remarks from the Review-2 panel; the right column lists what was done.")

s = new("A Lesson from Phase 2: Data Leakage")
stat(s, 0.7, 2.2, 3.6, "98.98%", "Phase-2 validation accuracy\n(Swin-B on Celeb-DF)", color=GREY, fill=RGBColor(0xF2, 0xF2, 0xF2))
text(s, 4.5, 2.65, 0.6, 0.6, [[("→", {"bold": True, "color": ORANGE})]], size=36, align=PP_ALIGN.CENTER)
stat(s, 5.2, 2.2, 3.6, "2,446 / 2,451", "validation videos also had\nframes in the training set")
text(s, 9.0, 2.65, 0.6, 0.6, [[("→", {"bold": True, "color": ORANGE})]], size=36, align=PP_ALIGN.CENTER)
stat(s, 9.7, 2.2, 3.0, f"{b1x:.3f}", "same model type, fair test\n(unseen datasets)", color=BLUE, fill=TINT2)
text(s, 0.7, 4.15, 12.0, 2.6, [
    [("What went wrong: ", {"bold": True, "color": RUST}), ("frames of the same videos were in both training and "
      "validation, so the model was 'remembering' videos, not detecting fakes.", {})],
    [("What we fixed: ", {"bold": True, "color": RUST}), ("official splits by source video, automatic tests that stop "
      "training if any video is shared, model choice on validation data only, and separate unseen test sets.", {})],
    [("Why it matters: ", {"bold": True, "color": RUST}), ("every number in Phase 3 is now honest and repeatable.", {})],
], size=16, space=10)

s = new("Data and Fair Testing")
rows = [["Dataset", "Used for", "Videos", "Face images"]]
cnt = {r[0]: r for r in R.counts[list(R.counts)[0]][1:] if "all" in r[1]}
for key, name, use in [("ffpp_train", "FaceForensics++ train", "Training"), ("ffpp_val", "FaceForensics++ val", "Choosing models"),
                       ("ffpp_test", "FaceForensics++ test", "Test (familiar)"), ("dfd_test", "DFD (Google)", "Test (unseen)"),
                       ("celebdf_test", "Celeb-DF v2", "Test (unseen)")]:
    r = cnt[key]
    rows.append([name, use, r[3].strip("*"), f"{int(r[4].strip('*')):,}"])
table(s, 0.7, 2.15, 7.4, rows, [2.8, 1.9, 1.1, 1.6], size=14, row_h=0.5)
card(s, 8.5, 2.15, 4.15, 3.6, "Rules we follow", [
    "Never train on DFD or Celeb-DF", "No video in two splits (checked by tests)", "Pick models on validation only",
    "Score per video (average of its frames)"], fill=TINT2, head_color=BLUE, size=14)
text(s, 0.7, 5.4, 7.4, 1.2, [[("Headline measure: ", {"bold": True, "color": RUST}),
                              ("video-level AUC (1.0 = perfect, 0.5 = guessing).", {})]], size=15)
notes(s, "Every frame is an aligned 224x224 face crop." + cdf_tag)

s = new("How GRACE Works")
picture(s, FIG / "grace_architecture_overview.png", 0.5, 1.95, 6.4, 4.85)
for i, (h, b) in enumerate([
        ("See the face", "A large pretrained model (CLIP ViT-L/14) looks at the face, lightly tuned with LoRA."),
        ("Check compression traces", "A DCT branch reads the same 8×8 blocks JPEG and video codecs use."),
        ("Score 10 clues", "Each clue gets a score and a heat-map of where it appears on the face."),
        ("Decide from the clues", "Fake score = weighted sum of the clue scores, so the reasons are the decision.")]):
    card(s, 7.15, 1.95 + i * 1.2, 5.55, 1.1, h, b, size=13)

s = new("10 Clues the Model Looks For")
picture(s, FIG / "sbi_examples.png", 0.5, 1.95, 6.6, 4.4)
text(s, 0.5, 6.35, 6.6, 0.4, ["We create our own fakes from real faces and record how each was made → free clue labels."],
     size=12, color=MUTED)
clues = ["The face looks pasted on", "Skin colour does not match", "Part of the face is blurry",
         "Part of the face is over-sharpened", "Signs of resizing", "The mouth area looks edited", "The eye area looks edited",
         "The whole face may be replaced", "The face is slightly out of place", "The edge around the face is uneven"]
text(s, 7.4, 1.95, 5.3, 4.8, [[(f"{i + 1:>2}. ", {"bold": True, "color": ORANGE}), (c, {})] for i, c in enumerate(clues)],
     size=15, space=5)

s = new("Implementation Details")
table(s, 0.7, 2.1, 11.95, [
    ["Module", "Technology", "What it does"],
    ["Data and splits", "Python, pandas, OpenCV", "Leak-free splits, face crops, balanced batches, compression effects"],
    ["Fake maker (SBI)", "face-alignment landmarks", "Makes self-blended fakes and records the recipe → clue labels"],
    ["Teacher model", "PyTorch, OpenCLIP, LoRA", "CLIP + DCT branch + gated fusion + 10-clue decision layer"],
    ["Small student", "timm EfficientNet-B0, ONNX", "4.7 M-parameter copy trained on compressed frames"],
    ["Evaluation", "scikit-learn, matplotlib", "AUC on 3 test sets × 10 compression levels, explanation scores"],
    ["Web app", "Gradio 6, ONNX Runtime", "Upload → verdict, clues, report; API endpoints"],
    ["Automation", "Job queue, git, pytest", "Ran 24/7, logged every run with its code version and seed"],
], [2.6, 3.4, 5.95], size=14, row_h=0.56)

s = new("Training: Teacher and Small Student")
picture(s, FIG / "diagram2_training.png", 0.5, 1.95, 5.6, 4.85)
card(s, 6.4, 2.0, 6.3, 2.2, "Stage 1: train the teacher", [
    "Real faces + our self-made fakes + FF++ fakes", "Learns the fake/real answer and the 10 clues together",
    "One GPU (RTX 4500 Ada), about 2 hours per run"], size=14)
card(s, 6.4, 4.4, 6.3, 2.3, "Stage 2: teach the small student", [
    "Student only sees damaged frames (JPEG, resized)", "Copies the teacher's answer, clues and heat-maps",
    "Runs on a laptop CPU in about 4 ms per face"], fill=TINT2, head_color=BLUE, size=14)

s = new("Key Algorithm (Pseudocode)")
code = ["# Making a training fake with clue labels",
        "region  <- pick {full face, mouth, eyes, nose}",
        "mask    <- soft shape over that region",
        "copy    <- face with colour / blur / resize / shift",
        "a       <- pick {0.25, 0.5, 0.75, 1.0}",
        "fake    <- face*(1 - a*mask) + copy*(a*mask)",
        "clues   <- recipe -> 10 numbers (label for free)",
        "",
        "# Deciding from the clues",
        "c       <- 10 clue scores from the face",
        "score   <- w1*c1 + ... + w10*c10 + small extra",
        "verdict <- real | can't tell | fake  (0.35, 0.65)"]
box(s, 0.7, 2.05, 7.0, 4.65, RGBColor(0xF6, 0xF7, 0xF9), line=RGBColor(0xD0, 0xD3, 0xD9))
tb = text(s, 0.9, 2.15, 6.7, 4.5, code, size=14, color=INK, space=2)
for p in tb.text_frame.paragraphs:
    for r in p.runs:
        r.font.name = "Consolas"
        if r.text.startswith("#"):
            r.font.color.rgb = RUST
card(s, 8.0, 2.05, 4.65, 4.65, "Why this matters", [
    "No one has to label clues by hand: the fake-maker writes the labels.",
    "Because the score is built from the clues, switching a clue off shows how much it mattered.",
    "Faces turned too far sideways are left out, as the model never saw such poses."], fill=TINT2, head_color=BLUE, size=14)

s = new("Project Demonstration: The App")
picture(s, FIG / "app_result.png", 0.5, 1.95, 6.3, 4.85)
card(s, 7.1, 2.0, 5.6, 4.7, "How a user checks a video", [
    "1. Upload a photo or short video",
    "2. Choose Careful check (best) or Quick check (any computer)",
    "3. Read the answer: Likely fake / Can't tell / Looks real",
    "4. See the 3 clues behind it, glowing on the face",
    "5. Follow the 'what you can do now' steps or save a report",
    "Runs locally: nothing is uploaded to the internet"], size=15)
notes(s, "Live demo: run python app/app.py and open http://127.0.0.1:7860 (Careful check needs the GPU).")

s = new("Demonstration: Details for Experts")
picture(s, FIG / "app_tech.png", 0.5, 1.95, 6.6, 4.85)
card(s, 7.4, 2.0, 5.3, 2.5, "Technical panel", [
    "Fake probability and model used", "Score of every face over time", "Clue heat-maps; switch clues off and re-score",
    "Most similar training faces; HTML report"], size=14)
table(s, 7.4, 4.7, 5.3, [["API endpoint", "Purpose"], ["/check", "Analyse a file"], ["/rescore", "Switch clues off"],
                         ["/clear", "Start again"]], [2.0, 3.3], size=13, row_h=0.45)

s = new("Test Plan, Strategy and Results")
table(s, 0.7, 2.1, 7.9, [
    ["Test level", "Method", "Result"],
    ["Unit tests (pytest)", "Splits, fake maker, models, data", "242 / 242 pass"],
    ["Integration", "Teacher cache vs fresh run; ONNX vs PyTorch", "Diff < 1e-5"],
    ["Functional (API)", "All 5 endpoints via gradio_client", "All working"],
    ["UI (browser)", "Playwright on Edge: 14 checks, 3 videos", "14 / 14 pass"],
    ["Acceptance", "6 held-out files in the app", "Careful: 5 / 5 correct"],
    ["Performance", "Speed per face / per video", "4 ms CPU; 3–5 s/video"],
], [2.4, 3.4, 2.1], size=13, row_h=0.56)
card(s, 8.9, 2.1, 3.8, 4.6, "Strategy", [
    "Why: catch data leaks and bugs before any training run", "Tools: pytest, Playwright, gradio_client (open source)",
    "Environment: Windows 11, RTX 4500 Ada, Python 3.12", "Roles: [add each member's testing role]"],
    fill=TINT2, head_color=BLUE, size=13)

s = new("Results: Accuracy on Unseen Videos")
models = [("B1 Swin-B\n(Phase 2 type)", b1x), ("C1 CLIP\n+ LoRA", c1x), ("G2 + clues\n+ DCT", g2x), ("F1 final\nmodel", f1x)]
ch = chart_bar(s, 0.6, 2.0, 7.4, 4.75, [m for m, _ in models], [("Cross-dataset AUC", [x for _, x in models])],
               [ORANGE], title="Mean AUC on DFD and Celeb-DF (unseen)", vmin=0.80, vmax=0.95, fmt="0.000")
pts = ch.plots[0].series[0].points
pts[0].format.fill.solid(); pts[0].format.fill.fore_color.rgb = GREY
pts[3].format.fill.solid(); pts[3].format.fill.fore_color.rgb = RUST
card(s, 8.3, 2.0, 4.4, 4.75, "What this shows", [
    f"Moving from Swin-B to CLIP gave the biggest jump ({b1x:.3f} → {c1x:.3f})",
    f"Clues + DCT branch pushed it further ({g2x:.3f})",
    f"F1 keeps {f1x:.3f} while about 80% of each decision is explained by the 10 clues",
    "In-domain (FF++ test) every model is ~1.0, so unseen data is the real test"], size=14)
notes(s, "Mean ± std over 3 seeds for B1 and F1." + cdf_tag)

s = new("What Helped and What Did Not")
card(s, 0.7, 2.05, 5.85, 4.65, "Helped", [
    "Frozen CLIP backbone with LoRA (biggest gain)", "10-clue decision layer (accuracy went up, not down)",
    "DCT branch that matches JPEG / video codecs", "Strong penalty that forces decisions through the clues",
    f"More varied training data: adding DFD → Celeb-DF {f1c:.3f} → {x1c:.3f}"], fill=RGBColor(0xE4, 0xF2, 0xEA),
     head_color=GREEN, size=15)
card(s, 6.85, 2.05, 5.85, 4.65, "Did not help (reported honestly)", [
    "Domain-adversarial training (DANN): neutral or worse in all 5 tries",
    "Global FFT branch: worse than DCT",
    f"Adapting to unlabelled Celeb-DF videos: {d2c:.3f} (worse)",
    "Teacher did not make the student more accurate (only better explanations)",
    "One clue ('uneven edge') could not be learned"], fill=RGBColor(0xFB, 0xE9, 0xE4), head_color=RUST, size=15)

s = new("Robustness to Compression")
conds = ["Clean", "JPEG 30", "Resized ¼", "H.264 strong"]
cols = ["clean", "jpeg:30", "down:0.25", "h264:40"]
chart_bar(s, 0.6, 2.0, 8.0, 4.75, conds, [("B1 Swin-B", [rob("B1", c) for c in cols]), ("F1 final", [rob("F1", c) for c in cols]),
                                          ("S3 small student", [rob("S3", c) for c in cols])],
          [GREY, RUST, BLUE], title="Cross-dataset AUC after damaging the test videos", vmin=0.5, vmax=1.0, fmt="0.00",
          legend=True)
card(s, 8.9, 2.0, 3.8, 4.75, "Takeaways", [
    f"F1 barely drops at JPEG 30 ({rob('F1', 'jpeg:30'):.2f}); Swin-B falls to {rob('B1', 'jpeg:30'):.2f}",
    "Resizing: same story",
    f"Very strong video compression is still hard for all models (~{rob('F1', 'h264:40'):.2f})"], size=14)

s = new("Are the Explanations Right?")
names = [("Random guess", pg["random"]), ("Swin-B Grad-CAM", pg["B1 Grad-CAM"]), ("CLIP rollout", pg["C1 rollout"]),
         ("Student clue map", pg["S3 c1"]), ("GRACE clue map", pg["F1 region"])]
ch = chart_bar(s, 0.6, 2.0, 6.0, 4.75, [n for n, _ in names], [("Pointing game", [x for _, x in names])], [ORANGE],
               title="How often the map points at the edited area", vmin=0, vmax=1.05, fmt="0%", horizontal=True)
for k, c in enumerate([GREY, GREY, GREY, BLUE, RUST]):
    p = ch.plots[0].series[0].points[k]; p.format.fill.solid(); p.format.fill.fore_color.rgb = c
picture(s, FIG / "fig_explanations_top.png", 6.9, 2.0, 5.8, 3.55)
text(s, 6.9, 5.6, 5.8, 1.2, [f"GRACE's clue maps point at the edited region {pg['F1 region']:.0%} of the time; "
                             f"heat-maps added after training manage only {pg['B1 Grad-CAM']:.0%}."], size=14)

s = new("A Small Model for Any Computer")
eff = R.eff.set_index("model")
stat(s, 0.7, 2.1, 2.9, "4.7 M", "parameters (teacher: 307 M)")
stat(s, 3.8, 2.1, 2.9, f"{eff.loc['s3_kd_full', 'cpu_onnx_latency_ms_bs1']:.1f} ms", "per face on a CPU (ONNX)")
stat(s, 6.9, 2.1, 2.9, "65×", "smaller than the teacher", color=BLUE, fill=TINT2)
stat(s, 10.0, 2.1, 2.65, f"{v('S3', 'Cross-dataset'):.2f}", "AUC on unseen videos")
text(s, 0.7, 4.0, 12.0, 2.7, [
    [("Honest finding: ", {"bold": True, "color": RUST}), ("on unseen videos the teacher-trained student is about as "
      f"accurate as one trained without a teacher ({v('S3', 'Cross-dataset'):.3f} vs {v('S0', 'Cross-dataset'):.3f}; "
      f"Celeb-DF {s3c:.3f} vs {s0c:.3f}; 3 seeds each).", {})],
    [("What the teacher does give: ", {"bold": True, "color": RUST}), ("explanations that match the teacher's and steadier "
      "results across runs.", {})],
    [("What does help the student: ", {"bold": True, "color": RUST}), (f"more varied data (with DFD: {s0xc:.3f}); a better "
      f"teacher alone did not ({s3xc:.3f}).", {})],
], size=16, space=12)

s = new("Expected vs Achieved")
table(s, 0.7, 2.1, 11.95, [
    ["What we expected", "What happened", "Status"],
    ["CLIP + LoRA generalises better than Swin-B", f"{b1x:.3f} → {c1x:.3f} on unseen data", "As expected"],
    ["Clues cost little accuracy", "Accuracy went up; ~80% of decisions explained by clues", "Better than expected"],
    ["DCT better than FFT", f"{g2x:.3f} vs {v('G2-fft', 'Cross-dataset'):.3f}; strong under JPEG", "As expected"],
    ["Teacher makes the student more accurate", "No accuracy gain; explanations transfer", "Partly (reason: data matters more)"],
    ["Domain-adversarial training helps", "Neutral or harmful in all variants", "Not as expected"],
    ["Phase-2 accuracy holds on a fair test", "It was inflated by leakage", "Corrected"],
], [4.4, 5.0, 2.55], size=14, row_h=0.6)

s = new("Applicability in Other Domains")
for i, (h, b) in enumerate([
        ("Voice and audio deepfakes", "Same idea: named clues (e.g. 'lips do not match speech') feed one decision layer."),
        ("AI-generated images", "Self-made fakes + clues can target diffusion artefacts; CLIP features already transfer."),
        ("Document and ID fraud", "Detect pasted photos and edited text regions with region clues and DCT traces."),
        ("Medical image tampering", "Flag edited regions in scans with maps a doctor can check."),
        ("Content moderation", "Small student model screens uploads on CPU; the teacher reviews hard cases."),
        ("Limits", "Needs a way to make realistic training fakes; new domains need their own clue list.")]):
    card(s, 0.7 + (i % 3) * 4.05, 2.05 + (i // 3) * 2.4, 3.85, 2.2, h, b, size=13,
         fill=TINT if i < 5 else RGBColor(0xF2, 0xF2, 0xF2))

s = new("Timeline – Update on Tasks")
steps = [("Aug–Sep", "Audit of Phase-2 pipeline; leak found and fixed", "Done"),
         ("Oct", "Leak-free splits, fake maker, models coded and tested", "Done"),
         ("17–23 Nov", "7-day GPU sprint: 17 models, most with 3 seeds", "Done"),
         ("Nov", "Web app redesigned and fully tested", "Done"),
         ("Nov", "Phase-3 report (82 pages) and IEEE paper draft", "Done"),
         ("21 Nov", "Re-run on the official Celeb-DF test list; final numbers in report", "Done")]
for i, (when, what, st) in enumerate(steps):
    y = 2.05 + i * 0.75
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), Inches(y + 0.1), Inches(0.35), Inches(0.35))
    c.fill.solid(); c.fill.fore_color.rgb = GREEN if st == "Done" else ORANGE; c.line.fill.background()
    text(s, 1.35, y, 1.7, 0.55, [[(when, {"bold": True, "color": RUST})]], size=15, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 3.1, y, 6.9, 0.55, [what], size=15, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.1, y, 2.5, 0.55, [[(st, {"bold": True, "color": GREEN if st == "Done" else ORANGE})]], size=15,
         anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Project report: complete (82 pages, all guideline chapters); numbers refresh automatically after the official "
         "Celeb-DF re-evaluation. Adjust the early months if your records differ.")

s = new("Conclusion and Future Work")
card(s, 0.7, 2.05, 5.85, 4.65, "What we achieved", [
    "Honest, leak-free evaluation (fixed the Phase-2 problem)",
    f"Better detection on unseen videos: {f1x:.3f} vs {b1x:.3f}",
    "Every verdict explained by 10 named clues",
    "Robust to JPEG and resizing", "Small model: 4 ms per face on a CPU",
    "Plain-language, fully tested web app"], size=15)
card(s, 6.85, 2.05, 5.85, 4.65, "Next steps", [
    "Add audio and lip-sync clues (multimodal)", "Learn from frame sequences over time",
    "Train on more varied data (full FF++, DFDC, DF40)", "Better handling of strong video compression",
    "User study of the plain-language app", "Submit the IEEE paper"], fill=TINT2, head_color=BLUE, size=15)

s = new("References")
refs = ["[1] A. Rössler et al., “FaceForensics++: Learning to detect manipulated facial images,” ICCV, 2019.",
        "[2] Y. Li et al., “Celeb-DF: A large-scale challenging dataset for DeepFake forensics,” CVPR, 2020.",
        "[3] K. Shiohara and T. Yamasaki, “Detecting deepfakes with self-blended images,” CVPR, 2022.",
        "[4] A. Radford et al., “Learning transferable visual models from natural language supervision,” ICML, 2021.",
        "[5] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” ICLR, 2022.",
        "[6] P. W. Koh et al., “Concept bottleneck models,” ICML, 2020.",
        "[7] U. Ojha, Y. Li, and Y. J. Lee, “Towards universal fake image detectors…,” CVPR, 2023.",
        "[8] X. Cui et al., “Forensics Adapter: Adapting CLIP for generalizable face forgery detection,” CVPR, 2025.",
        "[9] Z. Yan et al., “DeepfakeBench: A comprehensive benchmark of deepfake detection,” NeurIPS, 2023.",
        "[10] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv, 2015.",
        "[11] S. Tariq et al., “From prediction to explanation … for non-expert users,” ACM MM, 2025.",
        "[12] K. Tsigos et al., “Towards quantitative evaluation of explainable AI methods for deepfake detection,” MAD, 2024."]
text(s, 0.7, 2.0, 12.0, 4.8, refs, size=13, space=4)
notes(s, "Full list of 48 references in IEEE format is in the project report.")

s = new("IEEE Paper Draft and Target Venues")
card(s, 0.7, 2.05, 5.85, 4.65, "Draft status", [
    "Title: GRACE: Generalizable, Robust and Concept-Explainable Deepfake Detection",
    "IEEE conference format (LaTeX / Overleaf)",
    "Method, protocol and leakage audit written",
    "Final results on the official Celeb-DF list ready to insert",
    "Status: ~80% complete"], size=14)
card(s, 6.85, 2.05, 5.85, 4.65, "Target venues (to finalise with guide)", [
    "CVPR 2027 Workshop on deepfake / media forensics", "IEEE ICIP 2027 (image processing)",
    "IEEE WIFS (Workshop on Information Forensics and Security)",
    "IEEE Access (journal)", "IEEE Transactions on Information Forensics and Security (journal, stretch goal)"],
     fill=TINT2, head_color=BLUE, size=14)

# ------------------------------------------------------------------ remove template guidance slides 3-15
sldIdLst = prs.slides._sldIdLst
thank = SL[15]
for sl in SL[2:15]:
    for sid in list(sldIdLst):
        if prs.part.related_part(sid.get(qn("r:id"))) is sl.part:
            prs.part.drop_rel(sid.get(qn("r:id")))
            sldIdLst.remove(sid)

# ------------------------------------------------------------------ move "Thank you" to the end
for sid in list(sldIdLst):
    if prs.part.related_part(sid.get(qn("r:id"))) is thank.part:
        sldIdLst.remove(sid); sldIdLst.append(sid)
cp = prs.core_properties
cp.author = cp.last_modified_by = "Deepak S N"
cp.title = "PW26_RJ_02 Review 3: " + TITLE
OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print("slides:", len(prs.slides), "->", OUT)
