"""Build the Phase-3 report: fill the department templates (pages 1-8), generate chapters 1-11 + back matter,
assemble one .docx, let Microsoft Word update the TOC / lists / page numbers, and export a PDF.

Run:  python "ESA Report/build/build_report.py"      (re-run after new results: every number is read from results/)
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docxcompose.composer import Composer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backmatter import appendix_a, appendix_b, references  # noqa: E402
from ch01_03 import abstract, ch1, ch2, ch3  # noqa: E402
from ch04_07 import ch4, ch5, ch6, ch7  # noqa: E402
from ch08_09 import ch8, ch9  # noqa: E402
from ch10_11 import ch10, ch11  # noqa: E402
from engine import A4_H, A4_W, FONT, Report, add_field  # noqa: E402

ESA = HERE.parent
TPL = ESA / "Report initial Pages(Pg 1-10)" / "Report initial Pages(Pg 1-10)"
OUT = ESA / "output"
OUT.mkdir(exist_ok=True)
NAME = "PW26_RJ_02_Phase3_Report"

TITLE = "Generalized Deepfake Detection Framework with Scalable Multimodal Integration"
GUIDE, DESIG = "Dr. Jyothi R", "Associate Professor"
TEAM = [("Charan L", "PES1UG23CS159"), ("Deepak S N", "PES1UG23CS178"),
        ("Samrudh N Shindhe", "PES1UG23CS508"), ("Samarth Gowda G M", "PES1UG23CS502")]


def set_text(p, new):
    """Replace a paragraph's text, keeping the formatting of its first run."""
    runs = p.runs
    if not runs:
        p.add_run(new)
        return
    runs[0].text = new
    for r in runs[1:]:
        r.text = ""


def replace_in(p, old, new):
    if old in p.text:
        set_text(p, p.text.replace(old, new))
        return True
    return False


def all_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


def freeze_spacing(doc):
    """Front templates rely on their own Normal style; make spacing explicit so the master style cannot change it."""
    pf = doc.styles["Normal"].paragraph_format
    ls, sa, sb = pf.line_spacing or 1.0, pf.space_after or Pt(0), pf.space_before or Pt(0)
    for p in all_paragraphs(doc):
        f = p.paragraph_format
        if f.line_spacing is None:
            f.line_spacing = ls
        if f.space_after is None:
            f.space_after = sa
        if f.space_before is None:
            f.space_before = sb
        if p.alignment is None:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            if r.font.name is None:
                r.font.name = FONT


def page_break_before(p):
    p.paragraph_format.page_break_before = True


def _is_blank(el):
    """An empty paragraph: no text, no picture, no field, no break."""
    if el.tag != qn("w:p"):
        return False
    if "".join(el.itertext()).strip():
        return False
    return not (el.xpath(".//w:drawing") or el.xpath(".//w:fldChar") or el.xpath(".//w:br") or el.xpath(".//w:sectPr"))


def drop_trailing_blanks(doc, heads):
    """The template ends each part with runs of empty lines; with the next heading starting a new page they spill
    onto an extra, blank page. Remove empty paragraphs right before such a heading and at the end of the body."""
    body = doc.element.body
    head_els = {p._p for p in doc.paragraphs if p.text.strip() in heads}
    kids = list(body)
    for i, el in enumerate(kids):
        if el in head_els:
            j = i - 1
            while j >= 0 and _is_blank(kids[j]):
                body.remove(kids[j])
                j -= 1
    kids = [k for k in body if k.tag != qn("w:sectPr")]
    for el in reversed(kids):
        if not _is_blank(el):
            break
        body.remove(el)


def fill_cover():
    doc = Document(TPL / "Page 1-2.docx")
    for p in all_paragraphs(doc):
        replace_in(p, "Title of the project", TITLE)
        for i, (name, srn) in enumerate(TEAM, 1):
            if p.text.strip() == f"Name {i}":
                set_text(p, name)
            replace_in(p, f"<SRN {i}>", srn)
        replace_in(p, "Prof. Guide Name", GUIDE)
        replace_in(p, "<Name of the Guide>", GUIDE)
        if p.text.strip() == "Designation":
            set_text(p, DESIG)
    # the certificate starts on its own page: break right after the cover's address line (the paragraph that
    # anchors the cover's rule drawing), so the cover keeps all its lines
    paras = doc.paragraphs
    anchor = next(i for i, p in enumerate(paras) if p._p.xpath(".//wp:anchor"))
    page_break_before(paras[anchor + 1])
    # floating tables make the following text wrap around them ("Ext / ernal Viva"); make them inline
    for el in doc.element.body.iter(qn("w:tblpPr")):
        el.getparent().remove(el)
    # long names must not wrap: widen the name column of the two name tables
    for t in doc.tables:
        if any("PES1UG23" in c.text for c in t.rows[0].cells) and len(t.columns) == 2:
            for row in t.rows:
                row.cells[0].width, row.cells[1].width = Cm(6.0), Cm(5.0)
            grid = t._tbl.tblGrid
            for gc, w in zip(grid.findall(qn("w:gridCol")), (Cm(6.0), Cm(5.0))):
                gc.set(qn("w:w"), str(int(w.twips)))
    freeze_spacing(doc)
    return doc


def field_paragraph(doc, instr):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    add_field(p, instr, "Right-click and choose Update Field to build this list.")
    return p


def fill_front():
    doc = Document(TPL / "Page 3-8.docx")
    for p in all_paragraphs(doc):
        replace_in(p, "Title of the project", TITLE)
        replace_in(p, "<Prof. Guide Name, Designation>", f"{GUIDE}, {DESIG}")
        replace_in(p, "Prof. <Guide Name>", f"{GUIDE}, {DESIG}")
        replace_in(p, "for his/her continuous", "for the continuous")
        replace_in(p, "Finally, this project could not have been completed",
                   "I thank my parents for their constant encouragement and support. Finally, this project could not "
                   "have been completed")
        for i, (name, srn) in enumerate(TEAM, 1):
            replace_in(p, f"<SRN {i}>", srn)
            replace_in(p, f"<Name {i}>", name)
        replace_in(p, "<Signature>", "")
    heads = {"DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", "LIST OF FIGURES", "LIST OF TABLES"}
    for p in doc.paragraphs:
        if p.text.strip() in heads:
            page_break_before(p)
    drop_trailing_blanks(doc, heads)
    # abstract text
    absp = next(p for p in doc.paragraphs if "Write the Abstract" in p.text)
    paras = abstract()
    words = sum(len(t.split()) for t in paras)
    assert words <= 150, f"abstract has {words} words (limit 150)"
    set_text(absp, paras[0])
    absp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for r in absp.runs:
        r.bold, r.font.size = False, Pt(12)
        r.font.highlight_color = None
        for h in r._r.findall(".//" + qn("w:highlight")) + r._r.findall(".//" + qn("w:shd")):
            h.getparent().remove(h)
    absp.paragraph_format.line_spacing = 1.5
    absp.paragraph_format.left_indent = absp.paragraph_format.right_indent = Cm(0)
    absp.paragraph_format.first_line_indent = Cm(0)
    absp.paragraph_format.space_after = Pt(10)
    for h in absp._p.findall(".//" + qn("w:shd")):
        h.getparent().remove(h)
    prev = absp
    for t in paras[1:]:
        new = copy.deepcopy(absp._p)
        prev._p.addnext(new)
        from docx.text.paragraph import Paragraph
        np_ = Paragraph(new, absp._parent)
        set_text(np_, t)
        prev = np_
    # replace the static TOC / figure / table tables with Word fields
    instrs = ['TOC \\h \\z \\t "GR Chapter,1,GR Section,2"', 'TOC \\h \\z \\t "GR Figure Caption,1"',
              'TOC \\h \\z \\t "GR Table Caption,1"']
    tables = doc.tables[1:4]
    for t, instr in zip(tables, instrs):
        fp = field_paragraph(doc, instr)
        t._tbl.addprevious(fp._p)
        t._tbl.getparent().remove(t._tbl)
    freeze_spacing(doc)
    # pages 3-8 in one typeface (the template mixes Arial into the acknowledgement)
    for p in all_paragraphs(doc):
        for r in p.runs:
            r.font.name = FONT
            rpr = r._r.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is not None:
                for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                    fonts.set(qn(a), FONT)
    return doc, words


def build_body():
    logo = HERE / "assets" / "hdr_image1.png"
    rep = Report(TITLE, logo)
    blocks = ch1() + ch2() + ch3() + ch4() + ch5() + ch6() + ch7() + ch8() + ch9() + ch10() + ch11() \
        + references() + appendix_a() + appendix_b()
    rep.render(blocks)
    return rep


def assemble():
    rep = build_body()
    body = rep.doc
    cover = fill_cover()
    front, words = fill_front()
    # insert the template pages in place of the placeholder paragraph of the front section
    ph = next(p for p in body.paragraphs if p.text == "FRONT_MATTER_PLACEHOLDER")
    idx = list(body.element.body).index(ph._p)
    comp = Composer(body)
    comp.insert(idx, front)
    comp.insert(idx, cover)
    doc = comp.doc
    ph = next(p for p in doc.paragraphs if p.text == "FRONT_MATTER_PLACEHOLDER")
    ph._p.getparent().remove(ph._p)
    cp = doc.core_properties
    cp.author = "Deepak S N"
    cp.title, cp.subject = TITLE, "UE23CS441A Capstone Project Phase-3 report, PW26_RJ_02"
    cp.comments = ""
    out = OUT / f"{NAME}.docx"
    doc.save(out)
    set_author(out)
    return out, rep, words


def set_author(docx, name="Deepak S N"):
    """Set the 'last modified by' field directly in docProps/core.xml (python-docx writes this field in the wrong
    namespace, which makes Word report the file as corrupted)."""
    import re
    import zipfile
    src = Path(docx)
    tmp = src.with_suffix(".tmp.docx")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                x = data.decode("utf-8")
                x = re.sub(r'<cp:lastModifiedBy xmlns:cp="[^"]*custom-properties">.*?</cp:lastModifiedBy>', "", x)
                x = re.sub(r"<cp:lastModifiedBy\s*/>|<cp:lastModifiedBy>.*?</cp:lastModifiedBy>",
                           f"<cp:lastModifiedBy>{name}</cp:lastModifiedBy>", x, count=1)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(src)


PS = r"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open('%DOCX%')
  foreach ($s in @(-20, -21)) { $st = $d.Styles.Item($s); $st.ParagraphFormat.SpaceAfter = 3 }
  $d.Repaginate()
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  $d.Repaginate()
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  $d.Save()
  $d.ExportAsFixedFormat('%PDF%', 17)
  "pages: " + $d.ComputeStatistics(2)
  $d.Close()
} finally { $w.Quit() }
"""


def word_update(docx):
    pdf = docx.with_suffix(".pdf")
    script = PS.replace("%DOCX%", str(docx)).replace("%PDF%", str(pdf))
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=600)
    print(r.stdout.strip(), r.stderr.strip()[:500])
    return pdf


if __name__ == "__main__":
    out, rep, words = assemble()
    print(f"written {out}  (abstract {words} words; {len(rep.figures)} figures, {len(rep.tables)} tables, "
          f"{len(rep.cite_order)} references)")
    if "--no-word" not in sys.argv:
        pdf = word_update(out)
        set_author(out)
        print("pdf", pdf, pdf.exists())
