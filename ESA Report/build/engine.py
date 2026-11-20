"""Document engine for the Phase-3 report: block list -> python-docx body that follows the PES guidelines.

Blocks (tuples):
  ("chapter", title)                       CHAPTER n: TITLE, new section, centred 18 pt bold, first page footer only
  ("unnumbered", title)                    REFERENCES / APPENDIX ... (new section, same heading style, no number)
  ("h2", title) / ("h3", title)            n.m Title (16 pt) / n.m.k Title (14 pt), left
  ("p", text)                              justified 12 pt, 1.5 spacing; **bold**, *italic*, [@key] citations, {fig:x} refs
  ("bul", [items]) / ("num", [items])      bullet / numbered list
  ("fig", label, path, caption, width_cm)  figure + "Figure n.k: caption" (style GR Figure Caption)
  ("tab", label, caption, header, rows, widths_cm or None)  "Table n.k: caption" above the table
  ("eq", label, text)                      centred equation with (n.k) at the right margin
  ("alg", label, title, lines)             pseudocode box, "Algorithm n.k: title"
  ("pb",)                                  page break
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
A4_W, A4_H, MARGIN = Cm(21.0), Cm(29.7), Cm(2.0)
TEXT_W_CM = 17.0


# ---------------------------------------------------------------- low-level helpers
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para_border(p, side="bottom", sz=6, color="000000"):
    pPr = p._p.get_or_add_pPr()
    pBdr = pPr.find(qn("w:pBdr"))
    if pBdr is None:
        pBdr = OxmlElement("w:pBdr")
        pPr.append(pBdr)
    el = OxmlElement(f"w:{side}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "1")
    el.set(qn("w:color"), color)
    pBdr.append(el)


def add_field(paragraph, instr, placeholder="", size=None, bold=False):
    """Complex field (PAGE, TOC ...). Word fills it in when fields are updated."""
    def r(tag, **attrs):
        run = paragraph.add_run()
        if size:
            run.font.size = size
        run.bold = bold
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        run._r.append(el)
        return run
    r("w:fldChar", **{"w:fldCharType": "begin", "w:dirty": "true"})
    run = paragraph.add_run()
    if size:
        run.font.size = size
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    run._r.append(it)
    r("w:fldChar", **{"w:fldCharType": "separate"})
    ph = paragraph.add_run(placeholder)
    if size:
        ph.font.size = size
    ph.bold = bold
    r("w:fldChar", **{"w:fldCharType": "end"})


def keep_with_next(p, on=True):
    p.paragraph_format.keep_with_next = on


def set_outline(style, level):
    pPr = style.element.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(level))
    pPr.append(ol)


# ---------------------------------------------------------------- styles
def setup_styles(doc):
    st = doc.styles
    normal = st["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def mk(name, size, align, before, after, outline=None, bold=True, italic=False, spacing=1.15):
        s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH) if name not in [x.name for x in st] else st[name]
        s.base_style = normal
        s.font.name, s.font.size, s.font.bold, s.font.italic = FONT, Pt(size), bold, italic
        s.font.color.rgb = RGBColor(0, 0, 0)
        p = s.paragraph_format
        p.alignment = align
        p.space_before, p.space_after = Pt(before), Pt(after)
        p.line_spacing = spacing
        p.keep_with_next = True
        if outline is not None:
            set_outline(s, outline)
        return s

    mk("GR Chapter", 18, WD_ALIGN_PARAGRAPH.CENTER, 0, 24, outline=0)
    mk("GR Section", 16, WD_ALIGN_PARAGRAPH.LEFT, 18, 8, outline=1)
    mk("GR Subsection", 14, WD_ALIGN_PARAGRAPH.LEFT, 12, 6, outline=2)
    cap = mk("GR Figure Caption", 11, WD_ALIGN_PARAGRAPH.CENTER, 4, 14, bold=False)
    cap.paragraph_format.keep_with_next = False
    mk("GR Table Caption", 11, WD_ALIGN_PARAGRAPH.CENTER, 10, 4, bold=False)
    mk("GR Algorithm Caption", 11, WD_ALIGN_PARAGRAPH.CENTER, 10, 4, bold=False)
    tbl = mk("GR Table Text", 10.5, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, bold=False, spacing=1.0)
    tbl.paragraph_format.keep_with_next = False
    code = mk("GR Code", 9.5, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, bold=False, spacing=1.0)
    code.font.name = "Consolas"
    code.paragraph_format.keep_with_next = False
    eq = mk("GR Equation", 12, WD_ALIGN_PARAGRAPH.CENTER, 6, 6, bold=False, italic=False, spacing=1.15)
    eq.paragraph_format.keep_with_next = False
    for name in ("List Bullet", "List Number"):
        s = st[name]
        s.font.name, s.font.size = FONT, Pt(12)
        s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        s.paragraph_format.space_after = Pt(2)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


# ---------------------------------------------------------------- inline text
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|_\{[^}]+\}|\^\{[^}]+\})")


def add_text(p, text, size=None, color=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("_{") or part.startswith("^{"):
            run = p.add_run(part[2:-1])
            if part[0] == "_":
                run.font.subscript = True
            else:
                run.font.superscript = True
            if size:
                run.font.size = size
            continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2]); run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = p.add_run(part[1:-1]); run.font.name = "Consolas"; run.font.size = Pt((size.pt if size else 12) - 1.5)
            continue
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = p.add_run(part[1:-1]); run.italic = True
        else:
            run = p.add_run(part)
        if size:
            run.font.size = size
        if color:
            run.font.color.rgb = color


# ---------------------------------------------------------------- the builder
class Report:
    def __init__(self, title_for_header, logo_path, period="August – Dec, 2026", dept="Dept. of CSE"):
        self.doc = Document()
        self.title = title_for_header
        self.logo = logo_path
        self.period, self.dept = period, dept
        setup_styles(self.doc)
        sec = self.doc.sections[0]
        self._page(sec)
        self.doc.add_paragraph("FRONT_MATTER_PLACEHOLDER")  # template front pages are inserted here
        self.chapter = 0
        self.counters = {}
        self.labels = {}
        self.cite_order = []
        self.first_section = True
        self.figures, self.tables = [], []

    # ---- numbering (pass 1)
    def number(self, blocks):
        ch, cnt = 0, {}
        for b in blocks:
            if b[0] == "chapter":
                ch += 1
                cnt = {"fig": 0, "tab": 0, "eq": 0, "alg": 0}
            elif b[0] == "unnumbered":
                ch = b[2] if len(b) > 2 and b[2] else ch  # appendix letter: Table A.1, Figure B.2
                cnt = {"fig": 0, "tab": 0, "eq": 0, "alg": 0}
            elif b[0] in ("fig", "tab", "eq", "alg"):
                cnt[b[0]] += 1
                self.labels[b[1]] = f"{ch}.{cnt[b[0]]}"
        # citation order = order of first appearance over all text
        for b in blocks:
            for s in _strings(b):
                for grp in re.findall(r"\[(@[^\]]+)\]", s):
                    for key in re.findall(r"@([\w\-]+)", grp):
                        if key not in self.cite_order:
                            self.cite_order.append(key)

    def resolve(self, text):
        def ref(m):
            kind, lab = m.group(1), m.group(2)
            if lab not in self.labels:
                raise KeyError(f"unknown label {kind}:{lab}")
            return self.labels[lab]
        text = re.sub(r"\{(fig|tab|eq|alg):([\w\-]+)\}", ref, text)

        def cite(m):
            keys = re.findall(r"@([\w\-]+)", m.group(1))
            return ", ".join(f"[{self.cite_order.index(k) + 1}]" for k in keys)
        return re.sub(r"\[(@[^\]]+)\]", cite, text)

    # ---- page setup, header, footer
    def _page(self, sec):
        sec.page_width, sec.page_height = A4_W, A4_H
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = MARGIN
        sec.header_distance = sec.footer_distance = Cm(1.0)

    def _header(self, hdr):
        hdr.is_linked_to_previous = False
        p = hdr.paragraphs[0]
        p.text = ""
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(self.title)
        r.font.size, r.italic = Pt(10), True
        p.add_run("\t")
        p.add_run().add_picture(str(self.logo), height=Cm(1.0))
        para_border(p, "bottom", 6)

    def _footer(self, ftr):
        ftr.is_linked_to_previous = False
        p = ftr.paragraphs[0]
        p.text = ""
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
        p.paragraph_format.line_spacing = 1.0
        para_border(p, "top", 6)
        for t in (self.dept, "\t", self.period, "\t"):
            r = p.add_run(t); r.font.size = Pt(10)
        add_field(p, "PAGE", "1", size=Pt(10))

    def new_section(self):
        # section 0 holds the front matter (no header/footer); every chapter starts a new section
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        for old in sec._sectPr.findall(qn("w:pgNumType")):  # add_section copies the previous section's settings
            sec._sectPr.remove(old)
        if self.first_section:
            self.first_section = False
            pg = OxmlElement("w:pgNumType"); pg.set(qn("w:start"), "1"); sec._sectPr.append(pg)  # page 1 = Chapter 1
        self._page(sec)
        sec.different_first_page_header_footer = True
        self._header(sec.header)
        self._footer(sec.footer)
        fp = sec.first_page_header
        fp.is_linked_to_previous = False
        fp.paragraphs[0].text = ""
        self._footer(sec.first_page_footer)
        return sec

    # ---- blocks
    def render(self, blocks):
        self.number(blocks)
        for b in blocks:
            getattr(self, "_" + b[0])(*b[1:])

    def _chapter(self, title):
        self.chapter += 1
        self.sec_n = self.sub_n = 0
        self.new_section()
        p = self.doc.add_paragraph(style="GR Chapter")
        add_text(p, f"CHAPTER {self.chapter}: {title.upper()}")

    def _unnumbered(self, title, letter=None):
        self.sec_n = self.sub_n = 0
        self.chapter_label = title
        self.new_section()
        p = self.doc.add_paragraph(style="GR Chapter")
        add_text(p, title.upper())
        self.unnumbered = True

    def _h2(self, title):
        self.sec_n += 1
        self.sub_n = 0
        p = self.doc.add_paragraph(style="GR Section")
        prefix = f"{self.chapter}.{self.sec_n} " if not getattr(self, "unnumbered", False) else ""
        add_text(p, prefix + title)

    def _h3(self, title):
        self.sub_n += 1
        p = self.doc.add_paragraph(style="GR Subsection")
        prefix = f"{self.chapter}.{self.sec_n}.{self.sub_n} " if not getattr(self, "unnumbered", False) else ""
        add_text(p, prefix + title)

    def _p(self, text):
        p = self.doc.add_paragraph()
        add_text(p, self.resolve(text))

    def _bul(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="List Bullet")
            add_text(p, self.resolve(it))

    def _num(self, items):
        for i, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.paragraph_format.space_after = Pt(2)
            add_text(p, f"{i}.  " + self.resolve(it))

    def _fig(self, label, path, caption, width_cm=15.5):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        c = self.doc.add_paragraph(style="GR Figure Caption")
        add_text(c, f"Figure {self.labels[label]}: " + self.resolve(caption))
        self.figures.append((self.labels[label], caption))

    def _tab(self, label, caption, header, rows, widths=None):
        c = self.doc.add_paragraph(style="GR Table Caption")
        add_text(c, f"Table {self.labels[label]}: " + self.resolve(caption))
        ncol = len(header)
        widths = widths or [TEXT_W_CM / ncol] * ncol
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths[j])
                cp = cell.paragraphs[0]
                cp.style = self.doc.styles["GR Table Text"]
                add_text(cp, self.resolve(str(val)), size=Pt(10.5))
                if i == 0:
                    for r in cp.runs:
                        r.bold = True
                    set_cell_shading(cell, "E7E9EE")
                if i < len(rows):  # keep rows together with the caption where possible
                    cp.paragraph_format.keep_with_next = True
        # repeat header row on page breaks
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 1.0
        self.tables.append((self.labels[label], caption))

    def _eq(self, label, text):
        p = self.doc.add_paragraph(style="GR Equation")
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        lines = [s.strip() for s in text.split("||")]  # '||' wraps a long equation; the number goes on the last line
        for k, line in enumerate(lines):
            p.add_run("\t")
            start = len(p.runs)
            add_text(p, line)
            for r in p.runs[start:]:
                r.font.name = "Cambria Math"
            if k < len(lines) - 1:
                p.add_run().add_break()
        p.add_run(f"\t({self.labels[label]})")

    def _alg(self, label, title, lines):
        c = self.doc.add_paragraph(style="GR Algorithm Caption")
        add_text(c, f"Algorithm {self.labels[label]}: {self.resolve(title)}")
        t = self.doc.add_table(rows=1, cols=1)
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.cell(0, 0)
        cell.width = Cm(TEXT_W_CM)
        set_cell_shading(cell, "F6F7F9")
        first = True
        for ln in lines:
            cp = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            cp.style = self.doc.styles["GR Code"]
            cp.paragraph_format.keep_with_next = True
            r = cp.add_run(ln if ln else " ")
            r.font.name = "Consolas"; r.font.size = Pt(9.5)
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(4)

    def _pb(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def _refs(self, refdb):
        for i, key in enumerate(self.cite_order, 1):
            if key not in refdb:
                raise KeyError(f"missing reference {key}")
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-1.0)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            add_text(p, f"[{i}]\t{refdb[key]}")
            p.paragraph_format.tab_stops.add_tab_stop(Cm(1.0))

    def save(self, path):
        self.doc.save(path)


def _strings(b):
    for x in b[1:]:
        if isinstance(x, str):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                if isinstance(y, str):
                    yield y
                elif isinstance(y, (list, tuple)):
                    for z in y:
                        if isinstance(z, str):
                            yield z
