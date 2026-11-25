"""Writing-guideline audit of the built Phase-3 report (same rules as paper/audit_style.py).

python "ESA Report/build/audit_report.py" [report.docx]
Reads the generated .docx (chapters, abstract and appendices; the department template pages and the reference
list are skipped) and prints every banned word or phrase, em dash, participial tail, sentence-initial
"Additionally", "key" used as an adjective and "significant" without "statistically".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
from audit_style import BANNED, KEY_ADJ, SIGNIFICANT, TAILS  # noqa: E402

DOCX = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ESA Report" / "output" / "PW26_RJ_02_Phase3_Report.docx"


def paragraphs():
    doc = Document(str(DOCX))
    started = False
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if not started:
            started = t.upper() == "ABSTRACT"
            continue
        if t.upper() in ("REFERENCES", "BIBLIOGRAPHY"):
            started = False  # skip the reference list; appendices after it restart at their heading
            continue
        if not started and t.upper().startswith("APPENDIX"):
            started = True
        yield t
    for tb in doc.tables:
        for row in tb.rows:
            for c in row.cells:
                if c.text.strip():
                    yield c.text.strip()


def main():
    issues = 0
    seen = set()
    for t in paragraphs():
        if t in seen:
            continue
        seen.add(t)
        low = t.lower()
        found = [f"banned '{w}'" for w in BANNED if re.search(rf"\b{re.escape(w)}", low)]
        if KEY_ADJ.search(t):
            found.append("'key' as adjective")
        if "—" in t or "---" in t:
            found.append("em dash")
        if TAILS.search(t):
            found.append("participial tail")
        if SIGNIFICANT.search(t):
            found.append("'significant' without 'statistically'")
        if re.search(r"(^|[.!?]\s+)Additionally\b", t):
            found.append("sentence-initial Additionally")
        for f in found:
            issues += 1
            print(f"{f}: {t[:140]}")
    print(f"{issues} issues in {len(seen)} paragraphs")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
