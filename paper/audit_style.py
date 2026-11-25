"""Style and structure audit for paper/Paper.tex (writing guidelines + LaTeX sanity).

python paper/audit_style.py [file]
Reports: banned words/phrases (case-insensitive, whole words), em dashes, sentence-initial "Additionally",
common participial tails, unbalanced braces, undefined references, missing figures, citations without entries.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
BANNED = ["additionally", "moreover", "furthermore", "subsequently", "meanwhile", "crucial", "pivotal", "vital",
          "landscape", "tapestry", "testament", "interplay", "intricacies", "insights", "synergies", "paradigm",
          "delve", "underscore", "underscores", "highlight", "highlights", "highlighted", "showcase", "boasts", "garner",
          "fostering", "cultivating", "bolstered", "enhance", "enhances", "enhanced", "align with", "resonate",
          "elevate", "revolutionize", "reimagine", "leverage", "leverages", "unleash", "harness", "vibrant", "profound",
          "groundbreaking", "renowned", "meticulous", "meticulously", "enduring", "diverse array", "intricate",
          "game changer", "deep dive", "cross-functional", "enablement", "touch point", "there's no denying",
          "across different", "human oversight", "to bridge", "whisper", "hustle", "it's like having",
          "stands as", "serves as", "a reminder of", "plays a", "reflects broader", "symboliz", "evolution of",
          "setting the stage", "shaping the future", "marking the", "represents a shift", "turning point",
          "focal point", "indelible", "deeply rooted", "isn't just", "aren't just", "don't just", "not only",
          "researchers believe", "experts argue", "it is worth noting", "novel", "state-of-the-art performance"]
KEY_ADJ = re.compile(r"\bkey\s+(?!\w*s\b)(component|idea|finding|role|step|factor|part|feature|insight|contribution)", re.I)
SIGNIFICANT = re.compile(r"(?<!statistically )\bsignificant(ly)?\b", re.I)
TAILS = re.compile(r",\s+(highlighting|underscoring|emphasizing|emphasising|ensuring|reflecting|symbolizing|"
                   r"contributing|fostering|encompassing|cultivating|showcasing)\b", re.I)


def prose(tex: str) -> str:
    t = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    t = re.sub(r"\\(cite|ref|label|includegraphics|input|url|eqref)\*?(\[[^\]]*\])?\{[^}]*\}", " ", t)
    return t


def main():
    f = P / (sys.argv[1] if len(sys.argv) > 1 else "Paper.tex")
    tex = f.read_text(encoding="utf-8")
    body = prose(tex)
    lines = body.splitlines()
    issues = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        for w in BANNED:
            if re.search(rf"\b{re.escape(w)}", low):
                issues.append((i, f"banned '{w}'", line.strip()[:110]))
        if KEY_ADJ.search(line):
            issues.append((i, "'key' as adjective", line.strip()[:110]))
        if "\u2014" in line or "---" in line:
            issues.append((i, "em dash", line.strip()[:110]))
        if TAILS.search(line):
            issues.append((i, "participial tail", line.strip()[:110]))
    flat = re.sub(r"\s+", " ", body)  # phrases can wrap across source lines
    for m in SIGNIFICANT.finditer(flat):
        issues.append((0, "'significant' without 'statistically'", flat[max(0, m.start() - 60):m.end() + 20]))
    if re.search(r"(^|[.!?]\s+)Additionally\b", body):
        issues.append((0, "sentence-initial Additionally", ""))
    # LaTeX structure, including \input-ed tables
    full = tex
    for m in re.findall(r"\\input\{([^}]+)\}", tex):
        q = P / (m if m.endswith(".tex") else m + ".tex")
        if q.exists():
            full += q.read_text(encoding="utf-8")
        else:
            issues.append((0, f"missing \\input file {m}", ""))
    t = re.sub(r"(?m)(?<!\\)%.*$", "", full).replace(r"\{", "").replace(r"\}", "")
    if t.count("{") != t.count("}"):
        issues.append((0, f"brace balance {t.count('{') - t.count('}')}", ""))
    b, e = Counter(re.findall(r"\\begin\{(\w+\*?)\}", t)), Counter(re.findall(r"\\end\{(\w+\*?)\}", t))
    if b != e:
        issues.append((0, f"environment mismatch {b - e} {e - b}", ""))
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", t))
    for r in sorted(refs - labels):
        issues.append((0, f"undefined reference {r}", ""))
    for r in sorted(l for l in labels - refs if l.startswith(("tab:", "fig:"))):
        issues.append((0, f"figure/table never referenced {r}", ""))
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
        if not (P / "figs" / g).exists():
            issues.append((0, f"missing figure {g}", ""))
    cites = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", t) for k in grp.split(",")}
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", t))
    for c in sorted(cites - bib):
        issues.append((0, f"citation without bibitem {c}", ""))
    for i, kind, txt in issues:
        print(f"line {i}: {kind}: {txt}")
    words = len(re.sub(r"\\[a-zA-Z]+\*?|[{}$\\]", " ", body.split(r"\begin{abstract}")[1]).split())
    print(f"{len(issues)} issues; about {words} words after the title block")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
