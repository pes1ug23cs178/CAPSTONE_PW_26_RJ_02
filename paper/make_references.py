"""paper/references.tex from the verified reference list of the report (ESA Report/build/backmatter.py: REFS),
in order of first citation in paper/main.tex. Fails if main.tex cites a key that is not in REFS.

python paper/make_references.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ESA Report" / "build"))
from backmatter import REFS  # noqa: E402

SUBS = [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("\u201c", "``"), ("\u201d", "''"),
        ("\u2019", "'"), ("\u2013", "--"), ("\u2014", "---")]


def tex(s: str) -> str:
    urls = re.findall(r"https?://\S+", s)
    for i, u in enumerate(urls):
        s = s.replace(u, f"@@URL{i}@@")
    for a, b in SUBS:
        s = s.replace(a, b)
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    for i, u in enumerate(urls):
        s = s.replace(f"@@URL{i}@@", r"\url{" + u.rstrip(".") + "}")
    return s


def main():
    src = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    keys = []
    for group in re.findall(r"\\cite\{([^}]+)\}", src):
        for k in group.split(","):
            k = k.strip()
            if k not in keys:
                keys.append(k)
    missing = [k for k in keys if k not in REFS]
    if missing:
        raise SystemExit(f"cited but not in REFS: {missing}")
    out = [r"\begin{thebibliography}{99}", r"\footnotesize"]
    out += [r"\bibitem{" + k + "} " + tex(REFS[k]) for k in keys]
    out.append(r"\end{thebibliography}")
    (ROOT / "paper" / "references.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(keys)} references")


if __name__ == "__main__":
    main()
