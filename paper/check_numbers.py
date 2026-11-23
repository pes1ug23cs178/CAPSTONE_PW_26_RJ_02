"""Check that every decimal number in paper/main.tex appears in a results file or a generated table.

python paper/check_numbers.py   (prints numbers it cannot find; exit code 1 if any)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"0.5", "0.4", "0.25", "0.2", "0.15", "0.3", "0.1", "0.05", "0.999", "0.75", "0.35", "0.01"}  # recipe/config values


def numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![\d.])\d+\.\d+", text))


def main():
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?m)%.*$", "", tex)
    pool = ""
    for p in list((ROOT / "results").glob("*.md")) + list((ROOT / "results").glob("*.csv")) + \
            list((ROOT / "paper" / "tables").glob("*.tex")) + [ROOT / "splits" / "counts.md"]:
        pool += p.read_text(encoding="utf-8", errors="ignore") + "\n"
    have = numbers(pool)
    # results csv files hold full precision: add their 3-decimal roundings
    for n in list(have):
        try:
            have.add(f"{float(n):.3f}")
            have.add(f"{float(n):.2f}")
            have.add(f"{float(n):.1f}")
            have.add(f"{float(n) * 100:.1f}")  # proportions quoted as percentages
        except ValueError:
            pass
    missing = sorted(n for n in numbers(tex) if n not in have and n not in ALLOWED)
    print("not found in results:", missing if missing else "none")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
