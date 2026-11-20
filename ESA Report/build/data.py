"""Loads every number the report quotes from the project's results files (nothing is typed in by hand)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def md_tables(path):
    """Markdown file -> {heading: [rows]} (each row a list of cells; first row = header)."""
    out, head, cur = {}, None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            head = line.lstrip("#").strip()
            cur = None
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if cur is None:
                key, n = head, 2
                while key in out:  # a second table under the same heading gets its own key
                    key, n = f"{head} ({n})", n + 1
                cur = out.setdefault(key, [])
            cur.append(cells)
        else:
            cur = None
    return out


class Results:
    def __init__(self):
        self.t = md_tables(RES / "tables.md")
        counts = (REPO / "splits" / "counts.md").read_text(encoding="utf-8")
        self.cdf_mode = re.search(r"Celeb-DF test mode: \*\*(\w+)\*\*", counts).group(1)
        self.counts = md_tables(REPO / "splits" / "counts.md")
        self.exp = pd.read_csv(REPO / "experiments.csv", dtype=str, keep_default_na=False)
        self.eff = pd.read_csv(RES / "efficiency.csv")
        self.app = md_tables(RES / "app_check.md")

    # main video-AUC table: key = first token of the model label (B1, C1, F1, S3 ...)
    def table(self, prefix):
        for k, v in self.t.items():
            if k.startswith(prefix):
                return v
        raise KeyError(prefix)

    def row(self, prefix, model):
        tab = self.table(prefix)
        for r in tab[1:]:
            if r[0].split()[0] == model:
                return dict(zip(tab[0], r))
        raise KeyError(model)

    def auc(self, model, col, table="1. Main"):
        return self.row(table, model)[col]

    def val(self, model, col, table="1. Main"):
        """numeric mean of a 'x ± s' cell"""
        s = self.auc(model, col, table)
        return float(s.split("±")[0]) if s not in ("–", "-", "nan", "") else float("nan")

    def main_rows(self):
        tab = self.table("1. Main")
        return tab

    def runs(self, config_prefix):
        d = self.exp[self.exp["config"].str.contains(f"configs/{config_prefix}.yaml", regex=False)]
        return d

    def seed_mean(self, config, col):
        d = self.runs(config)
        d = d[d[col] != ""]
        d = d.drop_duplicates("seed", keep="last")
        v = d[col].astype(float).to_numpy()
        return (float(v.mean()) if len(v) else float("nan"), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))

    def fmt(self, config, col):
        m, s, n = self.seed_mean(config, col)
        if not n:
            return "–"
        return f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f}"

    def nseeds(self, config):
        return self.seed_mean(config, "ffpp_test_video_auc")[2]


R = Results()


def cdf_note():
    if R.cdf_mode == "official":
        return "Celeb-DF v2 numbers use the official 518-video test list."
    return ("Celeb-DF v2 numbers in this version use the interim 518-video test set with the official composition "
            "(108 Celeb-real, 70 YouTube-real, 340 Celeb-synthesis); the official list was received on 20 November 2026 "
            "and the re-evaluation on it is in progress.")
