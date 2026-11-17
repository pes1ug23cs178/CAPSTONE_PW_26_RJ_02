"""Day 4 start: pick G3's frequency branch by FF++ VALIDATION video AUC (never test) and write
configs/g3_full.yaml and configs/a3_linear_head.yaml.

Decision rules applied (runbook):
- G3 = the better of G2 (dct) / G2-fft on val, + DANN + MMD.
- If neither G2 variant beats G1 on val, G3 = G1 + DANN (negative result for the branch).
Writes the decision to results/decisions.md.
"""
from __future__ import annotations

import copy
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402


def best_val(exp: pd.DataFrame, config_name: str) -> float:
    rows = exp[exp["config"].str.endswith(f"{config_name}.yaml") & (exp["seed"] == 42)]
    return float(rows["val_video_auc"].max()) if len(rows) else float("nan")


def main() -> None:
    exp = pd.read_csv(REPO_ROOT / "experiments.csv")
    scores = {name: best_val(exp, name) for name in ("g1_concepts", "g2_dct", "g2_fft")}
    g2_choice = max(("g2_dct", "g2_fft"), key=lambda n: (scores[n] if scores[n] == scores[n] else -1))
    if not scores[g2_choice] > scores["g1_concepts"]:
        base_name, reason = "g1_concepts", f"neither G2 variant beat G1 on val ({scores})"
    else:
        base_name, reason = g2_choice, f"{g2_choice} best on val ({scores})"
    base = yaml.safe_load((REPO_ROOT / "configs" / f"{base_name}.yaml").read_text())

    g3 = copy.deepcopy(base)
    g3.update(name="g3_full", parent=base_name, notes=f"{base_name} + DANN (0.1) + MMD between fake domains (0.1); {reason}")
    g3["model"]["domain"] = {"enabled": True, "num_domains": 5, "dann_weight": 0.1, "mmd_weight": 0.1, "mmd_on": "fake"}
    (REPO_ROOT / "configs" / "g3_full.yaml").write_text(yaml.safe_dump(g3, sort_keys=False, width=120))

    a3 = copy.deepcopy(g3)
    a3.update(name="a3_linear_head", parent="g3_full", notes="G3 with the concept head + decision layer replaced by Linear(512 -> 1) on z")
    a3["model"]["head"] = "linear"
    a3["loss"] = {}
    (REPO_ROOT / "configs" / "a3_linear_head.yaml").write_text(yaml.safe_dump(a3, sort_keys=False, width=120))

    out = REPO_ROOT / "results" / "decisions.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} G3 base = {base_name}: {reason}\n")
    print(f"G3 base = {base_name}: {reason}")


if __name__ == "__main__":
    main()
