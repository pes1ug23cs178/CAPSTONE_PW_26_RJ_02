"""Write configs for the optional experiments from the FINAL g3_full.yaml (each changes one thing from G3).

d2_unsup_adapt.yaml : G3 + unsupervised adaptation to unlabelled Celeb-DF (runbook filler D2). Domain head is
                      dataset-level (0 = labelled FF++/SBI, 1 = unlabelled Celeb-DF): not class-correlated, so it is
                      applied to all samples; MMD between the two datasets. Target rows have no class/concept loss.
x1_ffpp_dfd.yaml    : G3 trained on FF++ + DFD (extra, motivated by the old checkpoint's Celeb-DF 0.956). Tested on
                      FF++ and Celeb-DF only (DFD is training data here).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    g3 = yaml.safe_load((REPO_ROOT / "configs" / "g3_full.yaml").read_text())
    d2 = copy.deepcopy(g3)
    d2.update(name="d2_unsup_adapt", parent="g3_full",
              notes="G3 + unsupervised adaptation to unlabelled Celeb-DF (non-test videos); dataset-level DANN + MMD")
    d2["data"].update(train="cache/adapt/d2_train.csv", domain_mode="dataset",
                      cells=list(g3["data"]["cells"]) + ["target"],
                      cell_weights={**g3["data"].get("cell_weights", {}), "target": 3})
    d2["model"]["domain"] = {"enabled": True, "num_domains": 2, "dann_weight": 0.1, "mmd_weight": 0.1,
                             "mmd_on": "all", "domain_on": "all"}
    (REPO_ROOT / "configs" / "d2_unsup_adapt.yaml").write_text(yaml.safe_dump(d2, sort_keys=False, width=120))

    x1 = copy.deepcopy(g3)
    x1.update(name="x1_ffpp_dfd", parent="g3_full", notes="G3 trained on FF++ + DFD; tested on FF++ and Celeb-DF only")
    x1["data"].update(train="cache/adapt/x1_train.csv", cells=list(g3["data"]["cells"]) + ["DeepFakeDetection"],
                      cell_weights={**g3["data"].get("cell_weights", {}), "DeepFakeDetection": 1},
                      tests=["splits/ffpp_test.csv", "splits/celebdf_test.csv"])
    dom = dict(x1["model"].get("domain") or {})
    if dom.get("enabled"):
        dom["num_domains"] = 6  # + DeepFakeDetection
    x1["model"]["domain"] = dom
    (REPO_ROOT / "configs" / "x1_ffpp_dfd.yaml").write_text(yaml.safe_dump(x1, sort_keys=False, width=120))
    print("wrote configs/d2_unsup_adapt.yaml and configs/x1_ffpp_dfd.yaml")


if __name__ == "__main__":
    main()
