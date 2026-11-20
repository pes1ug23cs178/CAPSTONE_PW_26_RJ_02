"""Extra (Day 7): student data with DFD added, to separate "more data" from "better teacher" for the students.

cache/adapt/student_x_train.csv = cache/sbi_pool/student_train.csv (FF++ train + frozen SBI pool) + the DFD rows of
cache/adapt/x1_train.csv (all DFD videos, labelled). DFD is then not a held-out test set for these students; the
clean comparison is Celeb-DF.

configs/s0x_student_dfd.yaml   S0 recipe (no teacher) on student_x_train
configs/s3xx_kd_x1_dfd.yaml    S3 recipe distilled from X1 on student_x_train, teacher cache cache/teacher_x1x
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    stud = pd.read_csv(REPO_ROOT / "cache/sbi_pool/student_train.csv", dtype=str, keep_default_na=False)
    x1 = pd.read_csv(REPO_ROOT / "cache/adapt/x1_train.csv", dtype=str, keep_default_na=False)
    dfd = x1[x1["dataset"] == "dfd"].reindex(columns=stud.columns, fill_value="")
    out = pd.concat([stud, dfd], ignore_index=True)
    assert not out["frame_path"].duplicated().any()
    dst = REPO_ROOT / "cache/adapt/student_x_train.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(stud)} student rows + {len(dfd)} DFD rows ({dfd['method'].value_counts().to_dict()})")

    for src, name, parent, notes, cache_dir in [
        ("s0_student", "s0x_student_dfd", "s0_student",
         "S0 recipe (no teacher) with DFD added to the student data", None),
        ("s3_kd_full", "s3xx_kd_x1_dfd", "s3x_kd_from_x1",
         "S3 recipe distilled from X1, with DFD added to the student data", "cache/teacher_x1x"),
    ]:
        cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{src}.yaml").read_text())
        cfg.update(name=name, parent=parent, notes=notes)
        cfg["data"]["train"] = "cache/adapt/student_x_train.csv"
        if "DeepFakeDetection" not in cfg["data"]["cells"]:
            cfg["data"]["cells"].append("DeepFakeDetection")
        cfg["data"]["cell_weights"]["DeepFakeDetection"] = 1
        if cache_dir:
            cfg["distill"]["cache_dir"] = cache_dir
        (REPO_ROOT / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote configs/{name}.yaml")


if __name__ == "__main__":
    main()
