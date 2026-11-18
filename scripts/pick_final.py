"""Pick the final GRACE teacher among the final-model candidates on FF++ VALIDATION only.

Score = best epoch's val_select_score (mean of clean val video AUC, h264:40 val video AUC and SBI-val vs real-val AUC,
see train.py); ties -> clean val frame AUC. Test splits are never read. Writes results/final_teacher.json, which
the teacher cache, students, app, release, explanation and table scripts read (utils.common.final_teacher_name)."""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402

CANDIDATES = ["f1_grace_dct", "f2_grace_pdann"]


def best(name: str):
    runs = sorted(p for p in glob.glob(str(REPO_ROOT / "runs" / f"{name}_s42_*")) if Path(p, "best.pth").exists()
                  and Path(p, "history.json").exists())
    if not runs:
        return None
    hist = json.loads(Path(runs[-1], "history.json").read_text())
    h = max(hist, key=lambda r: (r.get("val_select_score", r["val_video_auc"]), r["val_frame_auc"]))
    return {"name": name, "run": Path(runs[-1]).name, **{k: h.get(k) for k in (
        "epoch", "val_select_score", "val_video_auc", "val_hard_video_auc", "val_sbi_auc", "val_frame_auc", "val_concept_share")}}


def main() -> None:
    res = [r for r in (best(n) for n in CANDIDATES) if r]
    if not res:
        raise SystemExit("no finished candidate")
    win = max(res, key=lambda r: (r["val_select_score"], r["val_frame_auc"]))
    out = {"name": win["name"], "run": win["run"], "candidates": res}
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    (REPO_ROOT / "results" / "final_teacher.json").write_text(json.dumps(out, indent=2))
    with open(REPO_ROOT / "results" / "decisions.md", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} final teacher = {win['name']} ({win['run']}) on val select score: "
                + "; ".join(f"{r['name']} {r['val_select_score']:.4f}" for r in res) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
