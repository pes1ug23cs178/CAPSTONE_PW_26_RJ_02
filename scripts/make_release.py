"""T7.4: copy final artefacts to release/ with SHA256SUMS (release/ is git-ignored).

release/grace_teacher_g3.pth, release/grace_student_s3.pth, release/grace_student.onnx (+ .json),
release/<name>_val_threshold.json, release/<name>_temperature.json, release/SHA256SUMS
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rel = REPO_ROOT / "release"
    rel.mkdir(exist_ok=True)
    for name, out in (("g3_full", "grace_teacher_g3"), ("s3_kd_full", "grace_student_s3")):
        run = resolve_runs([name], [42])[0]
        shutil.copy2(run / "best.pth", rel / f"{out}.pth")
        for extra in ("val_threshold.json", "temperature.json", "config.yaml"):
            if (run / extra).exists():
                shutil.copy2(run / extra, rel / f"{out}_{extra}")
        (rel / f"{out}_RUN.txt").write_text(run.name)
    files = sorted(p for p in rel.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (rel / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    print("release:", [p.name for p in files])


if __name__ == "__main__":
    main()
