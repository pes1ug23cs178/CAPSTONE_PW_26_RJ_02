"""Restore point for the whole project: run before any risky step (new data, retraining, refactors).

python scripts/snapshot.py <name>          e.g.  python scripts/snapshot.py capstone-final

Makes, under backups/<date>_<name>/ (gitignored, stays on C:):
  checkpoints.csv   run, file, size, sha256 of every runs/*/best.pth (proves a checkpoint was not changed)
  run_outputs.zip   every run's config, history, metrics, predictions and logs (no checkpoints)
  git.txt           commit hash, branch, tag
and then:
  - marks every best.pth read-only, so no script can overwrite a finished model
  - creates the git tag snap/<name> on the current commit (push it with: git push origin snap/<name>)

Restore: see docs/RECOVERY.md.
"""
from __future__ import annotations

import csv
import hashlib
import os
import stat
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = (".json", ".csv", ".yaml", ".txt", ".log")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit or stash tracked changes first, so the tag matches the files")
    out = ROOT / "backups" / f"{date.today():%Y%m%d}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    runs = sorted(d for d in (ROOT / "runs").iterdir() if d.is_dir() and not d.name.startswith("_"))
    with open(out / "checkpoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "file", "bytes", "sha256"])
        for d in runs:
            for ck in sorted(d.glob("*.pth")):
                w.writerow([d.name, ck.name, ck.stat().st_size, sha256(ck)])
                os.chmod(ck, stat.S_IREAD)
    n = 0
    with zipfile.ZipFile(out / "run_outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in runs:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix in KEEP:
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
    tag = f"snap/{name}"
    git("tag", "-a", tag, "-m", f"restore point {name}")
    (out / "git.txt").write_text(f"commit {git('rev-parse', 'HEAD')}\nbranch {git('branch', '--show-current')}\n"
                                 f"tag {tag}\n", encoding="utf-8")
    print(f"{out}: {len(runs)} runs, {n} output files zipped, checkpoints read-only, tag {tag}")


if __name__ == "__main__":
    main()
