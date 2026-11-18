"""Train another seed of the final teacher: python scripts/train_final.py --seed 1"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import final_teacher_name  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402

if __name__ == "__main__":
    name = final_teacher_name()
    cmd = [sys.executable, str(REPO_ROOT / "train.py"), "--config", f"configs/{name}.yaml", *sys.argv[1:]]
    print("train_final:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT))
