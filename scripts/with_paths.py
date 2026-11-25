"""Run a command with GRACE_PATHS_FILE set (for scripts that call load_paths() without a run config).

python scripts/with_paths.py configs/paths_fullff.yaml -- python scripts/extract_landmarks.py --splits fullff_train
"""
import os
import subprocess
import sys

i = sys.argv.index("--")
env = dict(os.environ, GRACE_PATHS_FILE=sys.argv[1])
sys.exit(subprocess.call(sys.argv[i + 1:], env=env))
