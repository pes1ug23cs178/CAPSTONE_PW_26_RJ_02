# Recovery: how to get back to a known-good state

## Restore points

| Tag | Date | State |
|---|---|---|
| `snap/capstone-final` | 2026-11-25 | Capstone complete: all results (reduced FF++ copy), paper (12 pages), Phase-3 report (77 pages), app. |

Each restore point has a folder `backups/<date>_<name>/` (local, C: drive, not in git) with
`checkpoints.csv` (sha256 of every trained model), `run_outputs.zip` (all metrics and predictions) and `git.txt`.
Make a new one with `python scripts/snapshot.py <name>` before any risky step.

## Rules for the full-FF++ retrain (so nothing above can be overwritten)

- Work happens on the git branch `fullff`, never on `sprint` or `main`.
- Videos go to `ff++_full/`, crops to `ff++_full/faces/`. The current `ff++/` folder is not touched.
- New split files: `splits/fullff_*.csv`. The current `splits/ffpp_*.csv` stay as they are.
- New model names end in `_full` (e.g. `f1_grace_dct_full`), so runs never mix with the current ones.
- New results go to `results/fullff/`. The current `results/` files stay as they are.
- Every step is logged in STATUS.md with date and time.

## How to go back

1. **Code, results, paper and report**: `git checkout sprint` (the full-FF++ branch is left aside), or for a
   specific point `git checkout snap/capstone-final`. Nothing is deleted; the branch can be revisited.
2. **Check that the trained models are unchanged**: compare `runs/*/best.pth` against
   `backups/20261125_capstone-final/checkpoints.csv` (sha256). The files are read-only, so they should match.
3. **Metrics or predictions overwritten by mistake**: unzip the files you need from
   `backups/20261125_capstone-final/run_outputs.zip` back into `runs/`.
4. **Models lost from disk**: the released models are also on Hugging Face (grace-cap/grace-deepfake-models).
