# Deploying the GRACE app on Hugging Face

> **Status (20 Nov 2026):** the models are on Hugging Face (`grace-cap/grace-deepfake-models`, public). Hosting a
> Gradio Space now requires a Hugging Face PRO plan, so the live app runs on a team laptop with a public link
> instead: see **`docs/LAPTOP_DEMO.md`**. The Space steps below still apply if PRO is bought later.

The public app is a **Hugging Face Space** (Gradio, free CPU). It downloads the trained model files from a
**Hugging Face model repo** when it starts, so the Space itself only holds code.

| Piece | Name | Holds |
| --- | --- | --- |
| GitHub | `pes1ug23cs178/CAPSTONE_PW_26_RJ_02` | All code, results, report, slides |
| Model repo | `grace-cap/grace-deepfake-models` (live) | Everything in `release/` (about 84 MB) |
| Space | `<user>/grace-deepfake-check` | Output of `scripts/build_space.py` |

## One-time setup (any laptop)

```bash
pip install -U huggingface_hub
huggingface-cli login                 # paste a WRITE token (Hugging Face → Settings → Access Tokens)
```

## 1. Upload the models

```bash
huggingface-cli repo create grace-deepfake-models --type model          # add --private for a private repo
huggingface-cli upload <user>/grace-deepfake-models release/ . --commit-message "GRACE release"
```

## 2. Build and push the Space

```bash
python scripts/build_space.py --model-repo <user>/grace-deepfake-models
huggingface-cli repo create grace-deepfake-check --type space --space_sdk gradio
huggingface-cli upload <user>/grace-deepfake-check deploy/hf_space . --repo-type space --commit-message "GRACE app"
```

In the Space's **Settings → Variables and secrets**, add:

| Name | Value | Kind |
| --- | --- | --- |
| `GRACE_MODEL_REPO` | `<user>/grace-deepfake-models` | variable |
| `HF_TOKEN` | a READ token (only if the model repo is private) | secret |

The Space builds itself (about 5 minutes), then runs. The first check downloads CLIP ViT-L/14 (about 1.2 GB) and
the face detector, so it takes a few minutes. Later checks take about 40–70 s (Careful) or 10–20 s (Quick) on the
free 2-core CPU.

## App settings (environment variables)

| Variable | Lab default | Space default | Meaning |
| --- | --- | --- | --- |
| `GRACE_ONLINE` | off | `1` | Public mode: 50 MB uploads, 16 frames, no dataset faces, privacy note, listens on 0.0.0.0 |
| `GRACE_MODEL_REPO` | unset (uses `release/`) | your model repo | Where the model files come from |
| `GRACE_MODEL_REVISION` | latest | latest | Pin a model-repo commit |
| `GRACE_DEVICE` | auto | `cpu` | Force `cpu` or `cuda` |
| `GRACE_FRAMES` | 32 | 16 | Frames per video |
| `GRACE_MAX_MB` | 200 | 50 | Upload limit |

## Test the public version locally (no GPU)

```bash
python scripts/build_space.py --model-repo <user>/grace-deepfake-models
cd deploy/hf_space
set CUDA_VISIBLE_DEVICES=-1        # Windows (Linux/macOS: export CUDA_VISIBLE_DEVICES=-1)
set PORT=7861
python space_app.py                # http://127.0.0.1:7861
```

Checked on 20 Nov 2026 with the GPU hidden: all six app test files gave the correct verdict.

## When something breaks

Open the Space's **Logs** tab first. Common fixes:

| Symptom | Fix |
| --- | --- |
| Build error | Fix the version named in the log in `requirements.txt`, rebuild with `build_space.py`, upload again |
| `libGL.so.1` missing | `packages.txt` must list `libgl1` (it does by default) |
| 401 / repository not found when loading models | Set the `HF_TOKEN` secret, or make the model repo public |
| `Killed` / out of memory | Restart the Space; keep `GRACE_FRAMES` at 16 |
| CUDA errors | `GRACE_DEVICE` must be `cpu` on the free Space (the launcher sets it) |
| Sleeping | Free Spaces sleep when idle; open the link a few minutes before a demo |

Roll back: each upload is a commit. Revert it in the repo's *Files and versions → History*, or pin
`GRACE_MODEL_REVISION` to a known-good model commit.
