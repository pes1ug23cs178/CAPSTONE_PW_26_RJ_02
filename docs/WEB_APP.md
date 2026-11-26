# GRACE web app

Our own web app (FastAPI + a plain HTML/CSS/JS page), replacing the Gradio UI. The detection code is unchanged
(`app/pipeline.py`); only the layer around it is new. The old Gradio app (`app/app.py`) still works.

## Architecture

```
Browser ── index.html + app.css + app.js (app/web/, Template A design, no framework)
   │  HTTPS
Cloudflare Tunnel (tools/cloudflared.exe) ── fixed address, no time limit
   │  http://127.0.0.1:8000
FastAPI server (app/server.py, uvicorn) ── API, upload checks, security headers
   │
Job queue (app/jobs.py) ── one FIFO queue per check type; Deep: 1 GPU worker, Quick: 2 CPU workers
   │
Models (app/pipeline.py, via app/render.py) ── face cropper (one per worker), F1 teacher (GPU), S3 student (ONNX, CPU)
   │
Model files ── runs/ (lab) → release/ → Hugging Face grace-cap/grace-deepfake-models (auto-download)
```

| File | Role |
| --- | --- |
| `app/server.py` | API endpoints, upload validation (size, type by content), client identity, worker warm-up |
| `app/jobs.py` | Job queue, workers, limits, cancel, time limit, expiry (no model code; tested without a GPU) |
| `app/render.py` | Shared by both UIs: settings, model loading, plain-language wording, images, report |
| `app/pipeline.py` | Face crops, the two engines, `analyse()` (now with a progress callback) |
| `app/web/` | The page |
| `scripts/start_server.ps1` | Starts the server and the tunnel |
| `tests/test_server.py` | API and queue tests (stand-in model) |
| `scripts/web_check.py` | Browser click-through (Playwright + Edge) |

## How one check works

1. The page uploads the file (`POST /api/check`) with a progress bar.
2. The server checks it: size limit, extension, and the first bytes (a renamed file is refused), then saves it under
   a random name, computes its SHA-256 and returns a job ID at once (HTTP 202). No request waits for the model.
3. A worker for that check type takes the job: reads frames, finds faces (32 of them for a video), scores them,
   builds the explanation. Each step updates the job's stage.
4. The page asks `GET /api/jobs/{id}` every second and shows the place in line or the stage ("Finding faces · 12 of 32").
5. When done, the page draws the verdict, the three clues and the technical details from the JSON result.
6. Overlay, switch-off re-score and report use the stored result (`/overlay`, `/rescore`, `/report`).
7. The upload is deleted when the check ends; the result is deleted after 30 minutes.

## Many clients

- The web layer is async and only queues work, so many people can upload and wait at once.
- Models are loaded once at start. Each worker warms up its model and face detector before the first visitor
  (one worker at a time: loading torch's compiler modules from several threads at once fails).
- Careful checks run one at a time on the GPU; quick checks two at a time on the CPU. Others wait in line and see
  their place and an estimated wait (average of the last 20 checks).
- Job IDs are random 128-bit tokens; there are no accounts or shared state between visitors.

| Limit | Default | Setting |
| --- | --- | --- |
| Waiting checks (all visitors) | 20, then "We're busy" (503) | `GRACE_MAX_WAITING` |
| Checks running or waiting per visitor | 1 (429) | `GRACE_MAX_ACTIVE` |
| Checks per visitor per 10 minutes | 10 (429) | `GRACE_PER_10MIN` |
| Time per check | 180 s | `GRACE_TIMEOUT` |
| Upload size | 200 MB lab, 95 MB through the tunnel | `GRACE_MAX_MB` |
| Result kept | 30 min (at most 30 results in memory) | `GRACE_TTL` |
| Workers | Deep 1, Quick 2 | `GRACE_WORKERS_DEEP`, `GRACE_WORKERS_QUICK` |
| Check types offered | Deep, Quick | `GRACE_MODES` (CPU-only machines: Quick) |

A visitor is identified by IP. Behind the tunnel every request comes from 127.0.0.1, so the server reads
Cloudflare's `CF-Connecting-IP` header, and only for local connections (it cannot be faked from outside).

Measured on the lab PC (RTX 4500, training running at the same time, 26 Nov 2026): one check 5-8 s; 8 visitors at
once (5 careful + 3 quick) all finished within 43 s; same verdicts as the Gradio app on the six app test files.

## Running it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Local          # this computer only
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1                 # + quick public link (changes each start)
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Tunnel grace   # + your fixed address
```

Add `-Hf` on a laptop without `runs/` to download the models from Hugging Face.

## Fixed address (named tunnel), one-time setup

Needs a free Cloudflare account and a domain added to it (bought, or moved to Cloudflare's name servers).

```powershell
tools\cloudflared.exe tunnel login                       # opens the browser: pick the domain
tools\cloudflared.exe tunnel create grace                # writes %USERPROFILE%\.cloudflared\<tunnel-id>.json
tools\cloudflared.exe tunnel route dns grace check.<your-domain>
```

Then create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: grace
credentials-file: C:\Users\<you>\.cloudflared\<tunnel-id>.json
ingress:
  - hostname: check.<your-domain>
    service: http://127.0.0.1:8000
  - service: http_status:404
```

and start with `-Tunnel grace`. The address stays the same across restarts. The credentials file is a secret:
never commit it.

## Later: Hugging Face Docker Space

After training and testing finish: the same server in a Docker Space (free CPU), with `GRACE_MODES=Quick`,
`GRACE_MODEL_REPO=grace-cap/grace-deepfake-models`, `GRACE_ONLINE=1`, `HOST=0.0.0.0`, `PORT=7860`.

## Tests

```powershell
python -m pytest tests/test_server.py -q                                   # API + queue, no GPU
python scripts/web_check.py cache/app_test_videos/fake_ffpp_Deepfakes_233_995.mp4 out/ffpp --no-face cache/app_test_videos/no_face.jpg
```
