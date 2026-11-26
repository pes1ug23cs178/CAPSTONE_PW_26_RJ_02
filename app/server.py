"""GRACE web server: FastAPI + our own page (app/web/), replacing the Gradio UI.

Run (lab):    python -m app.server            -> http://127.0.0.1:8000
Public link:  scripts/laptop/start_server.ps1 (adds the Cloudflare tunnel)

API (JSON; errors carry detail = {"title", "message"} in plain words for the page):
  GET  /api/config                 limits, available check types, concept names
  GET  /api/health                 models loaded, queue sizes
  POST /api/check                  multipart: file, mode (Deep | Quick)  -> 202 {"job_id", ...}
  GET  /api/jobs/{id}              state, stage, place in line, estimated wait; the result when done
  POST /api/jobs/{id}/cancel
  POST /api/jobs/{id}/overlay      {"concept": k, "opacity": x}  -> concept-map gallery
  POST /api/jobs/{id}/rescore      {"off": [k, ...]}             -> score with those concepts switched off
  GET  /api/jobs/{id}/report       HTML report (download)

Many clients: the web layer is async and only queues jobs; model work runs in app/jobs.py worker threads
(Deep: 1 GPU worker, Quick: 2 CPU workers), with queue / per-client limits. Settings (environment):
  GRACE_ONLINE=1, GRACE_MAX_MB, GRACE_FRAMES, GRACE_MODEL_REPO   as for the old app (app/render.py)
  GRACE_MODES=Deep,Quick   check types offered (HF CPU Space: Quick)
  GRACE_WORKERS_DEEP=1 GRACE_WORKERS_QUICK=2 GRACE_MAX_WAITING=20 GRACE_MAX_ACTIVE=1 GRACE_PER_10MIN=10
  GRACE_TIMEOUT=180 (s per check)  GRACE_TTL=1800 (s a result is kept)  GRACE_PRELOAD=1  HOST, PORT
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.jobs import Job, JobManager, QueueFull, RateLimited, UserError

WEB = Path(__file__).resolve().parent / "web"
UPLOADS = Path(tempfile.gettempdir()) / "grace_web_uploads"
CHUNK = 1 << 20

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODE_LABELS = {"Deep": "Careful check", "Quick": "Quick check"}


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def sniff(head: bytes) -> Optional[str]:
    """'image' / 'video' from the first bytes of the file (the file name alone is not trusted)."""
    if head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or head[:2] == b"BM":
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3" or (head[:4] == b"RIFF" and head[8:12] == b"AVI "):
        return "video"
    return None


def error(status: int, title: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"title": title, "message": message})


def client_id(request: Request) -> str:
    """The visitor's IP. Behind the Cloudflare tunnel every request comes from 127.0.0.1, and the real address is
    in CF-Connecting-IP (only trusted for local connections, so it cannot be faked from outside)."""
    host = request.client.host if request.client else "unknown"
    if host in ("127.0.0.1", "::1", "localhost"):
        return request.headers.get("cf-connecting-ip") or host
    return host


class OverlayIn(BaseModel):
    concept: int = Field(0, ge=0, le=9)
    opacity: float = Field(0.45, ge=0, le=1)


class RescoreIn(BaseModel):
    off: List[int] = Field(default_factory=list, max_length=10)


# ---------- the model work (only this part imports the models) ----------
class Models:
    """Engines and croppers for the worker threads. Engines are loaded once and shared; each worker thread has its
    own face cropper; the GPU model runs one thing at a time (DEEP lock, also held by the switch-off re-score)."""

    def __init__(self, modes: List[str]) -> None:
        self.modes = modes
        self.status: Dict[str, str] = {m: "not loaded" for m in modes}
        self.local = threading.local()
        self.init_lock = threading.RLock()  # model loading and first runs, one thread at a time
        self.locks = {m: threading.Lock() if m == "Deep" else contextlib.nullcontext() for m in modes}

    def available(self) -> List[str]:
        from app.render import quick_onnx
        return [m for m in self.modes if m != "Quick" or quick_onnx().exists()]

    def engine(self, mode: str):
        from app.render import engine
        try:
            with self.init_lock:
                e = engine(mode)
            self.status[mode] = "ready"
            return e
        except Exception as ex:  # noqa: BLE001
            self.status[mode] = f"failed: {type(ex).__name__}: {ex}"
            raise

    def warmup(self, mode: str) -> None:
        """Runs in each worker thread at start: loads the model and this thread's face detector and runs both once
        (detector and landmark network), so the first visitor does not pay ~40 s of start-up. Warm-ups run one at a
        time: importing torch's compiler modules from several threads at once fails (seen 26 Nov 2026)."""
        import numpy as np
        if os.environ.get("GRACE_PRELOAD", "1") != "1" or mode not in self.available():
            return
        with self.init_lock:
            eng = self.engine(mode)
            fc = self.cropper()
            frame = np.full((480, 640, 3), 120, np.uint8)
            fc.crop_with_pose(frame)                                       # face detector
            fc.fa.get_landmarks_from_image(frame, detected_faces=[(200, 120, 440, 400)])  # landmark network
            with self.locks[mode]:
                eng.run([np.zeros((224, 224, 3), np.uint8)])

    def cropper(self):
        if not hasattr(self.local, "cropper"):
            with self.init_lock:
                from app.pipeline import FaceCropper
                self.local.cropper = FaceCropper()
        return self.local.cropper

    def run(self, job: Job, progress: Callable) -> None:
        import cv2
        from app.pipeline import analyse
        from app import render as R
        if job.is_video:
            cap = cv2.VideoCapture(str(job.path))
            ok = cap.isOpened() and cap.grab()
            cap.release()
            if not ok:
                raise UserError("We can't read this video", "The file may be damaged or use an unusual format. "
                                "Try saving it again as MP4.")
        elif cv2.imread(str(job.path)) is None:
            raise UserError("We can't read this photo", "The file may be damaged. Try a JPG or PNG instead.")
        eng = self.engine(job.mode)
        with self.locks[job.mode]:
            a = analyse(str(job.path), eng, self.cropper(), is_video=job.is_video, n_frames=R.N_FRAMES,
                        progress=progress)
        if not a.faces:
            raise UserError("We couldn't find a face",
                            "We need a clear, front-facing face to check. " +
                            ("Try a clip where the face is visible for a few seconds." if job.is_video
                             else "Try a photo where the face is clearly visible."))
        progress("explaining", 0, 1)
        job.result = {"verdict": R.verdict_data(a, str(job.path), job.is_video, job.name),
                      "clues": R.clues_data(a), "technical": R.technical_data(a, eng)}
        job.analysis = {"a": R.compact(a), "eng": eng}

    def rescore(self, job: Job, off: List[int]) -> float:
        import numpy as np
        a, eng = job.analysis["a"], job.analysis["eng"]
        faces = [f for f in a.faces if f.reliable] or a.faces
        with self.locks[job.mode]:
            return float(np.mean([eng.rescore(f, off) for f in faces]))


# ---------- the web app ----------
def create_app(models=None, workers: Optional[Dict[str, int]] = None, **limits) -> FastAPI:
    from app.pipeline import CONCEPT_LABELS, band
    online = os.environ.get("GRACE_ONLINE") == "1"
    max_mb = env_int("GRACE_MAX_MB", 50 if online else 200)
    modes = [m.strip() for m in os.environ.get("GRACE_MODES", "Deep,Quick").split(",") if m.strip()]
    models = models or Models(modes)
    workers = workers or {"Deep": env_int("GRACE_WORKERS_DEEP", 1), "Quick": env_int("GRACE_WORKERS_QUICK", 2)}
    workers = {m: n for m, n in workers.items() if m in models.modes and n > 0}
    opts = dict(max_waiting=env_int("GRACE_MAX_WAITING", 20), max_active=env_int("GRACE_MAX_ACTIVE", 1),
                max_per_window=env_int("GRACE_PER_10MIN", 10), window=600, timeout=env_int("GRACE_TIMEOUT", 180),
                ttl=env_int("GRACE_TTL", 1800))
    opts.update(limits)
    manager = JobManager(models.run, workers, warmup=getattr(models, "warmup", None), **opts)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        for old in UPLOADS.glob("*"):  # files left by a crash
            with contextlib.suppress(OSError):
                old.unlink()
        manager.start()  # each worker warms up its own model and face detector (Models.warmup)
        yield
        manager.stop()

    app = FastAPI(title="GRACE deepfake check", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.manager, app.state.models = manager, models

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path == "/api/check":  # refuse big uploads before reading them
            n = request.headers.get("content-length")
            if n and n.isdigit() and int(n) > max_mb * 1_000_000 + 100_000:
                return JSONResponse(status_code=413, content={"detail": {
                    "title": "This file is too large",
                    "message": f"Please upload a file smaller than {max_mb} MB, or a shorter clip."}})
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def job_or_404(job_id: str) -> Job:
        job = manager.get(job_id)
        if job is None:
            raise error(404, "This result has expired", "Results are kept for 30 minutes. Please check the file again.")
        return job

    def ready_job(job_id: str) -> Job:
        job = job_or_404(job_id)
        if job.state != "done" or job.analysis is None:
            raise error(409, "Not ready yet", "Wait for the check to finish.")
        return job

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/api/config")
    def config():
        avail = models.available()
        return {"max_mb": max_mb, "online": online, "concepts": CONCEPT_LABELS,
                "modes": [{"id": m, "label": MODE_LABELS.get(m, m), "available": m in avail} for m in models.modes],
                "accept": sorted(VIDEO_EXT | IMAGE_EXT)}

    @app.get("/api/health")
    def health():
        return {"ok": True, "models": dict(models.status), "warm_workers": manager.warm, "queue": manager.stats()}

    @app.post("/api/check", status_code=202)
    async def check(request: Request, file: UploadFile = File(...), mode: str = Form("Deep")):
        if mode not in models.available():
            raise error(400, "This type of check is not available", "Please choose another type of check.")
        name = Path(file.filename or "upload").name[:120]
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT | IMAGE_EXT:
            raise error(415, "We can't read this file type",
                        f"Please upload a photo (JPG, PNG, WEBP) or a video (MP4, MOV, AVI, MKV, WEBM), not {ext or 'this file'}.")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        path = UPLOADS / f"{os.urandom(12).hex()}{ext}"
        h, size, head = hashlib.sha256(), 0, b""
        try:
            with open(path, "wb") as out:
                while chunk := await file.read(CHUNK):
                    if not head:
                        head = chunk[:16]
                    size += len(chunk)
                    if size > max_mb * 1_000_000:
                        raise error(413, "This file is too large",
                                    f"Please upload a file smaller than {max_mb} MB, or a shorter clip.")
                    h.update(chunk)
                    out.write(chunk)
            kind = sniff(head)
            if size == 0 or kind is None:
                raise error(415, "We can't read this file", "It doesn't look like a photo or a video. "
                            "Please upload a JPG, PNG, WEBP, MP4, MOV, AVI, MKV or WEBM file.")
            is_video = kind == "video"
            if is_video != (ext in VIDEO_EXT):
                raise error(415, "The file type doesn't match its name",
                            f"This file is named {ext} but contains a {kind}. Please rename it or save it again.")
            job = manager.submit(Job(mode=mode, path=path, name=name, client=client_id(request), is_video=is_video,
                                     sha256=h.hexdigest()))
        except QueueFull:
            path.unlink(missing_ok=True)
            raise error(503, "We're busy right now", "Many people are checking files at the moment. "
                        "Please try again in a minute.")
        except RateLimited as e:
            path.unlink(missing_ok=True)
            raise error(429, "Please wait a moment", str(e))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"job_id": job.id, "state": job.state, "position": manager.position(job), "eta_s": manager.eta(job)}

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str):
        job = job_or_404(job_id)
        out = {"job_id": job.id, "state": job.state, "stage": job.stage, "done": job.done, "total": job.total,
               "mode": job.mode, "position": manager.position(job), "eta_s": manager.eta(job), "error": job.error}
        if job.state == "done":
            out["result"] = job.result
        return out

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = job_or_404(job_id)
        return {"cancelled": manager.cancel(job.id), "state": job.state}

    @app.post("/api/jobs/{job_id}/overlay")
    def overlay(job_id: str, body: OverlayIn):
        from app.render import gallery_data
        job = ready_job(job_id)
        return {"gallery": gallery_data(job.analysis["a"], body.concept, body.opacity)}

    @app.post("/api/jobs/{job_id}/rescore")
    def rescore(job_id: str, body: RescoreIn):
        off = sorted({k for k in body.off if 0 <= k < len(CONCEPT_LABELS)})
        job = ready_job(job_id)
        a = job.analysis["a"]
        p = models.rescore(job, off)
        return {"prob": round(p, 4), "verdict": band(p), "was_prob": round(a.prob, 4), "was_verdict": a.verdict,
                "off": [CONCEPT_LABELS[k] for k in off]}

    @app.get("/api/jobs/{job_id}/report")
    def report(job_id: str):
        from app.render import report_html
        job = ready_job(job_id)
        html = report_html(job.analysis["a"], job.analysis["eng"], job.name, job.sha256,
                           datetime.fromtimestamp(job.created).strftime("%Y-%m-%d %H:%M:%S"))
        stem = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in Path(job.name).stem)[:40] or "file"
        return HTMLResponse(html, headers={
            "Content-Disposition": f'attachment; filename="grace_report_{stem}_{time.strftime("%Y%m%d_%H%M")}.html"'})

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("GRACE_ONLINE") == "1" else "127.0.0.1")
    uvicorn.run(create_app(), host=host, port=env_int("PORT", 8000), proxy_headers=False, log_level="info")


if __name__ == "__main__":
    main()
