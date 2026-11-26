"""Web server (app/server.py) and job queue (app/jobs.py), with a stand-in for the models (no GPU needed)."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import server as S
from app.jobs import Job, JobManager, QueueFull, RateLimited, UserError
from app.pipeline import Analysis, FaceResult


def fake_analysis(prob: float = 0.9) -> Analysis:
    rng = np.random.default_rng(0)
    faces = []
    for _ in range(4):
        c = rng.uniform(0, 1, 10).astype(np.float32)
        faces.append(FaceResult(crop=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), prob=prob, logit=2.0,
                                concepts=c, contributions=c * 0.5, residual=0.1,
                                maps=rng.uniform(0, 1, (10, 224, 224)).astype(np.float32)))
    a = Analysis(mode="fake", faces=faces, n_frames=4, prob=prob, verdict="Likely fake" if prob > 0.65 else "Likely real",
                 n_used=4)
    return a


class FakeEngine:
    name, version = "Fake engine", "test"

    def rescore(self, face, off):
        return 0.2 if off else face.prob


class FakeModels:
    """Same interface as server.Models; `gate` lets a test hold jobs in the running state."""

    def __init__(self, modes=("Deep", "Quick")):
        self.modes, self.status = list(modes), {m: "ready" for m in modes}
        self.gate = threading.Event()
        self.gate.set()
        self.behaviour = "ok"

    def available(self):
        return self.modes

    def run(self, job, progress):
        from app import render as R
        progress("faces", 0, 4)
        while not self.gate.wait(0.02):
            progress("faces", 1, 4)  # lets cancel / time limit stop the job
        if self.behaviour == "noface":
            raise UserError("We couldn't find a face", "Try another file.")
        if self.behaviour == "crash":
            raise RuntimeError("boom")
        a, eng = fake_analysis(), FakeEngine()
        job.result = {"verdict": R.verdict_data(a, str(job.path), job.is_video, job.name), "clues": R.clues_data(a),
                      "technical": R.technical_data(a, eng)}
        job.analysis = {"a": R.compact(a), "eng": eng}

    def rescore(self, job, off):
        a = job.analysis["a"]
        return float(np.mean([job.analysis["eng"].rescore(f, off) for f in a.faces]))


def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))
    return buf.tobytes()


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setenv("GRACE_PRELOAD", "0")
    clients = []

    def _make(models=None, **limits):
        models = models or FakeModels()
        c = TestClient(S.create_app(models=models, workers={"Deep": 1, "Quick": 1}, **limits))
        c.__enter__()
        clients.append(c)
        return c, models
    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def upload(c, name="face.png", data=None, mode="Deep"):
    return c.post("/api/check", files={"file": (name, data if data is not None else png_bytes())}, data={"mode": mode})


def wait(c, job_id, states=("done", "error", "cancelled"), timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["state"] in states:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job stuck: {j}")


def test_config_and_page(make):
    c, _ = make()
    cfg = c.get("/api/config").json()
    assert [m["id"] for m in cfg["modes"]] == ["Deep", "Quick"] and len(cfg["concepts"]) == 10
    page = c.get("/")
    assert page.status_code == 200 and "GRACE" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert c.get("/static/app.js").status_code == 200


def test_check_end_to_end(make, tmp_path):
    c, _ = make()
    r = upload(c)
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    j = wait(c, jid)
    assert j["state"] == "done", j
    res = j["result"]
    assert res["verdict"]["band"] == "Likely fake" and res["verdict"]["media"].startswith("data:image/jpeg")
    assert len(res["clues"]["cards"]) == 3 and len(res["technical"]["gallery"]) == 4
    assert not list((tmp_path / "uploads").glob("*")), "upload must be deleted after the check"
    g = c.post(f"/api/jobs/{jid}/overlay", json={"concept": 3, "opacity": 0.8}).json()
    assert len(g["gallery"]) == 4
    rs = c.post(f"/api/jobs/{jid}/rescore", json={"off": [0, 1]}).json()
    assert rs["prob"] == 0.2 and rs["verdict"] == "Likely real" and rs["was_verdict"] == "Likely fake"
    rep = c.get(f"/api/jobs/{jid}/report")
    assert rep.status_code == 200 and "attachment" in rep.headers["content-disposition"] and "SHA-256" in rep.text


@pytest.mark.parametrize("name,data,status", [
    ("notes.txt", b"hello", 415),                 # wrong extension
    ("clip.mp4", None, 415),                      # PNG content named .mp4
    ("face.jpg", b"not an image at all", 415),    # content is neither image nor video
    ("face.png", b"", 415),                       # empty
])
def test_bad_files(make, name, data, status):
    c, _ = make()
    r = upload(c, name, data)
    assert r.status_code == status and r.json()["detail"]["title"]


def test_too_large(make, monkeypatch):
    monkeypatch.setenv("GRACE_MAX_MB", "1")
    c, _ = make()
    r = upload(c, "big.png", png_bytes() + b"\0" * 1_200_000)
    assert r.status_code == 413 and "too large" in r.json()["detail"]["title"]


def test_unknown_mode_and_job(make):
    c, _ = make()
    assert upload(c, mode="Turbo").status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404
    assert c.post("/api/jobs/nope/rescore", json={"off": []}).status_code == 404


def test_one_active_check_per_client(make):
    c, m = make(max_active=1)
    m.gate.clear()
    first = upload(c)
    assert first.status_code == 202
    second = upload(c)
    assert second.status_code == 429 and "already have a check" in second.json()["detail"]["message"]
    m.gate.set()
    wait(c, first.json()["job_id"])
    assert upload(c).status_code == 202  # allowed again once the first one finished


def test_rate_window(make):
    c, _ = make(max_active=0, max_per_window=2)
    for _ in range(2):
        wait(c, upload(c).json()["job_id"])
    r = upload(c)
    assert r.status_code == 429 and "2 checks" in r.json()["detail"]["message"]


def test_queue_full_and_cancel(make):
    c, m = make(max_active=0, max_waiting=1)
    m.gate.clear()
    running = upload(c).json()["job_id"]
    wait(c, running, states=("running",))
    queued = upload(c).json()
    assert queued["position"] == 1
    full = upload(c)
    assert full.status_code == 503
    assert c.post(f"/api/jobs/{queued['job_id']}/cancel").json()["cancelled"]
    assert c.get(f"/api/jobs/{queued['job_id']}").json()["state"] == "cancelled"
    c.post(f"/api/jobs/{running}/cancel")
    assert wait(c, running)["state"] == "cancelled"
    m.gate.set()


def test_friendly_errors_and_time_limit(make):
    c, m = make(max_active=0, timeout=0.3)
    m.behaviour = "noface"
    j = wait(c, upload(c).json()["job_id"])
    assert j["state"] == "error" and j["error"]["title"] == "We couldn't find a face"
    m.behaviour = "crash"
    j = wait(c, upload(c).json()["job_id"])
    assert j["state"] == "error" and "our side" in j["error"]["title"]
    m.behaviour = "ok"
    m.gate.clear()
    j = wait(c, upload(c).json()["job_id"])
    assert j["state"] == "error" and "too long" in j["error"]["title"]
    m.gate.set()
    assert wait(c, upload(c).json()["job_id"])["state"] == "done"  # workers survive errors


def test_results_expire(tmp_path):
    dropped = []
    m = JobManager(lambda job, progress: None, {"Deep": 1}, ttl=0, max_kept=5, on_drop=dropped.append).start()
    try:
        job = m.submit(Job(mode="Deep", path=tmp_path / "x.png", name="x.png", client="a", is_video=False))
        for _ in range(100):
            if job.state == "done":
                break
            time.sleep(0.02)
        time.sleep(0.01)
        m.sweep()
        assert m.get(job.id) is None and dropped == [job]
    finally:
        m.stop()


def test_manager_limits_direct(tmp_path):
    m = JobManager(lambda job, progress: None, {"Deep": 1}, max_waiting=1, max_active=1)  # not started: jobs stay queued
    m.submit(Job(mode="Deep", path=tmp_path / "a", name="a", client="a", is_video=False))
    with pytest.raises(QueueFull):
        m.submit(Job(mode="Deep", path=tmp_path / "b", name="b", client="b", is_video=False))
    m2 = JobManager(lambda job, progress: None, {"Deep": 1}, max_waiting=5, max_active=1)
    m2.submit(Job(mode="Deep", path=tmp_path / "a", name="a", client="a", is_video=False))
    with pytest.raises(RateLimited):
        m2.submit(Job(mode="Deep", path=tmp_path / "c", name="c", client="a", is_video=False))


def test_client_id_trusts_cloudflare_header_only_from_localhost():
    class Req:
        def __init__(self, host, headers):
            self.client = type("C", (), {"host": host})()
            self.headers = headers
    assert S.client_id(Req("127.0.0.1", {"cf-connecting-ip": "203.0.113.5"})) == "203.0.113.5"
    assert S.client_id(Req("198.51.100.7", {"cf-connecting-ip": "203.0.113.5"})) == "198.51.100.7"
    assert S.client_id(Req("127.0.0.1", {})) == "127.0.0.1"


def test_sniff():
    assert S.sniff(png_bytes()[:16]) == "image"
    assert S.sniff(b"\x00\x00\x00\x18ftypmp42") == "video"
    assert S.sniff(b"\x1a\x45\xdf\xa3....") == "video"
    assert S.sniff(b"RIFF\0\0\0\0WEBPVP8 ") == "image"
    assert S.sniff(b"hello world") is None
