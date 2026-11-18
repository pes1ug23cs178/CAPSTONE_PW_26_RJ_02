"""Read-only board of the run queue: DONE (in the order they ran), RUNNING, then QUEUED in execution order.
python scripts/queue_board.py          (scripts/queue_board.ps1 refreshes it every 10 s)"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
Q = REPO / "runs" / "_queue"

ABOUT = {
    "landmarks": "face landmarks for self-blended images", "sbi_check": "self-blended image check",
    "sbi_eval_sets": "fixed self-blended test sets", "smoke": "short test run",
    "b1_swin": "B1 Swin baseline", "b2_swin_dann": "B2 Swin + DANN", "b2f_swin_dann_fake": "B2-fake Swin + DANN on fakes",
    "c1_clip_lora": "C1 CLIP + LoRA", "c2_clip_sbi": "C2 + self-blended 1/3", "c2q_clip_sbi_quarter": "C2q + self-blended 1/4",
    "pick_sbi_share": "choose self-blended share (val)", "g1_concepts": "G1 + concept head",
    "g1r_concepts_res1e2": "G1r stronger explanations", "pick_residual": "choose explanation setting (val)",
    "g2_dct": "G2 + DCT frequency branch", "g2_fft": "G2-fft + FFT frequency branch", "g1_concept_auc_rescore": "re-score concepts",
    "g3_full": "G3 full teacher (+DANN)", "a3_linear_head": "A3 G3 without concepts", "robustness_eval": "robustness tests, 10 models",
    "f1_grace_dct": "F1 final candidate (no DANN)", "f2_grace_pdann": "F2 final candidate (DANN on patches)",
    "robustness_eval_final": "robustness tests, F1 + F2", "pick_final": "choose final teacher (val)",
    "sbi_pool": "40k self-blended training pool", "teacher_cache": "save teacher outputs for students",
    "s0_student": "S0 student, no teacher", "s2_kd_logits": "S2 student, copies answers", "s3_kd_full": "S3 student, full copy",
    "final": "final teacher, extra seed", "tsne_figure": "t-SNE feature figure", "export_onnx": "export student to ONNX",
    "agreement": "student vs teacher agreement", "benchmark": "speed / size test", "calibrate": "score calibration",
    "fig_explanations": "explanation figure", "explain_eval": "explanation metrics", "app_check": "demo app check",
    "final_eval_all": "final evaluation, all models", "make_tables": "results tables", "make_summary": "summary draft",
    "make_release": "release package", "d2_unsup_adapt": "D2 adapt to unlabelled Celeb-DF", "x1_ffpp_dfd": "X1 train on FF++ + DFD",
    "old_ckpt": "old model on new splits",
}


def about(name: str) -> str:
    for k in sorted(ABOUT, key=len, reverse=True):
        if name.startswith(k):
            return ABOUT[k]
    return ""


def hm(h) -> str:
    if h is None:
        return ""
    m = int(round(float(h) * 60))
    return f"{m // 60}h{m % 60:02d}"


def short(ts) -> str:
    return datetime.fromisoformat(ts).strftime("%d %b %H:%M") if ts else ""


def main() -> None:
    color = "--no-color" not in sys.argv
    C = (lambda c, s: f"\033[{c}m{s}\033[0m") if color else (lambda c, s: s)
    jobs = yaml.safe_load((REPO / "queue.yaml").read_text(encoding="utf-8"))["jobs"]
    state = json.loads((Q / "state.json").read_text(encoding="utf-8"))
    after = {j["name"]: j.get("after") for j in jobs}

    ran = sorted(state.items(), key=lambda kv: kv[1].get("start", ""))
    done = [(k, v) for k, v in ran if v["status"] in ("done", "failed", "blocked")]
    running = [(k, v) for k, v in ran if v["status"] == "running"]
    queued = [j["name"] for j in jobs if j["name"] not in state]

    print(C("1;36", f"GRACE run queue   {datetime.now():%d %b %Y %H:%M:%S}   "
                    f"done {sum(v['status'] == 'done' for _, v in done)} | running {len(running)} | queued {len(queued)}"))
    print(C("1;32", "\nDONE (in the order they ran)"))
    for i, (k, v) in enumerate(done, 1):
        st = v["status"]
        mark = C("32", "done  ") if st == "done" else C("31", f"{st:6}")
        note = " (superseded by a rerun)" if st == "failed" else ""
        print(f"{i:3}. {mark} {k:34} {about(k):38} {short(v.get('start')):>12} {hm(v.get('hours')):>6}{note}")
    print(C("1;33", "\nRUNNING NOW"))
    for k, v in running:
        el = (datetime.now() - datetime.fromisoformat(v["start"])).total_seconds() / 3600
        print(f"  >>  {k:34} {about(k):38} started {short(v['start'])}, {hm(el)} so far")
    if not running:
        print("  (nothing: the runner starts the next job within a minute)")
    print(C("1;34", "\nQUEUED (will run in this order)"))
    for i, k in enumerate(queued, 1):
        dep = after.get(k)
        wait = f"after {dep}" if dep and state.get(dep, {}).get("status") != "done" else ""
        print(f"{i:3}. {k:34} {about(k):38} {wait}")


if __name__ == "__main__":
    main()
