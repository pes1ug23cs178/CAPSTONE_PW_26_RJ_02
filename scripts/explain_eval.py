"""T6.3: explanation metrics -> results/explain_metrics.csv (one row per method, metric).

Pointing game / mask IoU on cache/sbi_test (ground-truth blend masks):
  G3 c1 map, G3 region map (c6 mouth / c7 eyes / c8 full face by the true region; nose_cheeks has no
  region concept and is skipped for that method), B1 Grad-CAM, C1 attention rollout, S3 c1 map; random map floor.
Deletion (lower better) / insertion (higher better) AUC on 500 correctly flagged test fakes:
  14x14 patches replaced by a blurred copy in order of importance, 20 steps; each method scored with its own model.
Intervention effect (G3, S3): mean drop in fake probability, top concept off vs a random other concept.
Concept AUC (G3, S0, S3) on sbi_test; concept share (G1, G2, G3, S3) from the test metrics files.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from evaluate import concept_auc, load_model_from_ckpt  # noqa: E402
from explain.maps import attention_rollout, concept_maps, gradcam  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402

REGION_CONCEPT = {"mouth": 5, "eyes_brows": 6, "full_face": 7}
PATCH, STEPS = 14, 20


def to_x(img, device):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)[None].to(device)


def method_map(name, models, img, device, region=None):
    x = to_x(img, device)
    if name == "G3 c1":
        return concept_maps(models["g3"], x)[0]
    if name == "G3 region":
        return concept_maps(models["g3"], x)[REGION_CONCEPT[region]]
    if name == "B1 Grad-CAM":
        return gradcam(models["b1"], x)
    if name == "C1 rollout":
        return attention_rollout(models["c1"], x)
    if name == "S3 c1":
        return concept_maps(models["s3"], x)[0]
    if name == "random":
        return np.random.default_rng(abs(hash(img.tobytes())) % 2**32).random((224, 224))
    raise ValueError(name)


METHOD_MODEL = {"G3 c1": "g3", "G3 region": "g3", "B1 Grad-CAM": "b1", "C1 rollout": "c1", "S3 c1": "s3", "random": "g3"}


@torch.no_grad()
def fake_prob(model, imgs, device):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(x)["logit"].float()).cpu().numpy()


def del_ins(model, img, m, device):
    blurred = cv2.GaussianBlur(img, (0, 0), 8)
    g = 224 // PATCH
    imp = m[: g * PATCH, : g * PATCH].reshape(g, PATCH, g, PATCH).mean(axis=(1, 3)).ravel()
    order = np.argsort(-imp)
    per_step = int(np.ceil(len(order) / STEPS))
    dels, ins = [img.copy()], [blurred.copy()]
    cur_d, cur_i = img.copy(), blurred.copy()
    for s in range(STEPS):
        for p in order[s * per_step:(s + 1) * per_step]:
            r, c = divmod(int(p), g)
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            cur_d[sl] = blurred[sl]
            cur_i[sl] = img[sl]
        dels.append(cur_d.copy()); ins.append(cur_i.copy())
    pd_, pi = fake_prob(model, dels, device), fake_prob(model, ins, device)
    return float(np.trapezoid(pd_, dx=1 / STEPS)), float(np.trapezoid(pi, dx=1 / STEPS))


def main() -> None:
    device = torch.device("cuda")
    root = load_paths()["data_root"]
    names = {"g3": "g3_full", "b1": "b1_swin", "c1": "c1_clip_lora", "s3": "s3_kd_full", "s0": "s0_student"}
    runs = {k: resolve_runs([v], [42])[0] for k, v in names.items()}
    models = {k: load_model_from_ckpt(str(r / "best.pth"), device)[0] for k, r in runs.items() if k != "s0"}
    rows = []
    methods = ["G3 c1", "G3 region", "B1 Grad-CAM", "C1 rollout", "S3 c1", "random"]

    # pointing game + mask IoU
    sbi = pd.read_csv(REPO_ROOT / "cache" / "sbi_test" / "sbi_test.csv")
    for meth in methods:
        hits, ious = [], []
        for r in sbi.itertuples():
            if meth == "G3 region" and r.region not in REGION_CONCEPT:
                continue
            img = load_rgb(root / r.frame_path)
            gt = np.load(root / r.mask_path).astype(np.float32) > 0.5
            m = method_map(meth, models, img, device, r.region)
            y, x = np.unravel_index(np.argmax(m), m.shape)
            hits.append(bool(gt[y, x]))
            top = m >= np.quantile(m, 0.9)
            ious.append((top & gt).sum() / max((top | gt).sum(), 1))
        rows += [{"method": meth, "metric": "pointing_game", "value": float(np.mean(hits)), "n": len(hits)},
                 {"method": meth, "metric": "mask_iou", "value": float(np.mean(ious)), "n": len(ious)}]
        print(rows[-2:], flush=True)

    # deletion / insertion on 500 correctly flagged fakes (G3 threshold, mixed test splits)
    thr = json.loads((runs["g3"] / "val_threshold.json").read_text())["threshold"]
    preds = pd.concat([pd.read_csv(runs["g3"] / f"predictions_{s}.csv") for s in ("ffpp_test", "dfd_test", "celebdf_test")])
    tp = preds[(preds.label == 1) & (preds.prob >= thr)].sample(n=500, random_state=0)
    for meth in methods:
        if meth == "G3 region":
            continue
        model = models[METHOD_MODEL[meth]]
        d, i = [], []
        for r in tp.itertuples():
            img = load_rgb(root / r.frame_path)
            dd, ii = del_ins(model, img, method_map(meth, models, img, device), device)
            d.append(dd); i.append(ii)
        rows += [{"method": meth, "metric": "deletion_auc", "value": float(np.mean(d)), "n": len(d)},
                 {"method": meth, "metric": "insertion_auc", "value": float(np.mean(i)), "n": len(i)}]
        print(rows[-2:], flush=True)

    # intervention effect
    rng = random.Random(0)
    for key in ("g3", "s3"):
        model = models[key]
        top_drop, rand_drop = [], []
        for r in tp.itertuples():
            x = to_x(load_rgb(root / r.frame_path), device)
            with torch.no_grad():
                o = model(x)
                p0 = torch.sigmoid(o["logit"].float())
                top = int(o["contributions"][0].argmax())
                other = rng.choice([k for k in range(10) if k != top])
                for k, store in ((top, top_drop), (other, rand_drop)):
                    off = torch.zeros(10, device=device); off[k] = 1
                    store.append(float(p0 - torch.sigmoid(model(x, off=off)["logit"].float())))
        rows += [{"method": key.upper(), "metric": "intervention_drop_top", "value": float(np.mean(top_drop)), "n": len(top_drop)},
                 {"method": key.upper(), "metric": "intervention_drop_random", "value": float(np.mean(rand_drop)), "n": len(rand_drop)}]

    # concept AUC (sbi_test) and concept share (test splits)
    s0_model = load_model_from_ckpt(str(runs["s0"] / "best.pth"), device)[0]
    for key, model in (("G3", models["g3"]), ("S0", s0_model), ("S3", models["s3"])):
        for c, v in concept_auc(model, REPO_ROOT / "cache" / "sbi_test" / "sbi_test.csv", device).items():
            rows.append({"method": key, "metric": f"concept_auc_{c}", "value": v, "n": 2000})
    for name in ("g1_concepts", "g2_dct", "g2_fft", "g3_full", "s3_kd_full"):
        found = resolve_runs([name], [42])
        if not found:
            continue
        for split in ("ffpp_test", "dfd_test", "celebdf_test"):
            m = json.loads((found[0] / f"metrics_{split}.json").read_text())
            rows.append({"method": name, "metric": f"concept_share_{split}", "value": m.get("concept_share"), "n": m["n_frames"]})
    out = REPO_ROOT / "results" / "explain_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
