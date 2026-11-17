import numpy as np
import pandas as pd
import torch

from data.teacher_cache import TeacherCache
from models.build import build_model
from models.losses import compute_loss
from utils.config import load_config


def fake_batch(b=4):
    return {"image": torch.rand(b, 3, 224, 224), "label": torch.tensor([0., 1., 1., 0.]),
            "concepts": torch.rand(b, 10), "concept_mask": torch.ones(b, 10), "boundary": torch.rand(b, 16, 16),
            "boundary_mask": torch.ones(b), "domain_id": torch.zeros(b, dtype=torch.long),
            "teacher_logit": torch.randn(b), "teacher_z": torch.randn(b, 512), "teacher_concepts": torch.randn(b, 10),
            "teacher_maps": torch.randn(b, 10, 16, 16)}


def test_student_losses_per_config():
    expected = {"s0_student": {"bce", "concept", "map"}, "s2_kd_logits": {"bce", "kd_kl"},
                "s3_kd_full": {"bce", "kd_kl", "kd_feat", "kd_concept", "kd_map"}}
    for name, terms in expected.items():
        cfg = load_config(f"configs/{name}.yaml")
        m = build_model(cfg["model"], pretrained=False)
        b = fake_batch()
        if name == "s0_student":
            b = {k: v for k, v in b.items() if not k.startswith("teacher_")}
        loss, logs = compute_loss(m(b["image"]), b, cfg)
        assert torch.isfinite(loss) and terms <= set(logs), (name, logs)
        if name != "s3_kd_full":
            assert "kd_feat" not in logs
        loss.backward()


def test_teacher_cache_lookup(tmp_path):
    n = 5
    np.savez(tmp_path / "shard_00.npz", logit=np.arange(n, dtype=np.float32), z=np.zeros((n, 512), np.float16),
             concepts=np.zeros((n, 10), np.float32), maps=np.zeros((n, 10, 16, 16), np.float16))
    pd.DataFrame({"frame_path": [f"p{i}" for i in range(n)], "shard": 0, "row": range(n)}).to_csv(tmp_path / "index.csv", index=False)
    t = TeacherCache(tmp_path).lookup(["p3", "p1"], torch.device("cpu"))
    assert t["teacher_logit"].tolist() == [3.0, 1.0] and t["teacher_maps"].shape == (2, 10, 16, 16)
