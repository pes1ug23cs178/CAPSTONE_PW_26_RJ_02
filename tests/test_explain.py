import numpy as np
import torch

from explain.maps import attention_rollout, concept_maps, contributions, gradcam, intervene
from models.build import build_model

LORA = {"r": 4, "alpha": 8, "dropout": 0.0}


def in_range(m, shape):
    assert m.shape == shape and m.min() >= -1e-6 and m.max() <= 1 + 1e-6


def test_teacher_concept_maps_contributions_intervene():
    m = build_model({"backbone": "clip_vitl14_lora", "lora": LORA, "head": "concept"}, pretrained=False).eval()
    with torch.no_grad():
        m.decision.r[-1].weight.normal_()  # non-zero residual
    x = torch.rand(1, 3, 224, 224)
    in_range(concept_maps(m, x), (10, 224, 224))
    c = contributions(m, x)
    assert c["contributions"].shape == (10,)
    assert abs(c["contributions"].sum() + c["residual"] + c["bias"] - c["logit"]) < 1e-3
    all_off = intervene(m, x, list(range(10)))
    assert abs(all_off - (c["bias"] + c["residual"])) < 1e-3  # only b + r(z) remains


def test_rollout_and_student_gradcampp():
    c1 = build_model({"backbone": "clip_vitl14_lora", "lora": LORA}, pretrained=False).eval()
    in_range(attention_rollout(c1, torch.rand(1, 3, 224, 224)), (224, 224))
    st = build_model({"backbone": "efficientnet_b0"}, pretrained=False).eval()
    in_range(gradcam(st, torch.rand(1, 3, 224, 224)), (224, 224))


def test_swin_gradcam():
    sw = build_model({"backbone": "swin_b"}, pretrained=False).eval()
    in_range(gradcam(sw, torch.rand(1, 3, 224, 224)), (224, 224))
