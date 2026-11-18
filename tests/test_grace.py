import torch

from models.build import build_model
from models.frequency import DCTBranch, block_dct, dct_matrix
from models.losses import compute_loss, masked_bce

BASE = {"backbone": "clip_vitl14_lora", "lora": {"r": 4, "alpha": 8, "dropout": 0.0}}


def batch(b=4):
    return {
        "image": torch.rand(b, 3, 224, 224), "label": torch.tensor([0., 1., 1., 0.])[:b],
        "concepts": torch.rand(b, 10), "concept_mask": torch.tensor([[1.] * 10, [1.] * 10, [0.] * 10, [1.] * 10])[:b],
        "boundary": torch.rand(b, 16, 16), "boundary_mask": torch.tensor([1., 1., 0., 1.])[:b],
        "domain_id": torch.tensor([0, 4, 1, 0])[:b],
    }


def test_dct_constant_image_only_dc():
    y = torch.full((1, 1, 224, 224), 100.0)
    coef = block_dct(y, dct_matrix(8))
    assert coef.shape == (1, 64, 28, 28)
    assert torch.allclose(coef[:, 0], torch.full_like(coef[:, 0], 800.0), atol=1e-3)  # 8 * mean for orthonormal DCT
    assert coef[:, 1:].abs().max() < 1e-3


def test_dct_is_orthonormal():
    d = dct_matrix(8)
    assert torch.allclose(d @ d.T, torch.eye(8), atol=1e-5)


def test_masked_bce_empty_mask_is_zero():
    assert masked_bce(torch.randn(3, 10), torch.rand(3, 10), torch.zeros(3, 10)).item() == 0.0


def run_variant(extra, expect):
    cfg_model = {**BASE, **extra}
    m = build_model(cfg_model, pretrained=False)
    b = batch()
    out = m(b["image"], alpha=0.5)
    for k, shape in expect.items():
        assert tuple(out[k].shape) == shape, (k, out[k].shape)
    loss, logs = compute_loss(out, b, {"model": cfg_model, "loss": {}})
    assert torch.isfinite(loss)
    loss.backward()
    return logs


def test_g1_concepts():
    logs = run_variant({"head": "concept"}, {"logit": (4,), "z": (4, 512), "concepts": (4, 10), "maps": (4, 10, 16, 16)})
    assert {"bce", "concept", "map", "w_l1", "residual"} <= set(logs)


def test_g2_dct_and_fft_gated():
    for fb in ("dct", "fft"):
        run_variant({"head": "concept", "freq_branch": fb, "fusion": "gated"}, {"logit": (4,), "z": (4, 512), "gate": (4,)})


def test_g3_domain_and_a3_linear():
    dom = {"enabled": True, "num_domains": 5, "dann_weight": 0.1, "mmd_weight": 0.1, "mmd_on": "fake"}
    logs = run_variant({"head": "concept", "freq_branch": "dct", "fusion": "gated", "domain": dom}, {"domain_logits": (4, 5)})
    assert "domain_ce" in logs
    logs = run_variant({"head": "linear", "freq_branch": "dct", "fusion": "gated", "domain": dom}, {"logit": (4,)})
    assert "concept" not in logs


def test_decision_starts_concept_only_and_intervention():
    m = build_model({**BASE, "head": "concept"}, pretrained=False).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
        assert torch.allclose(out["residual"], torch.zeros(2))
        off = m(x, off=torch.ones(10))
    assert torch.allclose(off["logit"], m.decision.b + off["residual"], atol=1e-5)


def test_unlabelled_rows_give_no_class_loss():
    """D2: rows with label_mask 0 must not affect the real/fake loss, whatever their label."""
    out = {"logit": torch.tensor([2.0, -1.0, 0.5, 0.3])}
    cfg = {"model": {}, "loss": {}}
    b1 = {"label": torch.tensor([1., 0., 0., 0.]), "label_mask": torch.tensor([1., 1., 0., 0.])}
    b2 = {"label": torch.tensor([1., 0., 1., 1.]), "label_mask": torch.tensor([1., 1., 0., 0.])}
    l1, _ = compute_loss(out, b1, cfg)
    l2, _ = compute_loss(out, b2, cfg)
    assert torch.allclose(l1, l2)


def test_domain_on_patch_features():
    """F2: the domain head reads mean-pooled CLIP patches (1024-d), and MMD uses the same features."""
    dom = {"enabled": True, "input": "patches", "num_domains": 5, "dann_weight": 0.1, "mmd_weight": 0.1,
           "mmd_on": "fake", "domain_on": "fake"}
    cfg_model = {**BASE, "head": "concept", "freq_branch": "dct", "fusion": "gated", "domain": dom}
    m = build_model(cfg_model, pretrained=False)
    out = m(torch.rand(4, 3, 224, 224), alpha=0.5)
    assert out["domain_feat"].shape == (4, 1024) and out["domain_logits"].shape == (4, 5)
    logs = run_variant({"head": "concept", "freq_branch": "dct", "fusion": "gated", "domain": dom}, {"domain_logits": (4, 5)})
    assert "domain_ce" in logs
