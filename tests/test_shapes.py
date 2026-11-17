import torch

from models.build import build_model
from models.domain import compute_domain_alpha, grad_reverse, pairwise_mmd


def test_swin_shapes():
    model = build_model({"backbone": "swin_b"}, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 224, 224))
    assert out["logits2"].shape == (2, 2)
    assert out["logit"].shape == (2,)
    assert out["z"].shape == (2, 1024)


def test_swin_dann_shapes():
    model = build_model({"backbone": "swin_b", "domain": {"enabled": True, "num_domains": 4}}, pretrained=False)
    out = model(torch.rand(4, 3, 224, 224), alpha=0.5)
    assert out["domain_logits"].shape == (4, 4)


def test_grl_and_alpha():
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))
    assert compute_domain_alpha(0.0) == 0.0 and compute_domain_alpha(1.0) > 0.99


def test_mmd_zero_for_same_distribution():
    torch.manual_seed(0)
    f = torch.randn(16, 8)
    d = torch.tensor([0] * 8 + [1] * 8)
    same = pairwise_mmd(torch.cat([f[:8], f[:8]]), d)
    diff = pairwise_mmd(torch.cat([f[:8], f[8:] + 5]), d)
    assert same.item() < 1e-5 < diff.item()


def test_clip_lora_shapes_and_trainable_count():
    model = build_model({"backbone": "clip_vitl14_lora", "lora": {"r": 16, "alpha": 32, "dropout": 0.1}}, pretrained=False)
    n_lora = sum(p.numel() for p in model.backbone.lora_parameters())
    assert 1.5e6 < n_lora < 1.7e6
    assert all(not p.requires_grad for n, p in model.backbone.vit.named_parameters() if "lora_" not in n)
    with torch.no_grad():
        out = model(torch.rand(2, 3, 224, 224))
        feats = model.backbone(torch.rand(2, 3, 224, 224))
    assert out["logit"].shape == (2,) and out["z"].shape == (2, 1024)
    assert feats["patches"].shape == (2, 1024, 16, 16)
    assert torch.allclose(feats["cls"].norm(dim=1), torch.ones(2), atol=1e-4)
