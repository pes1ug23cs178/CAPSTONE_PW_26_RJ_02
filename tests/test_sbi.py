import numpy as np
import pytest

from data.sbi import REGIONS, make_sbi, region_mask


def template_landmarks():
    """A plausible 68-point layout on a 224 crop (used when cached landmarks are not needed)."""
    lm = np.zeros((68, 2), np.float32)
    t = np.linspace(0, np.pi, 17)
    lm[0:17] = np.stack([112 - 90 * np.cos(t), 100 + 100 * np.sin(t)], 1)          # jaw
    lm[17:22] = np.stack([np.linspace(50, 100, 5), np.full(5, 70)], 1)              # brows
    lm[22:27] = np.stack([np.linspace(124, 174, 5), np.full(5, 70)], 1)
    lm[27:31] = np.stack([np.full(4, 112), np.linspace(90, 130, 4)], 1)             # nose bridge
    lm[31:36] = np.stack([np.linspace(95, 129, 5), np.full(5, 140)], 1)
    lm[36:42] = np.stack([60 + 18 * np.cos(np.linspace(0, 2 * np.pi, 6)), 90 + 6 * np.sin(np.linspace(0, 2 * np.pi, 6))], 1)
    lm[42:48] = lm[36:42] + [104, 0]
    a = np.linspace(0, 2 * np.pi, 20)
    lm[48:68] = np.stack([112 + 30 * np.cos(a), 175 + 12 * np.sin(a)], 1)        # mouth
    return lm


@pytest.mark.parametrize("seed", range(200))
def test_make_sbi_ranges(seed):
    rng = np.random.default_rng(seed)
    img = (np.random.default_rng(seed + 1000).random((224, 224, 3)) * 255).astype(np.uint8)
    out, concepts, boundary, recipe, mask = make_sbi(img, template_landmarks(), rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert concepts.shape == (10,) and (concepts >= 0).all() and (concepts <= 1).all()
    assert boundary.shape == (16, 16) and boundary.min() >= 0 and abs(boundary.max() - 1) < 1e-5
    assert mask.max() == pytest.approx(1.0) and (mask > 0.5).sum() > 50
    assert any(recipe[k] for k in ("colour", "blurred", "sharpened", "resampled", "geometry"))
    assert concepts[0] == recipe["blend_ratio"]
    assert concepts[5:8].sum() == (1 if recipe["region"] != "nose_cheeks" else 0)


@pytest.mark.parametrize("region", REGIONS)
def test_region_masks_non_empty(region):
    assert region_mask(template_landmarks(), region, 224).sum() > 100
