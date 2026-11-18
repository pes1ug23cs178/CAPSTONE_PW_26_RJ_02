import numpy as np
import pytest

from utils.config import REPO_ROOT

pipeline = pytest.importorskip("app.pipeline")

MEAN = REPO_ROOT / "cache" / "landmarks" / "mean_landmarks.npy"


@pytest.mark.skipif(not MEAN.exists(), reason="mean landmarks not cached")
def test_frontal_mean_face_is_reliable_and_pose_is_rotation_invariant():
    lm = np.load(MEAN).reshape(-1, 2)[:68]
    yaw, pitch = pipeline.head_pose(lm)
    assert abs(yaw) < 0.05 and pipeline.pose_ok(yaw, pitch)
    th = np.radians(25)
    rot = (lm - lm.mean(0)) @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T * 1.7
    y2, p2 = pipeline.head_pose(rot)
    assert abs(y2 - yaw) < 1e-6 and abs(p2 - pitch) < 1e-6


@pytest.mark.skipif(not MEAN.exists(), reason="mean landmarks not cached")
def test_turned_head_is_flagged():
    lm = np.load(MEAN).reshape(-1, 2)[:68].copy()
    lm[30, 0] = lm[0, 0] + 0.03 * (lm[16, 0] - lm[0, 0])  # nose tip at the jaw edge = near-profile turn
    assert not pipeline.pose_ok(*pipeline.head_pose(lm))
