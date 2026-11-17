import numpy as np
import pandas as pd

from explain.video_cues import identity_drift, temporal_flicker, video_cues


def test_constant_sequence_has_zero_flicker_and_drift():
    assert abs(temporal_flicker(np.tile(np.random.rand(1, 10), (8, 1)))) < 1e-9
    assert abs(identity_drift(np.tile(np.random.rand(1, 32), (8, 1)))) < 1e-6


def test_changing_sequence_positive():
    rng = np.random.default_rng(0)
    assert temporal_flicker(rng.random((8, 10))) > 0.1
    assert identity_drift(rng.standard_normal((8, 32))) > 0.5


def test_video_cues_table():
    pred = pd.DataFrame({"video_id": ["a"] * 3 + ["b"] * 2, "frame_path": ["a/2", "a/1", "a/3", "b/1", "b/2"],
                         "label": [1, 1, 1, 0, 0], "prob": [0.9, 0.8, 0.7, 0.1, 0.2],
                         **{f"c{k}": [0.5] * 5 for k in range(1, 11)}})
    t = video_cues(pred)
    assert list(t["video_id"]) == ["a", "b"] and t.loc[0, "frame_scores"] == "0.8000;0.9000;0.7000"
