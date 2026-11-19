# App check (T6.4)

Test videos are built from test-split face crops (see `scripts/make_app_test_videos.py`); file names give the true label.

| File | Mode | Verdict | Probability | Faces | Flicker | Drift | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fake_cdf_id39_id42_0000.mp4 | Quick | Likely fake | 0.961 | 32/32 | 0.042 | None | 2.9 |
| fake_dfd_fake_21_09__outside_talking_still_laughing__O9ER29Z6.mp4 | Quick | Likely fake | 0.961 | 32/32 | 0.09 | None | 2.9 |
| fake_ffpp_Deepfakes_233_995.mp4 | Quick | Likely fake | 0.933 | 32/32 | 0.143 | None | 3.2 |
| real_cdf_id34_0000.mp4 | Quick | Likely fake | 0.907 | 32/32 | 0.104 | None | 4.1 |
| real_ffpp_220.mp4 | Quick | Inconclusive | 0.531 | 32/32 | 0.14 | None | 4.6 |
| fake_cdf_id39_id42_0000.mp4 | Deep | Likely fake | 0.925 | 32/32 | 0.107 | 0.149 | 4.5 |
| fake_dfd_fake_21_09__outside_talking_still_laughing__O9ER29Z6.mp4 | Deep | Likely fake | 0.900 | 32/32 | 0.164 | 0.232 | 3.9 |
| fake_ffpp_Deepfakes_233_995.mp4 | Deep | Likely fake | 0.922 | 32/32 | 0.185 | 0.209 | 4.2 |
| real_cdf_id34_0000.mp4 | Deep | Likely real | 0.127 | 32/32 | 0.049 | 0.24 | 4.2 |
| real_ffpp_220.mp4 | Deep | Likely real | 0.134 | 32/32 | 0.045 | 0.209 | 4.4 |
| no_face.jpg | Quick | No face found in this image. No verdict given. | | 0 | | | |
