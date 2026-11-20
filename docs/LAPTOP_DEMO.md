# Running the GRACE app on a laptop with a public link

This is how we host the app: on a team member's laptop, with Gradio's free public link. (Hugging Face now needs a paid
PRO plan to host Gradio apps; the trained models are still stored there for free at
https://huggingface.co/grace-cap/grace-deepfake-models, and the laptop downloads them automatically.)

## One-time setup (about 15 minutes, needs internet)

1. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*) and **Git**.
2. Get the code:
   ```powershell
   git clone https://github.com/pes1ug23cs178/CAPSTONE_PW_26_RJ_02.git
   cd CAPSTONE_PW_26_RJ_02
   ```
3. Install everything (finds an NVIDIA GPU automatically; otherwise uses the CPU):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\laptop\setup_laptop.ps1
   ```

## Start the app

```powershell
powershell -ExecutionPolicy Bypass -File scripts\laptop\start_app.ps1
```

- The window prints **`Running on public URL: https://xxxx.gradio.live`**. Anyone can open that link from a phone or
  laptop while this window stays open. The link changes every time the app restarts and lasts up to a week.
- The laptop itself can also use http://127.0.0.1:7860.
- First start downloads the models (about 1.3 GB: our 84 MB release plus the CLIP backbone). Later starts take
  under a minute.
- To run without a public link (for example offline in the review room): add `-Local`.
- To stop: close the window or press Ctrl+C.

## Speed (measured 20 Nov 2026)

| Laptop | Careful check (teacher) | Quick check (student) |
| --- | --- | --- |
| NVIDIA GPU (4 GB or more) | a few seconds per video | a few seconds |
| CPU only | about 15–30 s per video on a fast CPU; up to about 1 minute on a slow one | about 5–15 s |

Through the public link the first check after a start took 78 s (model download included); later checks took
14–28 s on a CPU, with the correct verdict on every test file.

## Demo-day checklist

- [ ] Laptop charged, on power, sleep turned off (Settings → Power → Screen and sleep: *Never*).
- [ ] Start the app 15 minutes early; run one test video so the models are loaded.
- [ ] Put the new public link on the slide or share it in the room (it changes on every restart).
- [ ] Keep `-Local` mode ready in case the network is down.
- [ ] Test files ready: a fake and a real video from `cache/app_test_videos` (copy them to the laptop) and a photo
      without a face.
- [ ] A short screen recording of a full run, as a last resort.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `python` not found | Reinstall Python 3.12 with *Add to PATH*, open a new terminal |
| Scripts are blocked | Use the exact `powershell -ExecutionPolicy Bypass -File ...` commands above |
| No public URL appears | Check the internet connection; antivirus may block Gradio's tunnel program (`frpc`); allow it, or use `-Local` |
| Very slow Careful check | Normal on a CPU; use Quick check for live demos, or a laptop with an NVIDIA GPU |
| "We couldn't find a face" | Use a video where the face is clear and facing the camera |
| Anything else | Read the last lines in the app window; see `docs/DEPLOY.md` |
