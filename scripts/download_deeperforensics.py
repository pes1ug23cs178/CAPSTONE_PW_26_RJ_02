"""DeeperForensics-1.0: download manipulated-video zip parts one at a time, keep only the requested subfolders
(default: end_to_end, the 1,000 standard fakes) and delete each zip after extraction. Stops once every video
of the requested lists is on disk.

python scripts/download_deeperforensics.py <drive_folder_id> [--keep end_to_end ...]

The Drive folder id comes from the approval email and is passed on the command line (it is not stored in git).
Output: deeperforensics/manipulated_videos/<subset>/*.mp4, log deeperforensics/download.log.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deeperforensics"
LOG = OUT / "download.log"


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wanted(keep):
    names = set()
    for k in keep:
        lst = OUT / "lists" / "manipulated_videos_lists" / f"manipulated_videos_{k}_list.txt"
        names |= {l.strip() for l in lst.read_text().splitlines() if l.strip()}
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_id")
    ap.add_argument("--keep", nargs="+", default=["end_to_end"])
    a = ap.parse_args()
    need = wanted(a.keep)
    files = gdown.download_folder(id=a.folder_id, skip_download=True, quiet=True)
    parts = sorted((f for f in files if "manipulated_videos_part_" in f.path), key=lambda f: f.path)
    for f in parts:
        missing = [n for n in need if not (OUT / n).exists()]
        if not missing:
            break
        name = Path(f.path.replace("\\", "/")).name
        zp = OUT / name
        for attempt in range(1, 6):
            log(f"download {name} (try {attempt}); {len(missing)} wanted videos still missing")
            try:
                gdown.download(id=f.id, output=str(zp), quiet=True, resume=True)
                zipfile.ZipFile(zp).testzip()
                break
            except Exception as e:  # quota limits and dropped connections
                log(f"  failed: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(600)
        else:
            log(f"GIVING UP on {name}")
            return
        z = zipfile.ZipFile(zp)
        members = [m for m in z.namelist() if m in need]
        for m in members:
            z.extract(m, OUT)
        z.close()
        zp.unlink()
        log(f"{name}: extracted {len(members)} wanted videos, zip deleted")
    left = [n for n in need if not (OUT / n).exists()]
    log(f"FINISHED: {len(need) - len(left)} / {len(need)} wanted videos on disk")


if __name__ == "__main__":
    main()
