#!/usr/bin/env bash
# Face crops for the full FF++ download, following the download as it arrives: repeatedly runs
# scripts/extract_faces.py over every downloaded folder (finished videos are skipped) until the download log
# says ALL FINISHED and a final pass finds nothing new. Log: ff++_full/extract.log
# (A video still being written is never picked up: the downloader writes to a temp name and renames at the end.)
cd "$(dirname "$0")/.." || exit 1
LOG=ff++_full/extract.log
pass() {
  python scripts/extract_faces.py extract "ff++_full/original_sequences/youtube/c23/videos/*.mp4" ff++_full/faces/original >> "$LOG" 2>&1
  for m in Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter; do
    [ -d "ff++_full/manipulated_sequences/$m/c23/videos" ] && \
      python scripts/extract_faces.py extract "ff++_full/manipulated_sequences/$m/c23/videos/*.mp4" "ff++_full/faces/$m" >> "$LOG" 2>&1
  done
}
while true; do
  echo "$(date '+%F %T') pass" >> "$LOG"
  pass
  if grep -q "ALL FINISHED" ff++_full/download.log 2>/dev/null; then pass; echo "$(date '+%F %T') EXTRACTION FINISHED" >> "$LOG"; break; fi
  sleep 300
done
