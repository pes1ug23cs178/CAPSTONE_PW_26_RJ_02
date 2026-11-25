#!/usr/bin/env bash
# Full FF++ c23 download into ff++_full/ (videos for the real set and all five methods, then masks for the four
# methods that have them). Each dataset is retried up to 5 times; files already on disk are skipped.
# Log: ff++_full/download.log.   Needs faceforensics_download_v4.py (from the FF++ authors, not in git).
cd "$(dirname "$0")/.." || exit 1
LOG=ff++_full/download.log
mkdir -p ff++_full
run() {  # $1 dataset, $2 type
  for try in 1 2 3 4 5; do
    echo "$(date '+%F %T') start $1 $2 (try $try)" >> "$LOG"
    if echo "" | python faceforensics_download_v4.py ff++_full -d "$1" -c c23 -t "$2" --server EU2 >> "$LOG" 2>&1; then
      echo "$(date '+%F %T') done $1 $2" >> "$LOG"; return 0
    fi
    sleep 60
  done
  echo "$(date '+%F %T') FAILED $1 $2" >> "$LOG"
}
[ "$1" = "masks" ] || for d in original Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter; do run "$d" videos; done
for d in Deepfakes Face2Face FaceSwap NeuralTextures; do run "$d" masks; done
echo "$(date '+%F %T') ALL FINISHED" >> "$LOG"
