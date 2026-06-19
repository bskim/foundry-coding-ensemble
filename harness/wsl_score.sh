#!/usr/bin/env bash
# Score SWE-bench predictions with the official Docker harness inside WSL.
# Usage: wsl_score.sh <run_id> <preds_file1> [preds_file2 ...]
# preds files are absolute paths (e.g. /mnt/c/.../preds_gpt-5.4.jsonl).
set -u
VENV=/opt/swebench-venv/bin/python
DATASET=princeton-nlp/SWE-bench_Verified
WORK=/root/sweval
mkdir -p "$WORK"

# --- ensure docker daemon ---
if ! docker info >/dev/null 2>&1; then
  echo "== starting docker daemon =="
  service docker start >/dev/null 2>&1 || nohup dockerd >/var/log/dockerd.log 2>&1 &
  for i in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
docker info >/dev/null 2>&1 && echo "docker: UP" || { echo "docker: DOWN"; exit 1; }

RUN_ID="$1"; shift
# Reports are copied next to the prediction files.
EVALDIR=$(cd "$(dirname "$1")" && pwd)
cd "$WORK"
for PREDS in "$@"; do
  MODEL=$(basename "$PREDS" .jsonl)
  echo "===================================================="
  echo "scoring $PREDS  (run_id=${RUN_ID})"
  echo "===================================================="
  "$VENV" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" \
    --predictions_path "$PREDS" \
    --max_workers 2 \
    --cache_level instance \
    --run_id "${RUN_ID}" 2>&1 | tail -n 40
  # copy any produced report json back to the eval dir
  for rep in *."${RUN_ID}".json; do
    [ -f "$rep" ] && cp -f "$rep" "$EVALDIR/report_$rep" && echo "copied report -> report_$rep"
  done
done
echo "SCORE_DONE"
