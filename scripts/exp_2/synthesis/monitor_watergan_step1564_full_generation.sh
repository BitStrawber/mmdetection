#!/usr/bin/env bash
set -uo pipefail

# Continuously report throughput and status for the model-1564 full generation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_flat_results_48shards}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated_step1564}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_full_48shards}"
INTERVAL="${INTERVAL:-60}"
EXPECTED_TRAIN_FLAT="${EXPECTED_TRAIN_FLAT:-250048}"
EXPECTED_VAL_FLAT="${EXPECTED_VAL_FLAT:-10048}"
EXPECTED_TOTAL_FLAT=$((EXPECTED_TRAIN_FLAT + EXPECTED_VAL_FLAT))

count_fake() {
  find "$1" -type f -name 'fake_*.png' 2>/dev/null | wc -l | tr -d ' '
}

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

trap 'echo; echo "Monitoring stopped; generation was not interrupted."; exit 0' INT TERM

previous="$(count_fake "${FLAT_ROOT}")"
previous_time="$(date +%s)"
echo "Monitoring baseline: ${previous}/${EXPECTED_TOTAL_FLAT} flat images"
echo "First throughput sample will appear in ${INTERVAL} seconds."

while true; do
  sleep "${INTERVAL}"

  train_flat="$(count_fake "${FLAT_ROOT}/train")"
  val_flat="$(count_fake "${FLAT_ROOT}/val")"
  current=$((train_flat + val_flat))
  current_time="$(date +%s)"
  elapsed=$((current_time - previous_time))
  delta=$((current - previous))
  (( delta < 0 )) && delta=0
  remaining=$((EXPECTED_TOTAL_FLAT - current))
  (( remaining < 0 )) && remaining=0

  rate="$(awk -v n="${delta}" -v s="${elapsed}" \
    'BEGIN { printf "%.2f", s > 0 ? n / s : 0 }')"
  progress="$(awk -v n="${current}" -v total="${EXPECTED_TOTAL_FLAT}" \
    'BEGIN { printf "%.2f", total > 0 ? n / total * 100 : 0 }')"
  eta="$(awk -v remaining="${remaining}" -v rate="${rate}" '
    BEGIN {
      if (rate <= 0) { print "unknown"; exit }
      seconds = remaining / rate
      hours = int(seconds / 3600)
      minutes = int((seconds - hours * 3600) / 60)
      printf "%dh %02dm", hours, minutes
    }')"

  train_restored="$(count_images "${FINAL_ROOT}/train")"
  val_restored="$(count_images "${FINAL_ROOT}/val")"
  process_count="$(pgrep -fc '[m]ainmhl.py.*shard.*of48' 2>/dev/null || true)"

  [[ -t 1 ]] && printf '\033[H\033[2J'
  echo "============================================================"
  echo "WaterGAN model-1564 full generation monitor"
  echo "============================================================"
  date
  echo
  echo "===== Flat generation ====="
  printf 'train:    %8d / %8d\n' "${train_flat}" "${EXPECTED_TRAIN_FLAT}"
  printf 'val:      %8d / %8d\n' "${val_flat}" "${EXPECTED_VAL_FLAT}"
  printf 'total:    %8d / %8d  (%s%%)\n' \
    "${current}" "${EXPECTED_TOTAL_FLAT}" "${progress}"
  echo
  echo "===== Speed ====="
  printf 'last %ds:       +%d images\n' "${elapsed}" "${delta}"
  printf 'current speed:  %s images/s\n' "${rate}"
  printf 'per minute:     %.0f images/min\n' "$(awk -v r="${rate}" 'BEGIN { print r * 60 }')"
  printf 'remaining:      %d images\n' "${remaining}"
  printf 'estimated ETA:  %s\n' "${eta}"
  echo
  echo "===== Restored outputs ====="
  printf 'train:    %8d / 250000\n' "${train_restored}"
  printf 'val:      %8d /  10000\n' "${val_restored}"
  echo
  echo "===== Processes ====="
  if [[ -s "${LOG_ROOT}/launcher.pid" ]]; then
    launcher_pid="$(cat "${LOG_ROOT}/launcher.pid")"
    if ps -p "${launcher_pid}" >/dev/null 2>&1; then
      ps -ww -p "${launcher_pid}" -o pid,ppid,stat,etime,%cpu,%mem,args
    else
      echo "launcher PID ${launcher_pid}: NOT RUNNING"
    fi
  else
    echo "launcher PID file: MISSING"
  fi
  echo "active WaterGAN shard processes: ${process_count:-0}"
  echo
  echo "===== GPU 0-7 ====="
  nvidia-smi -i 0,1,2,3,4,5,6,7 \
    --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || true
  echo
  echo "===== Recent pipeline events ====="
  grep -aE \
    'reuse shard|started shard|finished shard|FAILED|complete|Error|Traceback' \
    "${LOG_ROOT}/launcher.log" 2>/dev/null | tail -n 20 || true
  echo
  echo "===== Recent errors ====="
  grep -RInaE \
    'Traceback|FAILED|Error|InvalidArgument|ResourceExhausted|CUDA out of memory|No such file' \
    "${LOG_ROOT}" 2>/dev/null | tail -n 10 || true
  echo
  echo "Press Ctrl+C to stop monitoring only."

  previous="${current}"
  previous_time="${current_time}"
done
