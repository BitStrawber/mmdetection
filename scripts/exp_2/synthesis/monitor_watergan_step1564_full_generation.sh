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

count_zero_fake() {
  find "$1" -type f -name 'fake_*.png' -size 0 2>/dev/null | wc -l | tr -d ' '
}

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

trap 'echo; echo "Monitoring stopped; generation was not interrupted."; exit 0' INT TERM

previous_total="$(count_fake "${FLAT_ROOT}")"
previous_zero="$(count_zero_fake "${FLAT_ROOT}")"
previous=$((previous_total - previous_zero))
previous_time="$(date +%s)"
echo "Monitoring baseline: ${previous}/${EXPECTED_TOTAL_FLAT} valid flat images"
echo "First throughput sample will appear in ${INTERVAL} seconds."

while true; do
  sleep "${INTERVAL}"

  train_total="$(count_fake "${FLAT_ROOT}/train")"
  val_total="$(count_fake "${FLAT_ROOT}/val")"
  train_zero="$(count_zero_fake "${FLAT_ROOT}/train")"
  val_zero="$(count_zero_fake "${FLAT_ROOT}/val")"
  train_flat=$((train_total - train_zero))
  val_flat=$((val_total - val_zero))
  current=$((train_flat + val_flat))
  current_time="$(date +%s)"
  elapsed=$((current_time - previous_time))
  delta=$((current - previous))
  (( delta < 0 )) && delta=0
  remaining=$((EXPECTED_TOTAL_FLAT - current))
  (( remaining < 0 )) && remaining=0

  rate_scaled=0
  per_minute=0
  if (( elapsed > 0 )); then
    rate_scaled=$((delta * 100 / elapsed))
    per_minute=$((delta * 60 / elapsed))
  fi
  printf -v rate '%d.%02d' $((rate_scaled / 100)) $((rate_scaled % 100))

  progress_scaled=0
  if (( EXPECTED_TOTAL_FLAT > 0 )); then
    progress_scaled=$((current * 10000 / EXPECTED_TOTAL_FLAT))
  fi
  printf -v progress '%d.%02d' \
    $((progress_scaled / 100)) $((progress_scaled % 100))

  eta="unknown"
  if (( delta > 0 )); then
    eta_seconds=$((remaining * elapsed / delta))
    eta_hours=$((eta_seconds / 3600))
    eta_minutes=$(((eta_seconds % 3600) / 60))
    printf -v eta '%dh %02dm' "${eta_hours}" "${eta_minutes}"
  fi

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
  printf 'train:    %8d / %8d valid  (total=%d, zero=%d)\n' \
    "${train_flat}" "${EXPECTED_TRAIN_FLAT}" "${train_total}" "${train_zero}"
  printf 'val:      %8d / %8d valid  (total=%d, zero=%d)\n' \
    "${val_flat}" "${EXPECTED_VAL_FLAT}" "${val_total}" "${val_zero}"
  printf 'total:    %8d / %8d  (%s%%)\n' \
    "${current}" "${EXPECTED_TOTAL_FLAT}" "${progress}"
  echo
  echo "===== Speed ====="
  printf 'last %ds:       +%d images\n' "${elapsed}" "${delta}"
  printf 'current speed:  %s images/s\n' "${rate}"
  printf 'per minute:     %d images/min\n' "${per_minute}"
  printf 'remaining:      %d images\n' "${remaining}"
  printf 'estimated ETA:  %s\n' "${eta}"
  echo
  echo "===== Restored outputs ====="
  printf 'train:    %8d / 250000\n' "${train_restored}"
  printf 'val:      %8d /  10000\n' "${val_restored}"
  echo
  echo "===== Processes ====="
  found_pid_file=0
  for split in train val; do
    pid_file="${LOG_ROOT}/launcher_${split}.pid"
    if [[ -s "${pid_file}" ]]; then
      found_pid_file=1
      launcher_pid="$(cat "${pid_file}")"
      if ps -p "${launcher_pid}" >/dev/null 2>&1; then
        printf '%s launcher:\n' "${split}"
        ps -ww -p "${launcher_pid}" -o pid,ppid,stat,etime,%cpu,%mem,args
      else
        echo "${split} launcher PID ${launcher_pid}: NOT RUNNING"
      fi
    fi
  done
  if [[ "${found_pid_file}" == 0 ]]; then
    if [[ -s "${LOG_ROOT}/launcher.pid" ]]; then
      launcher_pid="$(cat "${LOG_ROOT}/launcher.pid")"
      if ps -p "${launcher_pid}" >/dev/null 2>&1; then
        ps -ww -p "${launcher_pid}" -o pid,ppid,stat,etime,%cpu,%mem,args
      else
        echo "launcher PID ${launcher_pid}: NOT RUNNING"
      fi
    else
      echo "launcher PID files: MISSING"
    fi
  fi
  echo "active WaterGAN shard processes: ${process_count:-0}"
  echo
  echo "===== GPU 0-7 ====="
  nvidia-smi -i 0,1,2,3,4,5,6,7 \
    --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || true
  echo
  echo "===== Recent pipeline events ====="
  launcher_logs=()
  for split in train val; do
    latest_log="$(
      find "${LOG_ROOT}" -maxdepth 1 -type f \
        -name "launcher_${split}*.log" -printf '%T@|%p\n' 2>/dev/null \
        | sort -t '|' -k1,1nr | head -n 1 | cut -d '|' -f 2-
    )"
    [[ -n "${latest_log}" ]] && launcher_logs+=("${latest_log}")
  done
  if [[ "${#launcher_logs[@]}" -gt 0 ]]; then
    grep -ahE \
      'reuse |dispatch |started |finished |completed |FAILED|complete|Error|Traceback' \
      "${launcher_logs[@]}" 2>/dev/null | tail -n 20 || true
  fi
  echo
  echo "===== Recent errors ====="
  if [[ "${#launcher_logs[@]}" -gt 0 ]]; then
    grep -aHinE \
      'Traceback|FAILED|Error|InvalidArgument|ResourceExhausted|CUDA out of memory|No such file' \
      "${launcher_logs[@]}" 2>/dev/null | tail -n 10 || true
  fi
  echo
  echo "Press Ctrl+C to stop monitoring only."

  previous="${current}"
  previous_time="${current_time}"
done
