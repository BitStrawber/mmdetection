#!/usr/bin/env bash
# Reserve idle GPU memory while yielding it automatically to matching training jobs.
#
# Example:
#   OCCUPY_MB=8000 TRAIN_MATCH='main_dino.py|run_dino_with_index.py' \
#     bash scripts/exp_2/tri_pretrain/run_gpu_yielding_occupier.sh '0,1,2,3,4,5,6,7'

set -euo pipefail

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
OCCUPY_MB="${OCCUPY_MB:-8000}"
RESERVE_MB="${RESERVE_MB:-2500}"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
IDLE_CHECKS="${IDLE_CHECKS:-1}"
LOG_DIR="${LOG_DIR:-logs/gpu_yielding_occupier_$(date +%Y%m%d_%H%M%S)}"

# A process whose command line matches this expression gets priority over the
# occupier. Override this when protecting a different training entry point.
TRAIN_MATCH="${TRAIN_MATCH:-main_dino.py|run_dino_with_index.py|tools/train.py|torchrun}"

mkdir -p "$LOG_DIR"

declare -A WORKER_PIDS=()
declare -A IDLE_COUNTS=()

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/controller.log"
}

stop_worker() {
  local gpu="$1"
  local pid="${WORKER_PIDS[$gpu]:-}"

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "GPU $gpu: release ${OCCUPY_MB} MiB (worker PID $pid)."
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  unset 'WORKER_PIDS[$gpu]'
}

stop_all_workers() {
  local gpu
  for gpu in "${GPU_LIST[@]}"; do
    stop_worker "$gpu"
  done
}

cleanup() {
  log 'Stopping yielding occupier.'
  stop_all_workers
}
trap cleanup EXIT INT TERM

is_training_on_gpu() {
  local gpu="$1"
  local pid command

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command="$(ps -ww -p "$pid" -o args= 2>/dev/null || true)"
    [[ -z "$command" ]] && continue
    if [[ "$command" =~ $TRAIN_MATCH ]]; then
      return 0
    fi
  done < <(
    nvidia-smi --id="$gpu" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )

  return 1
}

gpu_free_mb() {
  nvidia-smi --id="$1" \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null |
    tr -dc '0-9'
}

start_worker() {
  local gpu="$1"
  local free_mb="$2"
  local pid

  if [[ -n "${WORKER_PIDS[$gpu]:-}" ]] && kill -0 "${WORKER_PIDS[$gpu]}" 2>/dev/null; then
    return
  fi

  log "GPU $gpu: reserve ${OCCUPY_MB} MiB (free before reservation: ${free_mb} MiB)."
  OCCUPY_MB="$OCCUPY_MB" CUDA_VISIBLE_DEVICES="$gpu" nohup python -c '
import os
import signal
import time
import torch

target_mb = int(os.environ["OCCUPY_MB"])
num_bytes = target_mb * 1024 * 1024
tensor = torch.empty(num_bytes // 4, dtype=torch.float32, device="cuda")
tensor.zero_()  # Materialize the allocation instead of relying on lazy pages.
print(f"yielding GPU occupier ready: {target_mb} MiB", flush=True)

def stop_handler(signum, frame):
    del tensor
    torch.cuda.empty_cache()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)
while True:
    time.sleep(60)
' > "$LOG_DIR/gpu_${gpu}.log" 2>&1 &
  pid=$!
  WORKER_PIDS[$gpu]="$pid"
}

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
for gpu in "${GPU_LIST[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU id: $gpu" >&2
    exit 2
  }
  IDLE_COUNTS[$gpu]=0
done

log "Yielding GPU occupier started. GPUs=$GPU_IDS occupy=${OCCUPY_MB}MiB reserve=${RESERVE_MB}MiB interval=${CHECK_INTERVAL}s idle_checks=${IDLE_CHECKS}."
log "Training priority regex: $TRAIN_MATCH"
log "Logs: $LOG_DIR"

while true; do
  for gpu in "${GPU_LIST[@]}"; do
    free_mb="$(gpu_free_mb "$gpu")"
    free_mb="${free_mb:-0}"

    if is_training_on_gpu "$gpu"; then
      IDLE_COUNTS[$gpu]=0
      stop_worker "$gpu"
      continue
    fi

    # Do not start another allocation unless there is enough room for both
    # the reservation and a safety margin.
    if (( free_mb < OCCUPY_MB + RESERVE_MB )); then
      IDLE_COUNTS[$gpu]=0
      stop_worker "$gpu"
      continue
    fi

    IDLE_COUNTS[$gpu]=$(( IDLE_COUNTS[$gpu] + 1 ))
    if (( IDLE_COUNTS[$gpu] >= IDLE_CHECKS )); then
      start_worker "$gpu" "$free_mb"
    fi
  done

  sleep "$CHECK_INTERVAL"
done
