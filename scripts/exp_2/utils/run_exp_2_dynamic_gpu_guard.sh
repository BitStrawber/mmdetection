#!/bin/bash
set -euo pipefail

# Dynamic GPU memory guard.
#
# Usage:
#   bash scripts/exp_2/utils/run_exp_2_dynamic_gpu_guard.sh "0,1" 5000
#
# Arguments:
#   $1: GPU ids, comma-separated. Example: "0,1,4"
#   $2: free memory reserve per GPU, in MB. Example: 5000
#
# The guard keeps each selected GPU close to RESERVE_FREE_MB free memory:
# - if free memory is much higher than reserve, it allocates extra guard tensors;
# - if free memory drops below reserve, it releases guard tensors.
#
# This is intended to discourage other processes from entering the GPU while
# still leaving a configurable safety buffer for the real training process.

GPU_IDS="${1:-}"
RESERVE_FREE_MB="${2:-}"

CHECK_INTERVAL=${CHECK_INTERVAL:-1.0}
CHUNK_MB=${CHUNK_MB:-256}
HYSTERESIS_MB=${HYSTERESIS_MB:-512}
MAX_GUARD_MB=${MAX_GUARD_MB:-0}
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE=${LOG_FILE:-}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}

usage() {
    echo "Usage: bash scripts/exp_2/utils/run_exp_2_dynamic_gpu_guard.sh \"GPU_IDS\" RESERVE_FREE_MB"
    echo "Example: bash scripts/exp_2/utils/run_exp_2_dynamic_gpu_guard.sh \"0,1\" 5000"
    echo ""
    echo "Optional env:"
    echo "  CHECK_INTERVAL=1.0   monitor interval in seconds"
    echo "  CHUNK_MB=256         allocation/release granularity"
    echo "  HYSTERESIS_MB=512    dead band around reserve"
    echo "  MAX_GUARD_MB=0       max guard allocation per GPU, 0 means unlimited"
}

if [ -z "$GPU_IDS" ] || [ -z "$RESERVE_FREE_MB" ]; then
    usage
    exit 1
fi

if ! [[ "$RESERVE_FREE_MB" =~ ^[0-9]+$ ]]; then
    echo "Error: RESERVE_FREE_MB must be an integer MB value."
    usage
    exit 1
fi

mkdir -p "$LOG_DIR"
if [ -z "$LOG_FILE" ]; then
    safe_gpu_ids="${GPU_IDS//,/ _}"
    safe_gpu_ids="${safe_gpu_ids// /}"
    LOG_FILE="$LOG_DIR/dynamic_gpu_guard_${safe_gpu_ids}_reserve${RESERVE_FREE_MB}MB_$(date +%Y%m%d_%H%M%S).log"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

guard_gpu() {
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -c "
import gc
import time
import torch

gpu = '$gpu'
reserve_mb = int('$RESERVE_FREE_MB')
check_interval = float('$CHECK_INTERVAL')
chunk_mb = int('$CHUNK_MB')
hysteresis_mb = int('$HYSTERESIS_MB')
max_guard_mb = int('$MAX_GUARD_MB')

if chunk_mb <= 0:
    raise SystemExit('CHUNK_MB must be positive')
if reserve_mb < 0:
    raise SystemExit('RESERVE_FREE_MB must be non-negative')

torch.cuda.set_device(0)
guards = []

def mem_info_mb():
    free_b, total_b = torch.cuda.mem_get_info(0)
    return free_b // 1024 // 1024, total_b // 1024 // 1024

def guard_mb():
    return len(guards) * chunk_mb

def allocate_one():
    numel = chunk_mb * 1024 * 1024 // 4
    x = torch.empty(numel, dtype=torch.float32, device='cuda')
    x.fill_(1.0)
    guards.append(x)

def release_one():
    if guards:
        guards.pop()
        gc.collect()
        torch.cuda.empty_cache()

print(
    f'GPU {gpu} dynamic guard started: reserve={reserve_mb}MB, '
    f'chunk={chunk_mb}MB, hysteresis={hysteresis_mb}MB, '
    f'interval={check_interval}s, max_guard={max_guard_mb}MB',
    flush=True,
)

while True:
    free_mb, total_mb = mem_info_mb()
    current_guard = guard_mb()

    # If training or another process needs memory, release guard chunks until
    # the visible free memory returns to the reserve band or no guard remains.
    if free_mb < reserve_mb and guards:
        released = 0
        while free_mb < reserve_mb and guards:
            release_one()
            released += chunk_mb
            time.sleep(0.02)
            free_mb, total_mb = mem_info_mb()
        print(
            f'{time.strftime(\"%F %T\")} GPU {gpu}: released={released}MB '
            f'guard={guard_mb()}MB free={free_mb}MB reserve={reserve_mb}MB',
            flush=True,
        )

    # If a lot of memory is free, occupy part of the extra space. The
    # hysteresis band avoids constant allocate/release oscillation.
    elif free_mb > reserve_mb + hysteresis_mb:
        can_add = free_mb - reserve_mb - hysteresis_mb
        if max_guard_mb > 0:
            can_add = min(can_add, max(0, max_guard_mb - current_guard))
        add_chunks = max(0, can_add // chunk_mb)
        allocated = 0
        for _ in range(int(add_chunks)):
            try:
                allocate_one()
                allocated += chunk_mb
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        if allocated:
            free_mb, total_mb = mem_info_mb()
            print(
                f'{time.strftime(\"%F %T\")} GPU {gpu}: allocated={allocated}MB '
                f'guard={guard_mb()}MB free={free_mb}MB reserve={reserve_mb}MB',
                flush=True,
            )

    else:
        print(
            f'{time.strftime(\"%F %T\")} GPU {gpu}: guard={current_guard}MB '
            f'free={free_mb}MB reserve={reserve_mb}MB total={total_mb}MB',
            flush=True,
        )

    time.sleep(check_interval)
"
}

echo "========================================="
echo "Dynamic GPU guard"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "RESERVE_FREE_MB: $RESERVE_FREE_MB"
echo "CHECK_INTERVAL: $CHECK_INTERVAL"
echo "CHUNK_MB: $CHUNK_MB"
echo "HYSTERESIS_MB: $HYSTERESIS_MB"
echo "MAX_GUARD_MB: $MAX_GUARD_MB"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "LOG_FILE: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "========================================="

trap 'echo "Stopping dynamic guards..."; jobs -pr | xargs -r kill; exit 130' INT TERM

pids=()
for gpu in ${GPU_IDS//,/ }; do
    (
        guard_gpu "$gpu"
    ) &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

exit "$failed"
