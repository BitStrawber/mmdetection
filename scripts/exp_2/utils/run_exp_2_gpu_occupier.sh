#!/bin/bash
set -euo pipefail

# GPU memory occupier with per-GPU free-memory waiting.
#
# Usage:
#   bash run_exp_2_gpu_occupier.sh "0,1" 16000
#
# Arguments:
#   $1: GPU ids, comma-separated. Example: "0,1,4"
#   $2: memory to occupy on each GPU, in MB. Example: 16000
#
# Each GPU is checked independently. If one GPU has enough free memory, it starts
# occupying immediately; GPUs without enough free memory keep waiting.

GPU_IDS="${1:-}"
OCCUPY_MEM_MB="${2:-}"

GPU_MAX_UTIL=${GPU_MAX_UTIL:-10}
GPU_START_MAX_USED_MB=${GPU_START_MAX_USED_MB:-}
GPU_IDLE_CHECKS=${GPU_IDLE_CHECKS:-2}
GPU_WAIT_INTERVAL=${GPU_WAIT_INTERVAL:-30}
OCCUPY_TARGET_UTIL=${OCCUPY_TARGET_UTIL:-60}
OCCUPY_CYCLE_SEC=${OCCUPY_CYCLE_SEC:-1.0}
OCCUPY_MATMUL_SIZE=${OCCUPY_MATMUL_SIZE:-2048}
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE=${LOG_FILE:-}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}

usage() {
    echo "Usage: bash run_exp_2_gpu_occupier.sh \"GPU_IDS\" OCCUPY_MEM_MB"
    echo "Example: bash run_exp_2_gpu_occupier.sh \"0,1\" 16000"
}

if [ -z "$GPU_IDS" ] || [ -z "$OCCUPY_MEM_MB" ]; then
    usage
    exit 1
fi

if ! [[ "$OCCUPY_MEM_MB" =~ ^[0-9]+$ ]]; then
    echo "Error: OCCUPY_MEM_MB must be an integer MB value."
    usage
    exit 1
fi

if [ -n "$GPU_START_MAX_USED_MB" ] && ! [[ "$GPU_START_MAX_USED_MB" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU_START_MAX_USED_MB must be an integer MB value when set."
    usage
    exit 1
fi

mkdir -p "$LOG_DIR"
if [ -z "$LOG_FILE" ]; then
    safe_gpu_ids="${GPU_IDS//,/ _}"
    safe_gpu_ids="${safe_gpu_ids// /}"
    LOG_FILE="$LOG_DIR/gpu_occupier_${safe_gpu_ids}_${OCCUPY_MEM_MB}MB_$(date +%Y%m%d_%H%M%S).log"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

query_gpu_state() {
    local gpu_id="$1"
    nvidia-smi \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v id="$gpu_id" '
            {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                gsub(/[[:space:]]/, "", $3)
                gsub(/[[:space:]]/, "", $4)
                if ($1 == id) {
                    print $2, $3, $4
                    exit
                }
            }'
}

wait_for_gpu() {
    local gpu="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU $gpu: nvidia-smi not found."
        exit 1
    fi

    local idle_rounds=0
    if [ -n "$GPU_START_MAX_USED_MB" ]; then
        echo "GPU $gpu: waiting until used_mem<${GPU_START_MAX_USED_MB}MB, free_mem>=${OCCUPY_MEM_MB}MB, and util<=${GPU_MAX_UTIL}% for ${GPU_IDLE_CHECKS} consecutive check(s)."
    else
        echo "GPU $gpu: waiting until free_mem>=${OCCUPY_MEM_MB}MB and util<=${GPU_MAX_UTIL}% for ${GPU_IDLE_CHECKS} consecutive check(s)."
    fi
    while true; do
        local state=""
        state=$(query_gpu_state "$gpu" || true)
        if [ -z "$state" ]; then
            echo "$(date '+%F %T') GPU $gpu: not found. Recheck after ${GPU_WAIT_INTERVAL}s."
            idle_rounds=0
            sleep "$GPU_WAIT_INTERVAL"
            continue
        fi

        local mem_used mem_free util
        read -r mem_used mem_free util <<< "$state"
        echo "$(date '+%F %T') GPU $gpu: used=${mem_used}MB free=${mem_free}MB util=${util}%"
        local memory_condition=0
        if [ -n "$GPU_START_MAX_USED_MB" ]; then
            if [ "$mem_used" -lt "$GPU_START_MAX_USED_MB" ] && [ "$mem_free" -ge "$OCCUPY_MEM_MB" ]; then
                memory_condition=1
            fi
        else
            if [ "$mem_free" -ge "$OCCUPY_MEM_MB" ]; then
                memory_condition=1
            fi
        fi

        if [ "$memory_condition" -eq 1 ] && [ "$util" -le "$GPU_MAX_UTIL" ]; then
            idle_rounds=$((idle_rounds + 1))
            echo "$(date '+%F %T') GPU $gpu: idle check ${idle_rounds}/${GPU_IDLE_CHECKS} passed."
            if [ "$idle_rounds" -ge "$GPU_IDLE_CHECKS" ]; then
                echo "$(date '+%F %T') GPU $gpu: enough free memory, start occupying."
                return
            fi
        else
            idle_rounds=0
            echo "$(date '+%F %T') GPU $gpu: start condition not met. Recheck after ${GPU_WAIT_INTERVAL}s."
        fi
        sleep "$GPU_WAIT_INTERVAL"
    done
}

occupy_gpu() {
    local gpu="$1"
    wait_for_gpu "$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -c "
import time
import torch

target_mb = int('$OCCUPY_MEM_MB')
target_util = max(0, min(100, int('$OCCUPY_TARGET_UTIL')))
cycle_sec = float('$OCCUPY_CYCLE_SEC')
matmul_size = int('$OCCUPY_MATMUL_SIZE')
numel = target_mb * 1024 * 1024 // 4
torch.cuda.set_device(0)
x = torch.empty(numel, dtype=torch.float32, device='cuda')
x.fill_(1.0)
allocated_mb = x.element_size() * x.nelement() / 1024 / 1024
print('GPU $gpu occupied with %.0f MB tensor, target util %d%%' % (allocated_mb, target_util), flush=True)

if target_util <= 0:
    while True:
        time.sleep(60)

a = torch.randn(matmul_size, matmul_size, dtype=torch.float32, device='cuda')
b = torch.randn(matmul_size, matmul_size, dtype=torch.float32, device='cuda')
c = torch.empty(matmul_size, matmul_size, dtype=torch.float32, device='cuda')
busy_sec = cycle_sec * target_util / 100.0
idle_sec = max(0.0, cycle_sec - busy_sec)

while True:
    start = time.time()
    while time.time() - start < busy_sec:
        torch.mm(a, b, out=c)
        torch.cuda.synchronize()
    if idle_sec > 0:
        time.sleep(idle_sec)
"
}

echo "========================================="
echo "GPU occupier"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "OCCUPY_MEM_MB: $OCCUPY_MEM_MB"
echo "GPU_MAX_UTIL: $GPU_MAX_UTIL"
echo "GPU_START_MAX_USED_MB: ${GPU_START_MAX_USED_MB:-disabled}"
echo "GPU_IDLE_CHECKS: $GPU_IDLE_CHECKS"
echo "GPU_WAIT_INTERVAL: $GPU_WAIT_INTERVAL"
echo "OCCUPY_TARGET_UTIL: $OCCUPY_TARGET_UTIL"
echo "OCCUPY_CYCLE_SEC: $OCCUPY_CYCLE_SEC"
echo "OCCUPY_MATMUL_SIZE: $OCCUPY_MATMUL_SIZE"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "LOG_FILE: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "========================================="

trap 'echo "Stopping occupiers..."; jobs -pr | xargs -r kill; exit 130' INT TERM

pids=()
for gpu in ${GPU_IDS//,/ }; do
    (
        occupy_gpu "$gpu"
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
