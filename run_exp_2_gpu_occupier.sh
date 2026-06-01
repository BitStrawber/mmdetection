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
GPU_IDLE_CHECKS=${GPU_IDLE_CHECKS:-2}
GPU_WAIT_INTERVAL=${GPU_WAIT_INTERVAL:-30}
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE=${LOG_FILE:-}

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
    echo "GPU $gpu: waiting until free_mem>=${OCCUPY_MEM_MB}MB and util<=${GPU_MAX_UTIL}% for ${GPU_IDLE_CHECKS} consecutive check(s)."
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
        if [ "$mem_free" -ge "$OCCUPY_MEM_MB" ] && [ "$util" -le "$GPU_MAX_UTIL" ]; then
            idle_rounds=$((idle_rounds + 1))
            echo "$(date '+%F %T') GPU $gpu: idle check ${idle_rounds}/${GPU_IDLE_CHECKS} passed."
            if [ "$idle_rounds" -ge "$GPU_IDLE_CHECKS" ]; then
                echo "$(date '+%F %T') GPU $gpu: enough free memory, start occupying."
                return
            fi
        else
            idle_rounds=0
            echo "$(date '+%F %T') GPU $gpu: insufficient free memory or busy. Recheck after ${GPU_WAIT_INTERVAL}s."
        fi
        sleep "$GPU_WAIT_INTERVAL"
    done
}

occupy_gpu() {
    local gpu="$1"
    wait_for_gpu "$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" python -c "
import time
import torch

target_mb = int('$OCCUPY_MEM_MB')
numel = target_mb * 1024 * 1024 // 4
torch.cuda.set_device(0)
x = torch.empty(numel, dtype=torch.float32, device='cuda')
x.fill_(1.0)
allocated_mb = x.element_size() * x.nelement() / 1024 / 1024
print('GPU $gpu occupied with %.0f MB tensor' % allocated_mb, flush=True)
while True:
    time.sleep(60)
"
}

echo "========================================="
echo "GPU occupier"
echo "========================================="
echo "GPU_IDS: $GPU_IDS"
echo "OCCUPY_MEM_MB: $OCCUPY_MEM_MB"
echo "GPU_MAX_UTIL: $GPU_MAX_UTIL"
echo "GPU_IDLE_CHECKS: $GPU_IDLE_CHECKS"
echo "GPU_WAIT_INTERVAL: $GPU_WAIT_INTERVAL"
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
