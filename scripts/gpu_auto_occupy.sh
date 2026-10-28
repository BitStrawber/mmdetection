#!/bin/bash
# GPU独立监控 + 自动占位 (每个GPU独立)
# 用法: bash scripts/gpu_auto_occupy.sh "0,1,2,3" 60 20

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
INTERVAL="${2:-60}"
TARGET_GB="${3:-20}"
LOG_DIR="${OCCUPY_LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

if ! [[ "$TARGET_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid target memory in GiB: $TARGET_GB" >&2
    exit 1
fi

echo "GPU独立占位监控: $GPU_IDS, 目标显存 ${TARGET_GB}GiB, 检测间隔 ${INTERVAL}s"
echo "按 Ctrl+C 停止"

declare -A OCCUPIED

occupy_gpu() {
    local GPU=$1
    if [ "${OCCUPIED[$GPU]}" = "1" ]; then
        return
    fi
    
    UTIL=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$UTIL" ] || [ "$UTIL" -le 10 ] 2>/dev/null; then
        echo "$(date): GPU $GPU 空闲(利用率${UTIL}%), 开始占位..."
        CUDA_VISIBLE_DEVICES=$GPU TARGET_GB="$TARGET_GB" nohup python -c "
import os
import time

import torch

target_bytes = int(float(os.environ['TARGET_GB']) * 1024 ** 3)
chunk_bytes = 1024 ** 3
chunk_elements = chunk_bytes // 4
blocks = []
allocated = 0
while allocated + chunk_bytes <= target_bytes:
    block = torch.empty(chunk_elements, dtype=torch.float32, device='cuda')
    block.zero_()
    blocks.append(block)
    allocated += chunk_bytes
remainder = target_bytes - allocated
if remainder:
    block = torch.empty((remainder + 3) // 4, dtype=torch.float32, device='cuda')
    block.zero_()
    blocks.append(block)
    allocated += remainder
print(f'GPU $GPU 占位完成: {allocated / 1024**3:.2f} GiB', flush=True)
while True:
    time.sleep(60)
" > "$LOG_DIR/gpu_occupy_${GPU}.log" 2>&1 &
        OCCUPIED[$GPU]=1
    fi
}

while true; do
    ALL_OCCUPIED=true
    for GPU in ${GPU_IDS//,/ }; do
        if [ "${OCCUPIED[$GPU]}" != "1" ]; then
            ALL_OCCUPIED=false
            occupy_gpu $GPU
        fi
    done
    
    if $ALL_OCCUPIED; then
        echo "$(date): 所有GPU已占位, 退出监控 (释放: pkill -f 'torch.zeros.*2000, 2000, 1000')"
        exit 0
    fi
    
    sleep $INTERVAL
done
