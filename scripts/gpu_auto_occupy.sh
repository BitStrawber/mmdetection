#!/bin/bash
# GPU独立监控 + 自动占位 (每个GPU独立)
# 用法: bash scripts/gpu_auto_occupy.sh "0,1,2,3" 60

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
INTERVAL="${2:-60}"

echo "GPU独立占位监控: $GPU_IDS, 检测间隔 ${INTERVAL}s"
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
        CUDA_VISIBLE_DEVICES=$GPU nohup python -c "
import torch, time
x = torch.zeros(2000, 2000, 1000, device='cuda')
print('GPU $GPU 占位完成')
while True:
    time.sleep(60)
" > logs/gpu_occupy_${GPU}.log 2>&1 &
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
