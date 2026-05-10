#!/bin/bash
# GPU空闲检测 + 自动占位
# 用法: bash scripts/gpu_auto_occupy.sh [GPU_IDS] [CHECK_INTERVAL]
# 例:   bash scripts/gpu_auto_occupy.sh "0,1,2,3" 60

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
INTERVAL="${2:-60}"

echo "GPU自动占位监控: $GPU_IDS, 检测间隔 ${INTERVAL}s"
echo "按 Ctrl+C 停止"

while true; do
    ALL_IDLE=true
    for GPU in ${GPU_IDS//,/ }; do
        UTIL=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        if [ "$UTIL" -gt 10 ] 2>/dev/null; then
            ALL_IDLE=false
            break
        fi
    done
    
    if $ALL_IDLE; then
        echo "$(date): GPU $GPU_IDS 空闲, 开始占位..."
        
        # 启动占位进程
        for GPU in ${GPU_IDS//,/ }; do
            CUDA_VISIBLE_DEVICES=$GPU nohup python -c "
import torch, time
x = torch.zeros(2000, 2000, 1000, device='cuda')
print(f'GPU $GPU 占位完成')
while True:
    time.sleep(60)
" > logs/gpu_occupy_${GPU}.log 2>&1 &
        done
        
        echo "GPU已占用, 等待手动释放 (pkill -f 'torch.zeros.*2000, 2000, 1000')"
        break
    else
        echo "$(date): GPU使用中, 下次检测 ${INTERVAL}s 后"
    fi
    
    sleep $INTERVAL
done
