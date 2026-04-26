#!/bin/bash

# exp_2 监测脚本
# 查看 j2 j3 j4 运行状态

echo "========================================="
echo "exp_2 任务监测"
echo "========================================="
echo ""

check_task() {
    local name=$1
    local log="logs/${name}.log"
    local work="work_dirs/$name"
    
    echo "--- $name ---"
    
    if [ -f "$log" ]; then
        # 获取最后几行
        local last=$(tail -20 "$log" | grep -E "epoch|loss|AP|bbox_mAP" | tail -3)
        if [ -n "$last" ]; then
            echo "$last"
        fi
        
        # 检查是否在运行
        if pgrep -f "configs/exp_2/.*${name}.*.py" > /dev/null 2>&1; then
            echo "状态: 运行中 ✓"
        else
            echo "状态: 已结束"
        fi
    else
        echo "日志: 不存在"
    fi
    
    # 查看checkpoint
    if [ -d "$work" ]; then
        local ckpt=$(ls -t "$work"/*.pth 2>/dev/null | head -1)
        if [ -n "$ckpt" ]; then
            echo "最新: $(basename $ckpt)"
        fi
    fi
    
    echo ""
}

check_task "j2_det"
check_task "j2_mask"
check_task "j3_det"
check_task "j3_mask"
check_task "j4_det"
check_task "j4_mask"

echo "========================================="
echo "GPU 状态"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv