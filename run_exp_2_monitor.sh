#!/bin/bash

# GPU Watcher - 监测GPU使用情况
# 用法: bash run_exp_2_watcher.sh

echo "========================================="
echo "GPU Watcher - $(date)"
echo "========================================="
echo ""

# 查看每个任务的GPU
echo "--- 任务GPU分配 ---"
echo "j2: GPU 0,1"
echo "j3: GPU 2,3"
echo "j4: GPU 4,5"
echo ""

# 查看GPU状态
echo "--- GPU状态 ---"
nvidia-smi --query-gpu=index,name,temperature,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv
echo ""

# 查看进程
echo "--- 训练进程 ---"
ps aux | grep -E "tools/train.py|dist_train.sh" | grep -v grep | head -10
echo ""

# 查看当前epoch
echo "--- 最新Epoch ---"
for task in j2_det j2_mask j3_det j3_mask j4_det j4_mask; do
    log="logs/${task}.log"
    if [ -f "$log" ]; then
        epoch=$(tail -50 "$log" | grep -oP 'epoch.*\K[0-9]+' | tail -1)
        if [ -n "$epoch" ]; then
            echo "$task: epoch $epoch"
        fi
    fi
done

echo "========================================="