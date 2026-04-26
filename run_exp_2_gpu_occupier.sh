#!/bin/bash

# GPU Occupier - 占位脚本，单纯占用GPU显存
# 用法: bash run_exp_2_gpu_occupier.sh [j2|j3|j4]
# 例如: bash run_exp_2_gpu_occupier.sh j2   # 占住GPU 0,1

GPU_J2="0,1"
GPU_J3="2,3"
GPU_J4="4,5"

case $1 in
    j2) GPU=$GPU_J2; echo "占用 J2 GPU 0,1" ;;
    j3) GPU=$GPU_J3; echo "占用 J3 GPU 2,3" ;;
    j4) GPU=$GPU_J4; echo "占用 J4 GPU 4,5" ;;
    *) echo "用法: bash run_exp_2_gpu_occupier.sh [j2|j3|j4]"; exit 1 ;;
esac

echo "开始占用 GPU $GPU ..."
echo "按 Ctrl+C 停止"

# 设置GPU并保持循环占用
export CUDA_VISIBLE_DEVICES=$GPU

# 简单循环占用，不运行任何训练
while true; do
    sleep 60
    echo "[$(date)] 占住 GPU $GPU 中..."
done