#!/bin/bash

# GPU Occupier - 占位脚本，真正占用GPU显存
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

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 用Python分配GPU显存
python -c "
import torch
import time
print('开始分配GPU显存...')
# 分配GPU 0,1 各约15GB (2000x2000x1000x4 ≈ 16GB)
for i in range(2):
    torch.cuda.set_device(i)
    x = torch.zeros(2000, 2000, 1000, device=f'cuda:{i}') * 1.0
    mem = x.element_size() * x.nelement() / 1e9
    print(f'GPU {i}: 已分配 {mem:.1f} GB')
while True:
    time.sleep(60)
"