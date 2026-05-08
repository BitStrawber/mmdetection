#!/usr/bin/env python3
"""
RUOD 训练集交叉筛选 + DFUI 合并

流程:
  1. 将 RUOD train 均分为 A/B
  2. 在 A 上训练 24ep，在 B 上验证，筛选 B 中 AP@50>60% 的图
  3. 在 B 上训练 24ep，在 A 上验证，筛选 A 中 AP@50>60% 的图
  4. 合并筛选出的 A+B 图片 → 新 COCO 数据集
  5. 与 DFUI 合并 → 最终预训练数据集

注意: 整个过程只使用 RUOD train 数据，不泄露 val 信息
"""

import os, sys, json, random, shutil, subprocess
from copy import deepcopy

# ========== 配置 ==========
RUOD_ROOT  = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco'
ANNOTATION = f'{RUOD_ROOT}/annotations/instances_train.json'
WORK_DIR   = 'work_dirs/ruod_cross'
GPU_IDS    = '0,1'
NUM_GPUS   = 2
SEED       = 42

def load_coco(path):
    with open(path) as f:
        return json.load(f)

def save_coco(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def split_coco(coco):
    """按图片随机均分, 返回 (half_a, half_b)"""
    images = coco['images']
    random.seed(SEED)
    random.shuffle(images)
    mid = len(images) // 2
    img_ids_a = {img['id'] for img in images[:mid]}
    img_ids_b = {img['id'] for img in images[mid:]}
    
    def filter_half(img_ids):
        half = deepcopy(coco)
        half['images'] = [img for img in coco['images'] if img['id'] in img_ids]
        half['annotations'] = [ann for ann in coco['annotations'] if ann['image_id'] in img_ids]
        return half
    
    return filter_half(img_ids_a), filter_half(img_ids_b)

def run_train(config_path, work_dir):
    """运行训练"""
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_IDS} bash tools/dist_train.sh {config_path} {NUM_GPUS} --work-dir {work_dir}'
    subprocess.run(cmd, shell=True, check=True)

def filter_high_ap(val_ann_path, thresh=0.6):
    """
    用训练的模型在验证集上推理, 筛选 AP50 > thresh 的图片
    返回: (筛选后的coco, 筛选出的image_ids)
    注: 这里先返回占位, 实际需要实现推理逻辑
    """
    val_coco = load_coco(val_ann_path)
    # TODO: 实际推理 + 筛选
    # 这里需要加载训练好的checkpoint, 对val_coco中的每张图进行推理,
    # 计算每张图的AP50, 保留 > thresh 的图片
    return val_coco, [img['id'] for img in val_coco['images']]

def merge_datasets(coco_a, coco_b, dfui_path=None):
    """合并多个COCO数据集"""
    merged = deepcopy(coco_a)
    max_img_id = max(img['id'] for img in merged['images'])
    max_ann_id = max(ann['id'] for ann in merged['annotations'])
    
    # 重新分配B的ID
    id_offset = max_img_id + 1
    for img in coco_b['images']:
        img['id'] += id_offset
    for ann in coco_b['annotations']:
        ann['id'] = max_ann_id + 1 + ann['id']
        ann['image_id'] += id_offset
    
    merged['images'].extend(coco_b['images'])
    merged['annotations'].extend(coco_b['annotations'])
    
    # 如果指定了DFUI, 合并
    if dfui_path:
        dfui = load_coco(dfui_path)
        # TODO: 合并DFUI
        pass
    
    return merged


def main():
    print("=" * 60)
    print("RUOD 交叉筛选 Pipeline")
    print("=" * 60)
    
    # 1. 分割
    print("\n1. 分割 RUOD train → A/B")
    coco = load_coco(ANNOTATION)
    coco_a, coco_b = split_coco(coco)
    
    split_dir = f'{RUOD_ROOT}/cross_split'
    save_coco(coco_a, f'{split_dir}/train_A.json')
    save_coco(coco_b, f'{split_dir}/train_B.json')
    print(f"   A: {len(coco_a['images'])} images, {len(coco_a['annotations'])} annotations")
    print(f"   B: {len(coco_b['images'])} images, {len(coco_b['annotations'])} annotations")
    
    # 2. 创建训练配置 (A训练, B验证)
    print("\n2. 创建配置文件...")
    # TODO: 生成 mmdet config, 用A训练, B做验证集
    
    # 3. Stage 1: 训练A, 筛选B
    print("\n3. Stage 1: 训练A → 筛选B")
    # run_train('configs/exp_2/xxx_A.py', f'{WORK_DIR}/stage1_A')
    # filtered_b, _ = filter_high_ap(f'{split_dir}/train_B.json')
    
    # 4. Stage 2: 训练B, 筛选A
    print("\n4. Stage 2: 训练B → 筛选A")
    # run_train('configs/exp_2/xxx_B.py', f'{WORK_DIR}/stage2_B')
    # filtered_a, _ = filter_high_ap(f'{split_dir}/train_A.json')
    
    # 5. 合并筛选结果 + DFUI
    print("\n5. 合并筛选结果...")
    # final = merge_datasets(filtered_a, filtered_b, dfui_path)
    # save_coco(final, f'{RUOD_ROOT}/cross_split/cross_filtered_train.json')
    
    print("\n" + "=" * 60)
    print("脚本逻辑完成 (需补充训练/推理实现)")
    print("=" * 60)

if __name__ == '__main__':
    main()
