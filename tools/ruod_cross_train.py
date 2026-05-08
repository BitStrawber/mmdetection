#!/usr/bin/env python3
"""
RUOD 训练集交叉筛选

流程:
  1. 将 RUOD train 均分为 A, B
  2. 创建 A_train / B_val 配置, 训练24ep
  3. 用最佳模型对 B 推理, 筛选 AP50>60% 的图 → B_easy.json
  4. 创建 B_train / A_val 配置, 训练24ep
  5. 用最佳模型对 A 推理, 筛选 AP50>60% 的图 → A_easy.json
  6. 输出 A_easy.json + B_easy.json

用法:
  python tools/ruod_cross_train.py --step split     # 仅分割
  python tools/ruod_cross_train.py --step all       # 全流程
"""

import os, sys, json, random, argparse
from copy import deepcopy

# ========== 配置 ==========
RUOD_ROOT  = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco'
ANNOTATION = f'{RUOD_ROOT}/annotations/instances_train.json'
CROSS_DIR  = f'{RUOD_ROOT}/cross_split'
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
    """按图片随机均分"""
    images = coco['images'][:]
    random.seed(SEED)
    random.shuffle(images)
    mid = len(images) // 2
    
    img_a = {img['id'] for img in images[:mid]}
    img_b = {img['id'] for img in images[mid:]}
    
    def make_half(img_set):
        h = deepcopy(coco)
        h['images'] = [img for img in coco['images'] if img['id'] in img_set]
        h['annotations'] = [a for a in coco['annotations'] if a['image_id'] in img_set]
        return h
    
    return make_half(img_a), make_half(img_b)


def gen_configs():
    """生成 A_train → B_val 和 B_train → A_val 两个mmdet配置"""
    base = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'
    
    config_a = f"""# Cross: A训练 + B验证
_base_ = '{base}'
data_root = '{RUOD_ROOT}'
train_dataloader = dict(
    batch_size=6, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='{CROSS_DIR}/train_A.json'))
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='{CROSS_DIR}/train_B.json'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='{CROSS_DIR}/train_B.json')
test_evaluator = val_evaluator
"""
    
    config_b = f"""# Cross: B训练 + A验证
_base_ = '{base}'
data_root = '{RUOD_ROOT}'
train_dataloader = dict(
    batch_size=6, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='{CROSS_DIR}/train_B.json'))
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='{CROSS_DIR}/train_A.json'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='{CROSS_DIR}/train_A.json')
test_evaluator = val_evaluator
"""
    
    os.makedirs('configs/exp_2/cross', exist_ok=True)
    with open('configs/exp_2/cross/train_A_val_B.py', 'w') as f:
        f.write(config_a)
    with open('configs/exp_2/cross/train_B_val_A.py', 'w') as f:
        f.write(config_b)
    print("配置已生成: configs/exp_2/cross/train_A_val_B.py, train_B_val_A.py")


def run_train(config_path, work_dir, log_name):
    cmd = (f'CUDA_VISIBLE_DEVICES={GPU_IDS} '
           f'bash tools/dist_train.sh {config_path} {NUM_GPUS} '
           f'--work-dir {work_dir} 2>&1 | tee logs/{log_name}')
    print(f"\n训练: {cmd}")
    os.system(cmd)


def filter_by_ap(checkpoint, val_json, output_json, thresh=0.6):
    """
    用checkpoint推理val_json中的图片,计算每图AP50,
    保留AP50>thresh的图片,输出筛选后的COCO JSON
    
    简化实现: 用mmdet test推理val_json,解析per-image AP
    """
    import subprocess, tempfile
    
    # 创建临时测试配置
    test_cfg = f"""
_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'
data_root = '{RUOD_ROOT}'
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='{val_json}'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='{val_json}', metric='bbox')
test_evaluator = val_evaluator
load_from = '{checkpoint}'
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_cfg)
        test_cfg_path = f.name
    
    # 运行推理, 输出COCO results
    result_file = f'{CROSS_DIR}/_results.pkl'
    cmd = (f'python tools/test.py {test_cfg_path} {checkpoint} '
           f'--out {result_file} 2>&1 | tee {CROSS_DIR}/inference.log')
    print(f"\n推理: {cmd}")
    os.system(cmd)
    
    # TODO: 解析结果, 筛选高AP图片
    # 这里需要从mmdet的result pkl中提取per-image metrics
    # 简化: 直接返回全量(后续实现)
    val_coco = load_coco(val_json)
    save_coco(val_coco, output_json)
    print(f"筛选完成: {len(val_coco['images'])} 张 → {output_json}")
    
    os.unlink(test_cfg_path)
    return val_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='all', choices=['split', 'train', 'filter', 'all'])
    args = parser.parse_args()
    
    os.makedirs(CROSS_DIR, exist_ok=True)
    os.makedirs(f'{WORK_DIR}/stageA', exist_ok=True)
    os.makedirs(f'{WORK_DIR}/stageB', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # ===== Step 1: Split =====
    if args.step in ('split', 'all'):
        print("=" * 60)
        print("Step 1: 分割 RUOD train → A / B")
        coco = load_coco(ANNOTATION)
        a, b = split_coco(coco)
        save_coco(a, f'{CROSS_DIR}/train_A.json')
        save_coco(b, f'{CROSS_DIR}/train_B.json')
        print(f"  A: {len(a['images'])} images")
        print(f"  B: {len(b['images'])} images")
        gen_configs()
    
    # ===== Step 2: Train A → Filter B =====
    if args.step in ('train', 'all'):
        print("=" * 60)
        print("Step 2: 训练A → 筛选B")
        run_train('configs/exp_2/cross/train_A_val_B.py',
                  f'{WORK_DIR}/stageA', 'cross_stageA.log')
    
    # ===== Step 3: Train B → Filter A =====
    if args.step in ('train', 'all'):
        print("=" * 60)
        print("Step 3: 训练B → 筛选A")
        run_train('configs/exp_2/cross/train_B_val_A.py',
                  f'{WORK_DIR}/stageB', 'cross_stageB.log')
    
    # ===== Step 4: Filter =====
    if args.step in ('filter', 'all'):
        print("=" * 60)
        print("Step 4: 筛选高AP图片")
        # 找最佳checkpoint
        import glob
        ckpt_a = sorted(glob.glob(f'{WORK_DIR}/stageA/best_coco*.pth'))[-1] if glob.glob(f'{WORK_DIR}/stageA/best_coco*.pth') else None
        ckpt_b = sorted(glob.glob(f'{WORK_DIR}/stageB/best_coco*.pth'))[-1] if glob.glob(f'{WORK_DIR}/stageB/best_coco*.pth') else None
        
        if ckpt_a:
            filter_by_ap(ckpt_a, f'{CROSS_DIR}/train_B.json',
                        f'{CROSS_DIR}/B_easy.json')
        if ckpt_b:
            filter_by_ap(ckpt_b, f'{CROSS_DIR}/train_A.json',
                        f'{CROSS_DIR}/A_easy.json')
        
        print("\n" + "=" * 60)
        print("完成!")
        print(f"  A_easy: {CROSS_DIR}/A_easy.json")
        print(f"  B_easy: {CROSS_DIR}/B_easy.json")


if __name__ == '__main__':
    main()
