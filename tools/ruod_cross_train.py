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

import os, sys, json, random, argparse, numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

# ========== 配置 ==========
RUOD_ROOT  = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco'
ANNOTATION = f'{RUOD_ROOT}/annotations/instances_train.json'
CROSS_DIR  = f'{RUOD_ROOT}/cross_split'
WORK_DIR   = 'work_dirs/ruod_cross'
GPU_IDS    = '4,5'
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
    """配置已预生成(不再动态创建)"""
    print(f"配置文件已预生成:")
    print(f"  configs/exp_2/cross/train_A_val_B.py")
    print(f"  configs/exp_2/cross/train_B_val_A.py")


def run_train(config_path, work_dir, log_name):
    cmd = (f'CUDA_VISIBLE_DEVICES={GPU_IDS} '
           f'bash tools/dist_train.sh {config_path} {NUM_GPUS} '
           f'--work-dir {work_dir} 2>&1 | tee logs/{log_name}')
    print(f"\n训练: {cmd}")
    os.system(cmd)


def filter_by_ap(checkpoint, val_json, output_json, thresh=0.6):
    """
    用checkpoint推理val_json, 计算每图mAP, 保留>thresh的图片
    """
    import sys
    sys.path.insert(0, os.getcwd())
    from mmdet.apis import init_detector, inference_detector
    from mmengine.config import Config
    
    # 创建测试配置
    test_cfg_dict = Config.fromfile('configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py')
    test_cfg_dict.data_root = RUOD_ROOT
    test_cfg_dict.val_dataloader = dict(
        batch_size=1, num_workers=2,
        dataset=dict(
            type='CocoDataset',
            data_root=RUOD_ROOT,
            data_prefix=dict(img='train/'),
            ann_file=val_json,
            metainfo=dict(classes=('holothurian','echinus','scallop','starfish','fish',
                                     'corals','diver','cuttlefish','turtle','jellyfish'))
        )
    )
    
    # 初始化模型
    print(f"  加载模型: {checkpoint}")
    model = init_detector(test_cfg_dict, checkpoint, device='cuda:0')
    
    val_coco = load_coco(val_json)
    img_map = {img['id']: img for img in val_coco['images']}
    
    # 逐图推理
    import numpy as np
    from collections import defaultdict
    
    good_ids = set()
    total = len(img_map)
    
    print(f"  推理 {total} 张图片...")
    for idx, img_info in enumerate(tqdm(list(img_map.values()))):
        if (idx + 1) % 200 == 0:
            print(f"    {idx+1}/{total}...")
        
        img_path = os.path.join(RUOD_ROOT, 'train', img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        
        result = inference_detector(model, img_path)
        pred_instances = result.pred_instances
        
        # 获取GT标注
        ann_ids = [a['id'] for a in val_coco['annotations'] if a['image_id'] == img_info['id']]
        gts = [a for a in val_coco['annotations'] if a['image_id'] == img_info['id']]
        
        if len(gts) == 0:
            good_ids.add(img_info['id'])
            continue
        
        if pred_instances is None or len(pred_instances.bboxes) == 0:
            continue
        
        # 提取预测: [bbox, score, label]
        pred_boxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()
        
        # 按置信度排序
        sort_idx = np.argsort(-pred_scores)
        
        gt_by_class = defaultdict(list)
        for gt in gts:
            gt_by_class[gt['category_id']].append(gt)
        gt_count = sum(len(v) for v in gt_by_class.values())
        
        # 计算per-image mAP
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        aps = []
        
        for iou_thr in iou_thresholds:
            tp = np.zeros(len(sort_idx))
            fp = np.zeros(len(sort_idx))
            gt_used = defaultdict(set)
            
            for rank, d_idx in enumerate(sort_idx):
                label = int(pred_labels[d_idx]) + 1  # 0-indexed → 1-indexed
                if label not in gt_by_class:
                    fp[rank] = 1
                    continue
                
                pred_box = [float(pred_boxes[d_idx][0]), float(pred_boxes[d_idx][1]),
                           float(pred_boxes[d_idx][2] - pred_boxes[d_idx][0]),
                           float(pred_boxes[d_idx][3] - pred_boxes[d_idx][1])]
                
                max_iou = 0
                max_gt_idx = -1
                for g_idx, gt in enumerate(gt_by_class[label]):
                    if g_idx in gt_used[label]:
                        continue
                    iou = _compute_iou(pred_box, gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = g_idx
                
                if max_iou >= iou_thr:
                    tp[rank] = 1
                    gt_used[label].add(max_gt_idx)
                else:
                    fp[rank] = 1
            
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(gt_count, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
            
            ap = 0
            for t in np.linspace(0, 1, 101):
                p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
                ap += p / 101
            aps.append(ap)
        
        per_image_map = np.mean(aps)
        if per_image_map >= thresh:
            good_ids.add(img_info['id'])
    
    print(f"\n  筛选结果: {len(good_ids)}/{total} ({len(good_ids)/total*100:.1f}%) 保留, {total-len(good_ids)} 丢弃")
    
    # 筛选并保存
    val_coco['images'] = [img for img in val_coco['images'] if img['id'] in good_ids]
    val_coco['annotations'] = [a for a in val_coco['annotations'] if a['image_id'] in good_ids]
    save_coco(val_coco, output_json)
    
    return val_coco
    
    # 计算per-image AP50
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    
    coco_gt = COCO(val_json)
    coco_dt = coco_gt.loadRes(result_json)
    
    # 逐图评估
    img_ids = coco_gt.getImgIds()
    good_ids = set()
    
    for iid in img_ids:
        # 获取该图GT和预测
        ann_ids = coco_gt.getAnnIds(imgIds=[iid])
        dt_ids = coco_dt.getAnnIds(imgIds=[iid])
        
        if len(ann_ids) == 0:
            # 无GT标注的图直接保留
            good_ids.add(iid)
            continue
        
        gts = coco_gt.loadAnns(ann_ids)
        dts = coco_dt.loadAnns(dt_ids)
        
        if len(dts) == 0:
            continue  # 没检测到, 跳过
        
        # 按类别分组GT + 按置信度排序预测
        from collections import defaultdict
        gt_by_class = defaultdict(list)
        for gt in gts:
            gt_by_class[gt['category_id']].append(gt)
        gt_count = sum(len(v) for v in gt_by_class.values())
        dts_sorted = sorted(dts, key=lambda x: x['score'], reverse=True)
        
        # 计算该图的mAP (IoU 0.50:0.05:0.95)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        aps = []
        
        for iou_thr in iou_thresholds:
            tp = np.zeros(len(dts_sorted))
            fp = np.zeros(len(dts_sorted))
            gt_used = defaultdict(set)
            
            for d_idx, dt in enumerate(dts_sorted):
                cat = dt['category_id']
                if cat not in gt_by_class:
                    fp[d_idx] = 1
                    continue
                
                dt_box = dt['bbox']
                max_iou = 0
                max_gt_idx = -1
                for g_idx, gt in enumerate(gt_by_class[cat]):
                    if g_idx in gt_used[cat]:
                        continue
                    iou = _compute_iou(dt_box, gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = g_idx
                
                if max_iou >= iou_thr:
                    tp[d_idx] = 1
                    gt_used[cat].add(max_gt_idx)
                else:
                    fp[d_idx] = 1
            
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(gt_count, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
            
            # 101点插值
            ap = 0
            for t in np.linspace(0, 1, 101):
                p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
                ap += p / 101
            aps.append(ap)
        
        per_image_map = np.mean(aps)
        
        if per_image_map >= thresh:
            good_ids.add(iid)
    
    total = len(img_ids)
    print(f"  筛选结果: {len(good_ids)}/{total} ({len(good_ids)/total*100:.1f}%) 保留, {total-len(good_ids)} 丢弃")
    
    # 筛选
    val_coco = load_coco(val_json)
    val_coco['images'] = [img for img in val_coco['images'] if img['id'] in good_ids]
    val_coco['annotations'] = [a for a in val_coco['annotations'] if a['image_id'] in good_ids]
    save_coco(val_coco, output_json)
    
    # 清理
    for f in [result_pkl, result_json]:
        if os.path.exists(f): os.remove(f)
    
    return val_coco


def _compute_iou(box1, box2):
    """计算两个[x,y,w,h]格式bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


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
            print(f"\n模型A ({ckpt_a}) → 筛选B图片")
            result = filter_by_ap(ckpt_a, f'{CROSS_DIR}/train_B.json',
                        f'{CROSS_DIR}/B_easy.json')
            print(f"  B_easy: {len(result['images'])} images")
        if ckpt_b:
            print(f"\n模型B ({ckpt_b}) → 筛选A图片")
            result = filter_by_ap(ckpt_b, f'{CROSS_DIR}/train_A.json',
                        f'{CROSS_DIR}/A_easy.json')
            print(f"  A_easy: {len(result['images'])} images")
        
        print("\n" + "=" * 60)
        print("完成!")
        print(f"  A_easy: {CROSS_DIR}/A_easy.json")
        print(f"  B_easy: {CROSS_DIR}/B_easy.json")
        
        # GPU占位
        print("\nGPU继续占位中... 手动停止: pkill -f gpu_cross_occupy")
        for gpu_id in GPU_IDS.split(','):
            os.system(
                f'CUDA_VISIBLE_DEVICES={gpu_id} nohup python -c '
                f'"import torch,time; x=torch.zeros(2000,2000,1000,device=\\\"cuda\\\"); print(\\\"GPU {gpu_id} 占位完成\\\"); '
                f'while True: time.sleep(60)" '
                f'> logs/gpu_cross_occupy_{gpu_id}.log 2>&1 &'
            )
        input("按回车键释放GPU...")


if __name__ == '__main__':
    main()
