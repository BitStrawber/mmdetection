#!/usr/bin/env python3
"""Build UIIS10K easy detection subset with A/B cross filtering.

Workflow:
  1. Split converted UIIS10K detection train json into train_A/train_B.
  2. Train a detector on A and validate/filter B.
  3. Train a detector on B and validate/filter A.
  4. Merge A_easy + B_easy into easy_merged.json.

The filtering criterion is per-image bbox mAP computed from detector outputs.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

import numpy as np
from tqdm import tqdm


CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--step',
        default='all',
        choices=['split', 'train', 'filter', 'merge', 'all'])
    parser.add_argument(
        '--data-root',
        default='/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco/',
        help='UIIS10K COCO root containing train/val image folders.')
    parser.add_argument(
        '--ann',
        default='/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco/annotations/instances_train_det.json',
        help='Converted detection train annotation.')
    parser.add_argument(
        '--img-prefix',
        default='train/',
        help='Image prefix relative to data root for the converted train json.')
    parser.add_argument(
        '--cross-dir',
        default=None,
        help='Output annotation dir. Default: <data-root>/annotations/cross_split_det.')
    parser.add_argument(
        '--config',
        default='configs/exp_2/cascade-rcnn_r50_fpn_2x_uiis10k_det.py')
    parser.add_argument('--work-dir', default='work_dirs/uiis10k_cross_easy')
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--gpu-ids', default='4,5')
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--port-a', default='29610')
    parser.add_argument('--port-b', default='29611')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-keep-ckpts', type=int, default=5)
    parser.add_argument(
        '--log-prefix',
        default='uiis10k_cross',
        help='Prefix for stageA/stageB training logs.')
    return parser.parse_args()


def load_coco(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_coco(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def split_coco(coco, seed):
    images = coco['images'][:]
    random.seed(seed)
    random.shuffle(images)
    mid = len(images) // 2
    img_a = {img['id'] for img in images[:mid]}
    img_b = {img['id'] for img in images[mid:]}

    def make_half(img_ids):
        half = deepcopy(coco)
        half['images'] = [img for img in coco['images'] if img['id'] in img_ids]
        half['annotations'] = [
            ann for ann in coco['annotations'] if ann['image_id'] in img_ids
        ]
        return half

    return make_half(img_a), make_half(img_b)


def run_command(cmd, env=None):
    print('\n' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def train_detector(args, train_ann, val_ann, work_dir, log_name, port):
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    img_prefix = args.img_prefix.rstrip('/') + '/'
    cfg_options = [
        f'train_dataloader.dataset.data_root={args.data_root}',
        f'train_dataloader.dataset.ann_file={train_ann}',
        f'train_dataloader.dataset.data_prefix.img={img_prefix}',
        f'val_dataloader.dataset.data_root={args.data_root}',
        f'val_dataloader.dataset.ann_file={val_ann}',
        f'val_dataloader.dataset.data_prefix.img={img_prefix}',
        f'test_dataloader.dataset.data_root={args.data_root}',
        f'test_dataloader.dataset.ann_file={val_ann}',
        f'test_dataloader.dataset.data_prefix.img={img_prefix}',
        f'val_evaluator.ann_file={val_ann}',
        f'test_evaluator.ann_file={val_ann}',
        'val_evaluator.metric=bbox',
        'test_evaluator.metric=bbox',
        'load_from=None',
        'resume=False',
        'default_hooks.checkpoint.save_best=coco/bbox_mAP',
        f'default_hooks.checkpoint.max_keep_ckpts={args.max_keep_ckpts}',
    ]
    shell_cmd = (
        f'set -o pipefail; '
        f'export MKL_THREADING_LAYER="${{MKL_THREADING_LAYER:-GNU}}"; '
        f'PORT={port} CUDA_VISIBLE_DEVICES={args.gpu_ids} '
        f'bash tools/dist_train.sh {args.config} {args.num_gpus} '
        f'--work-dir {work_dir} --cfg-options {" ".join(cfg_options)} '
        f'2>&1 | tee {os.path.join(args.log_dir, log_name)}'
    )
    run_command(['bash', '-lc', shell_cmd])


def find_best_checkpoint(work_dir):
    candidates = sorted(glob.glob(os.path.join(work_dir, 'best_coco_bbox_mAP*.pth')))
    if candidates:
        return candidates[-1]
    latest = os.path.join(work_dir, 'latest.pth')
    return latest if os.path.exists(latest) else None


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def per_image_map(pred_boxes, pred_scores, pred_labels, gts, cat_ids):
    if not gts:
        return 1.0
    if pred_boxes is None or len(pred_boxes) == 0:
        return 0.0

    gt_by_class = defaultdict(list)
    for gt in gts:
        gt_by_class[gt['category_id']].append(gt)
    gt_count = sum(len(v) for v in gt_by_class.values())
    sort_idx = np.argsort(-pred_scores)
    aps = []

    for iou_thr in np.linspace(0.5, 0.95, 10):
        tp = np.zeros(len(sort_idx))
        fp = np.zeros(len(sort_idx))
        gt_used = defaultdict(set)
        for rank, det_idx in enumerate(sort_idx):
            label = int(pred_labels[det_idx])
            if label >= len(cat_ids):
                fp[rank] = 1
                continue
            cat_id = cat_ids[label]
            if cat_id not in gt_by_class:
                fp[rank] = 1
                continue
            pred_box = [
                float(pred_boxes[det_idx][0]),
                float(pred_boxes[det_idx][1]),
                float(pred_boxes[det_idx][2] - pred_boxes[det_idx][0]),
                float(pred_boxes[det_idx][3] - pred_boxes[det_idx][1]),
            ]
            max_iou = 0.0
            max_gt_idx = -1
            for gt_idx, gt in enumerate(gt_by_class[cat_id]):
                if gt_idx in gt_used[cat_id]:
                    continue
                iou = compute_iou(pred_box, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            if max_iou >= iou_thr:
                tp[rank] = 1
                gt_used[cat_id].add(max_gt_idx)
            else:
                fp[rank] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(gt_count, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            ap += (np.max(precisions[recalls >= t])
                   if np.any(recalls >= t) else 0.0) / 101
        aps.append(ap)
    return float(np.mean(aps))


def filter_easy(args, checkpoint, val_ann, output_ann):
    sys.path.insert(0, os.getcwd())
    from mmdet.apis import inference_detector, init_detector

    print(f'Loading model: {checkpoint}')
    model = init_detector(args.config, checkpoint, device='cuda:0')
    coco = load_coco(val_ann)
    cat_ids = [cat['id'] for cat in sorted(coco['categories'], key=lambda x: x['id'])]
    anns_by_img = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    good_ids = set()
    img_prefix = args.img_prefix.rstrip('/')
    for img in tqdm(coco['images'], desc=f'filter {os.path.basename(val_ann)}'):
        img_path = os.path.join(args.data_root, img_prefix, img['file_name'])
        if not os.path.exists(img_path):
            continue
        result = inference_detector(model, img_path)
        pred = result.pred_instances
        score = per_image_map(
            pred.bboxes.cpu().numpy() if pred is not None else None,
            pred.scores.cpu().numpy() if pred is not None else None,
            pred.labels.cpu().numpy() if pred is not None else None,
            anns_by_img[img['id']],
            cat_ids)
        if score >= args.threshold:
            good_ids.add(img['id'])

    filtered = deepcopy(coco)
    filtered['images'] = [img for img in coco['images'] if img['id'] in good_ids]
    filtered['annotations'] = [
        ann for ann in coco['annotations'] if ann['image_id'] in good_ids
    ]
    save_coco(filtered, output_ann)
    total = len(coco['images'])
    print(f'Saved {len(good_ids)}/{total} easy images to {output_ann}')
    return filtered


def merge_easy(a_easy, b_easy, output_ann):
    a = load_coco(a_easy)
    b = load_coco(b_easy)
    merged = deepcopy(a)
    image_ids = {img['id'] for img in merged['images']}
    ann_ids = {ann['id'] for ann in merged['annotations']}
    next_img_id = max(image_ids) + 1 if image_ids else 1
    next_ann_id = max(ann_ids) + 1 if ann_ids else 1

    for img in b['images']:
        old_id = img['id']
        new_id = old_id
        if new_id in image_ids:
            new_id = next_img_id
            next_img_id += 1
        image_ids.add(new_id)
        new_img = deepcopy(img)
        new_img['id'] = new_id
        merged['images'].append(new_img)
        for ann in b['annotations']:
            if ann['image_id'] != old_id:
                continue
            new_ann = deepcopy(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = new_id
            next_ann_id += 1
            merged['annotations'].append(new_ann)

    save_coco(merged, output_ann)
    print(f'Merged easy images: {len(merged["images"])}')
    print(f'Output: {output_ann}')


def main():
    args = parse_args()
    args.data_root = args.data_root.rstrip('/') + '/'
    cross_dir = args.cross_dir or os.path.join(
        args.data_root, 'annotations', 'cross_split_det')
    Path(cross_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    train_a = os.path.join(cross_dir, 'train_A.json')
    train_b = os.path.join(cross_dir, 'train_B.json')
    easy_a = os.path.join(cross_dir, 'A_easy.json')
    easy_b = os.path.join(cross_dir, 'B_easy.json')
    easy_merged = os.path.join(cross_dir, 'easy_merged.json')

    if args.step in ('split', 'all'):
        coco = load_coco(args.ann)
        a, b = split_coco(coco, args.seed)
        save_coco(a, train_a)
        save_coco(b, train_b)
        print(f'A split: {len(a["images"])} images, {len(a["annotations"])} anns')
        print(f'B split: {len(b["images"])} images, {len(b["annotations"])} anns')

    if args.step in ('train', 'all'):
        train_detector(
            args, train_a, train_b,
            os.path.join(args.work_dir, 'stageA'),
            f'{args.log_prefix}_stageA.log',
            args.port_a)
        train_detector(
            args, train_b, train_a,
            os.path.join(args.work_dir, 'stageB'),
            f'{args.log_prefix}_stageB.log',
            args.port_b)

    if args.step in ('filter', 'all'):
        ckpt_a = find_best_checkpoint(os.path.join(args.work_dir, 'stageA'))
        ckpt_b = find_best_checkpoint(os.path.join(args.work_dir, 'stageB'))
        if not ckpt_a or not ckpt_b:
            raise RuntimeError('Missing stageA/stageB best checkpoints.')
        filter_easy(args, ckpt_a, train_b, easy_b)
        filter_easy(args, ckpt_b, train_a, easy_a)

    if args.step in ('merge', 'all'):
        merge_easy(easy_a, easy_b, easy_merged)


if __name__ == '__main__':
    main()
