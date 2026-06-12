#!/usr/bin/env python3
"""Convert USOD10K saliency masks to COCO detection annotations.

The official USOD10K layout is expected to look like:

  USOD10k/USOD10k/TR/RGB/RGB/*.png
  USOD10k/USOD10k/TR/GT/GT/*.png
  USOD10k/USOD10k/VAL/RGB/RGB/*.png
  USOD10k/USOD10k/VAL/GT/GT/*.png
  USOD10k/USOD10k/TE/RGB/*.png
  USOD10k/USOD10k/TE/GT/GT/*.png

The output is a self-contained COCO detection dataset:

  out_root/
    images/*.png
    annotations/instances_train.json
    annotations/instances_val.json
    annotations/instances_trainval.json
    annotations/instances_test.json

Each connected foreground component in the GT mask is converted to one bbox.
All boxes use a single category: object.
"""

import argparse
import json
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


SPLIT_INFO = {
    'train': ('TR', ('RGB', 'RGB'), ('GT', 'GT'), 'usod_tr'),
    'val': ('VAL', ('RGB', 'RGB'), ('GT', 'GT'), 'usod_val'),
    'test': ('TE', ('RGB',), ('GT', 'GT'), 'usod_te'),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        required=True,
        help='USOD10K root containing TR/VAL/TE, or a parent with nested USOD10k/USOD10k.')
    parser.add_argument(
        '--out-root',
        required=True,
        help='Output self-contained COCO detection dataset root.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Splits to convert.')
    parser.add_argument(
        '--mask-thr',
        type=int,
        default=10,
        help='Foreground threshold for GT mask pixels.')
    parser.add_argument(
        '--min-area',
        type=int,
        default=16,
        help='Drop connected components smaller than this area in pixels.')
    parser.add_argument(
        '--single-box',
        action='store_true',
        help='Use one bbox around the whole foreground mask instead of connected components.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite copied images and annotation files.')
    return parser.parse_args()


def find_dataset_root(root):
    root = Path(root)
    candidates = [
        root,
        root / 'USOD10k',
        root / 'USOD10k' / 'USOD10k',
        root / 'USOD10K',
        root / 'USOD10K' / 'USOD10K',
    ]
    for candidate in candidates:
        if (candidate / 'TR').is_dir() and (candidate / 'VAL').is_dir():
            return candidate
    raise FileNotFoundError(
        'Cannot find USOD10K split root with TR/VAL under: ' + str(root))


def join_parts(root, parts):
    path = Path(root)
    for part in parts:
        path = path / part
    return path


def list_pngs(path):
    return sorted(Path(path).glob('*.png'), key=lambda p: p.stem)


def mask_to_components(mask, min_area):
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=4)
        components = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id].tolist()
            if area >= min_area:
                components.append((x, y, x + w - 1, y + h - 1, area))
        return components

    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    ys, xs = np.where(mask)

    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        q = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        min_x = max_x = start_x
        min_y = max_y = start_y
        area = 0

        while q:
            y, x = q.popleft()
            area += 1
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))

        if area >= min_area:
            components.append((min_x, min_y, max_x, max_y, area))
    return components


def mask_to_bboxes(mask_path, threshold, min_area, single_box):
    mask_img = Image.open(mask_path).convert('L')
    arr = np.asarray(mask_img)
    fg = arr > threshold
    if not fg.any():
        return []

    if single_box:
        ys, xs = np.where(fg)
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        area = int(fg.sum())
        comps = [(min_x, min_y, max_x, max_y, area)]
    else:
        comps = mask_to_components(fg, min_area)

    bboxes = []
    for min_x, min_y, max_x, max_y, area in comps:
        bbox = [
            float(min_x),
            float(min_y),
            float(max_x - min_x + 1),
            float(max_y - min_y + 1),
        ]
        if bbox[2] * bbox[3] >= min_area:
            bboxes.append((bbox, int(area)))
    return bboxes


def convert_split(split, dataset_root, out_img_dir, args, image_start_id, ann_start_id):
    split_dir, rgb_parts, gt_parts, prefix = SPLIT_INFO[split]
    rgb_dir = join_parts(dataset_root / split_dir, rgb_parts)
    gt_dir = join_parts(dataset_root / split_dir, gt_parts)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f'Missing RGB dir for {split}: {rgb_dir}')
    if not gt_dir.is_dir():
        raise FileNotFoundError(f'Missing GT dir for {split}: {gt_dir}')

    rgb_files = list_pngs(rgb_dir)
    gt_by_stem = {p.stem: p for p in list_pngs(gt_dir)}
    images = []
    annotations = []
    image_id = image_start_id
    ann_id = ann_start_id
    missing_gt = 0
    empty_masks = 0

    for rgb_path in tqdm(rgb_files, desc=f'USOD10K {split}', unit='img'):
        gt_path = gt_by_stem.get(rgb_path.stem)
        if gt_path is None:
            missing_gt += 1
            continue
        bboxes = mask_to_bboxes(
            gt_path,
            threshold=args.mask_thr,
            min_area=args.min_area,
            single_box=args.single_box)
        if not bboxes:
            empty_masks += 1
            continue

        with Image.open(rgb_path) as img:
            width, height = img.size

        file_name = f'{prefix}_{rgb_path.name}'
        dst_path = out_img_dir / file_name
        if args.overwrite or not dst_path.exists():
            shutil.copy2(rgb_path, dst_path)

        images.append({
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height,
            'source_split': split,
        })

        for bbox, area in bboxes:
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [round(float(v), 3) for v in bbox],
                'area': float(area),
                'iscrowd': 0,
            })
            ann_id += 1
        image_id += 1

    print(
        f'{split}: rgb={len(rgb_files)}, images={len(images)}, '
        f'anns={len(annotations)}, missing_gt={missing_gt}, empty_masks={empty_masks}')
    return images, annotations, image_id, ann_id


def save_coco(path, images, annotations):
    data = {
        'info': {'description': 'USOD10K saliency masks converted to detection bboxes'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f'Output: {path} images={len(images)} anns={len(annotations)}')


def main():
    args = parse_args()
    dataset_root = find_dataset_root(args.root)
    out_root = Path(args.out_root)
    out_img_dir = out_root / 'images'
    ann_dir = out_root / 'annotations'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f'Dataset root: {dataset_root}')
    print(f'Output root: {out_root}')
    print(f'Splits: {args.splits}')

    split_outputs = {}
    image_id = 1
    ann_id = 1
    for split in args.splits:
        images, annotations, image_id, ann_id = convert_split(
            split, dataset_root, out_img_dir, args, image_id, ann_id)
        split_outputs[split] = (images, annotations)
        out_name = {
            'train': 'instances_train.json',
            'val': 'instances_val.json',
            'test': 'instances_test.json',
        }[split]
        save_coco(ann_dir / out_name, images, annotations)

    if 'train' in split_outputs and 'val' in split_outputs:
        train_images, train_anns = split_outputs['train']
        val_images, val_anns = split_outputs['val']
        save_coco(
            ann_dir / 'instances_trainval.json',
            train_images + val_images,
            train_anns + val_anns)


if __name__ == '__main__':
    main()
