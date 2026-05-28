#!/usr/bin/env python3
"""Merge DFUI + RUOD easy + UIIS10K easy into one detection source.

The output is a COCO detection dataset with a unified 11-class label space:
RUOD's 10 classes plus DFUI waterweeds. Images are copied into one image
directory and annotations are split into train/val so S1 can select best epoch.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


UNIFIED_CATEGORIES = [
    {'id': 1, 'name': 'holothurian'},
    {'id': 2, 'name': 'echinus'},
    {'id': 3, 'name': 'scallop'},
    {'id': 4, 'name': 'starfish'},
    {'id': 5, 'name': 'fish'},
    {'id': 6, 'name': 'corals'},
    {'id': 7, 'name': 'diver'},
    {'id': 8, 'name': 'cuttlefish'},
    {'id': 9, 'name': 'turtle'},
    {'id': 10, 'name': 'jellyfish'},
    {'id': 11, 'name': 'waterweeds'},
]

NAME_TO_ID = {cat['name']: cat['id'] for cat in UNIFIED_CATEGORIES}

# Historical DFUI annotations in this workspace used 0-based ids.
DFUI_ID_TO_NAME = {
    0: 'echinus',
    1: 'holothurian',
    2: 'scallop',
    3: 'starfish',
    4: 'waterweeds',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dfui-img-dir',
        default='/media/HDD0/XCX/exp_2/dfui/images')
    parser.add_argument(
        '--dfui-ann',
        default='/media/HDD0/XCX/exp_2/dfui/annotations/instances_trainval2017.json')
    parser.add_argument(
        '--ruod-easy-img-dir',
        default='/media/HDD0/XCX/exp_2/RUOD/coco/train')
    parser.add_argument(
        '--ruod-easy-ann',
        default='/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json')
    parser.add_argument(
        '--uiis-easy-img-dir',
        default='/media/HDD0/XCX/exp_2/UIIS10K/img')
    parser.add_argument(
        '--uiis-easy-ann',
        default='/media/HDD0/XCX/exp_2/UIIS10K/coco/annotations/cross_split_det/easy_merged.json')
    parser.add_argument(
        '--out-root',
        default='/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing copied images and annotation jsons.')
    return parser.parse_args()


def load_coco(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_coco(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def source_category_map(coco, source_name):
    id_to_name = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    mapping = {}
    for cat_id, cat_name in id_to_name.items():
        name = str(cat_name).lower()
        if name in NAME_TO_ID:
            mapping[cat_id] = NAME_TO_ID[name]
        elif source_name == 'dfui' and cat_id in DFUI_ID_TO_NAME:
            mapping[cat_id] = NAME_TO_ID[DFUI_ID_TO_NAME[cat_id]]
    return mapping


def collect_items(source_name, img_dir, ann_path):
    coco = load_coco(ann_path)
    cat_map = source_category_map(coco, source_name)
    anns_by_img = defaultdict(list)
    skipped_anns = 0
    for ann in coco.get('annotations', []):
        new_cat = cat_map.get(ann['category_id'])
        if new_cat is None:
            skipped_anns += 1
            continue
        new_ann = dict(ann)
        new_ann['category_id'] = new_cat
        new_ann.pop('segmentation', None)
        anns_by_img[ann['image_id']].append(new_ann)

    items = []
    for img in coco.get('images', []):
        anns = anns_by_img.get(img['id'], [])
        if not anns:
            continue
        src_path = Path(img_dir) / os.path.basename(img['file_name'])
        if not src_path.exists():
            src_path = Path(img_dir) / img['file_name']
        if not src_path.exists():
            print(f'Warning: missing image skipped: {src_path}')
            continue
        items.append((source_name, src_path, img, anns))

    print(
        f'{source_name}: {len(items)} images, '
        f'{sum(len(x[3]) for x in items)} anns, skipped_anns={skipped_anns}')
    return items


def build_split(items, split_name, out_img_dir, overwrite):
    images = []
    annotations = []
    ann_id = 1
    for img_id, (source_name, src_path, img, anns) in enumerate(
            tqdm(items, desc=split_name), start=1):
        file_name = f'{source_name}_{os.path.basename(img["file_name"])}'
        dst_path = out_img_dir / file_name
        if overwrite or not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        images.append({
            'id': img_id,
            'file_name': file_name,
            'width': img['width'],
            'height': img['height'],
        })
        for ann in anns:
            new_ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0),
            }
            annotations.append(new_ann)
            ann_id += 1

    return {
        'info': {'description': f'DFUI + RUOD easy + UIIS10K easy {split_name}'},
        'licenses': [],
        'categories': UNIFIED_CATEGORIES,
        'images': images,
        'annotations': annotations,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    out_root = Path(args.out_root)
    out_img_dir = out_root / 'images'
    out_ann_dir = out_root / 'annotations'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    items = []
    items.extend(collect_items('dfui', args.dfui_img_dir, args.dfui_ann))
    items.extend(collect_items('ruod_easy', args.ruod_easy_img_dir, args.ruod_easy_ann))
    items.extend(collect_items('uiis_easy', args.uiis_easy_img_dir, args.uiis_easy_ann))

    random.shuffle(items)
    val_count = max(1, int(len(items) * args.val_ratio))
    val_items = items[:val_count]
    train_items = items[val_count:]

    train_coco = build_split(train_items, 'train', out_img_dir, args.overwrite)
    val_coco = build_split(val_items, 'val', out_img_dir, args.overwrite)
    all_coco = build_split(items, 'all', out_img_dir, args.overwrite)

    save_coco(train_coco, out_ann_dir / 'instances_train.json')
    save_coco(val_coco, out_ann_dir / 'instances_val.json')
    save_coco(all_coco, out_ann_dir / 'instances_all.json')

    print(f'Output root: {out_root}')
    print(f'train: {len(train_coco["images"])} images, {len(train_coco["annotations"])} anns')
    print(f'val: {len(val_coco["images"])} images, {len(val_coco["annotations"])} anns')
    print(f'all: {len(all_coco["images"])} images, {len(all_coco["annotations"])} anns')


if __name__ == '__main__':
    main()
