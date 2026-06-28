#!/usr/bin/env python3
"""Build DFUI-based S1 sources with optional RUOD/UIIS easy data.

This is a small wrapper around ``merge_dfui_ruod_uiis_easy.py`` so the J10
source-comparison experiments can create:

* DFUI_ALL
* DFUI_RUOD_EASY
* DFUI_RUOD_UIIS_EASY

with the same unified category space and train/val split logic.
"""

import argparse
import random
from pathlib import Path

from merge_dfui_ruod_uiis_easy import (
    build_split,
    collect_dfui_items,
    collect_items,
    resolve_dfui_ann_paths,
    save_coco,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dfui-img-dir',
        default='/media/HDD0/XCX/exp_2/dfui/images')
    parser.add_argument(
        '--dfui-ann',
        nargs='+',
        default=None,
        help='One or more DFUI COCO annotation files.')
    parser.add_argument(
        '--include-ruod-easy',
        action='store_true',
        help='Append RUOD easy samples to DFUI.')
    parser.add_argument(
        '--ruod-easy-img-dir',
        default='/media/HDD0/XCX/exp_2/RUOD/coco/train')
    parser.add_argument(
        '--ruod-easy-ann',
        default='/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json')
    parser.add_argument(
        '--include-uiis-easy',
        action='store_true',
        help='Append UIIS10K easy samples. Usually used with RUOD easy.')
    parser.add_argument(
        '--uiis-easy-img-dir',
        default='/media/HDD0/XCX/exp_2/UIIS10K/img')
    parser.add_argument(
        '--uiis-easy-ann',
        default='/media/HDD0/XCX/exp_2/UIIS10K/coco/annotations/cross_split_det/easy_merged.json')
    parser.add_argument(
        '--out-root',
        required=True)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    out_root = Path(args.out_root)
    out_img_dir = out_root / 'images'
    out_ann_dir = out_root / 'annotations'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    dfui_ann_paths = resolve_dfui_ann_paths(args.dfui_ann)
    print('DFUI annotation files:')
    for ann_path in dfui_ann_paths:
        print(f'  {ann_path}')

    items = collect_dfui_items(args.dfui_img_dir, dfui_ann_paths)

    if args.include_ruod_easy:
        items.extend(collect_items(
            'ruod_easy', args.ruod_easy_img_dir, args.ruod_easy_ann))

    if args.include_uiis_easy:
        items.extend(collect_items(
            'uiis_easy', args.uiis_easy_img_dir, args.uiis_easy_ann))

    if not items:
        raise RuntimeError('No images/annotations collected.')

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
