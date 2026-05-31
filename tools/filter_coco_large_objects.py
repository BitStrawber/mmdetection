#!/usr/bin/env python3
"""Filter COCO annotations by largest object area ratio per image.

An image is kept when at least one bbox satisfies:

    bbox_width * bbox_height / (image_width * image_height) >= threshold

All annotations belonging to kept images are retained by default.
"""

import argparse
import glob
import json
import os
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', nargs='*', default=None, help='Input COCO json(s).')
    parser.add_argument('--out', nargs='*', default=None, help='Output COCO json(s).')
    parser.add_argument(
        '--ann-glob',
        default=None,
        help='Glob for batch mode, e.g. "/data/*/annotations/instances_*.json".')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output dir for batch mode. Keeps input basenames.')
    parser.add_argument(
        '--suffix',
        default='_large20',
        help='Suffix inserted before .json in batch mode.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
        help='Minimum largest bbox / image area ratio.')
    parser.add_argument(
        '--keep-only-large-anns',
        action='store_true',
        help='Keep only annotations whose bbox ratio passes the threshold.')
    return parser.parse_args()


def output_path_for(input_path, out_dir, suffix):
    path = Path(input_path)
    return str(Path(out_dir) / f'{path.stem}{suffix}{path.suffix}')


def bbox_ratio(ann, image_info):
    bbox = ann.get('bbox')
    if not bbox or len(bbox) != 4:
        return 0.0
    width = float(image_info.get('width') or 0)
    height = float(image_info.get('height') or 0)
    if width <= 0 or height <= 0:
        return 0.0
    box_w = max(0.0, float(bbox[2]))
    box_h = max(0.0, float(bbox[3]))
    return (box_w * box_h) / (width * height)


def filter_coco(ann_path, out_path, threshold, keep_only_large_anns=False):
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    anns_by_img = {img_id: [] for img_id in images_by_id}
    for ann in coco.get('annotations', []):
        if ann.get('image_id') in anns_by_img:
            anns_by_img[ann['image_id']].append(ann)

    keep_img_ids = set()
    max_ratio_by_img = {}
    for img_id, anns in anns_by_img.items():
        image_info = images_by_id[img_id]
        ratios = [bbox_ratio(ann, image_info) for ann in anns]
        max_ratio = max(ratios) if ratios else 0.0
        max_ratio_by_img[img_id] = max_ratio
        if max_ratio >= threshold:
            keep_img_ids.add(img_id)

    out = deepcopy(coco)
    out['images'] = [img for img in coco.get('images', []) if img['id'] in keep_img_ids]

    filtered_anns = []
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in keep_img_ids:
            continue
        if keep_only_large_anns and bbox_ratio(ann, images_by_id[img_id]) < threshold:
            continue
        filtered_anns.append(ann)
    out['annotations'] = filtered_anns

    out.setdefault('info', {})
    out['info']['large_object_filter'] = {
        'source_annotation': ann_path,
        'threshold': threshold,
        'criterion': 'max_bbox_area / image_area >= threshold',
        'keep_only_large_anns': keep_only_large_anns,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)

    ratios = list(max_ratio_by_img.values())
    print(f'{ann_path}')
    print(f'  images: {len(coco.get("images", []))} -> {len(out["images"])}')
    print(f'  annotations: {len(coco.get("annotations", []))} -> {len(out["annotations"])}')
    print(f'  threshold: {threshold}')
    if ratios:
        print(
            '  max-ratio stats: '
            f'min={min(ratios):.4f}, mean={sum(ratios) / len(ratios):.4f}, '
            f'max={max(ratios):.4f}')
    print(f'  output: {out_path}')


def main():
    args = parse_args()

    if args.ann_glob:
        if not args.out_dir:
            raise ValueError('--out-dir is required with --ann-glob')
        ann_paths = sorted(glob.glob(args.ann_glob, recursive=True))
        if not ann_paths:
            raise FileNotFoundError(f'No files matched: {args.ann_glob}')
        out_paths = [output_path_for(p, args.out_dir, args.suffix) for p in ann_paths]
    else:
        if not args.ann or not args.out:
            raise ValueError('Use --ann/--out or --ann-glob/--out-dir')
        if len(args.ann) != len(args.out):
            raise ValueError('--ann and --out must have the same length')
        ann_paths = args.ann
        out_paths = args.out

    for ann_path, out_path in zip(ann_paths, out_paths):
        filter_coco(
            ann_path=ann_path,
            out_path=out_path,
            threshold=args.threshold,
            keep_only_large_anns=args.keep_only_large_anns)


if __name__ == '__main__':
    main()
