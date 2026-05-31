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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


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
    print(f'Loading: {ann_path}')
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    anns_by_img = {img_id: [] for img_id in images_by_id}
    for ann in tqdm(
            coco.get('annotations', []),
            desc=f'{Path(ann_path).name} group annotations',
            unit='ann'):
        if ann.get('image_id') in anns_by_img:
            anns_by_img[ann['image_id']].append(ann)

    keep_img_ids = set()
    max_ratio_by_img = {}
    for img_id, anns in tqdm(
            anns_by_img.items(),
            desc=f'{Path(ann_path).name} select images',
            unit='img'):
        image_info = images_by_id[img_id]
        ratios = [bbox_ratio(ann, image_info) for ann in anns]
        max_ratio = max(ratios) if ratios else 0.0
        max_ratio_by_img[img_id] = max_ratio
        if max_ratio >= threshold:
            keep_img_ids.add(img_id)

    out = deepcopy(coco)
    out['images'] = [img for img in coco.get('images', []) if img['id'] in keep_img_ids]

    filtered_anns = []
    for ann in tqdm(
            coco.get('annotations', []),
            desc=f'{Path(ann_path).name} write annotations',
            unit='ann'):
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
    input_images = len(coco.get('images', []))
    input_annotations = len(coco.get('annotations', []))
    output_images = len(out['images'])
    output_annotations = len(out['annotations'])
    print(f'{ann_path}')
    print(f'  images: {input_images} -> {output_images}')
    print(f'  annotations: {input_annotations} -> {output_annotations}')
    print(f'  threshold: {threshold}')
    if ratios:
        min_ratio = min(ratios)
        mean_ratio = sum(ratios) / len(ratios)
        max_ratio = max(ratios)
        print(
            '  max-ratio stats: '
            f'min={min_ratio:.4f}, mean={mean_ratio:.4f}, '
            f'max={max_ratio:.4f}')
    else:
        min_ratio = mean_ratio = max_ratio = 0.0
    print(f'  output: {out_path}')

    keep_ratio = output_images / input_images if input_images else 0.0
    return {
        'input': ann_path,
        'output': out_path,
        'input_images': input_images,
        'output_images': output_images,
        'input_annotations': input_annotations,
        'output_annotations': output_annotations,
        'keep_ratio': keep_ratio,
        'min_ratio': min_ratio,
        'mean_ratio': mean_ratio,
        'max_ratio': max_ratio,
    }


def print_summary(stats):
    if not stats:
        return

    print('\nSummary')
    print('=' * 120)
    header = (
        f'{"input":50} {"images":>17} {"anns":>17} '
        f'{"keep%":>8} {"max":>8} {"output"}')
    print(header)
    print('-' * 120)
    total_in_images = 0
    total_out_images = 0
    total_in_anns = 0
    total_out_anns = 0
    for item in stats:
        total_in_images += item['input_images']
        total_out_images += item['output_images']
        total_in_anns += item['input_annotations']
        total_out_anns += item['output_annotations']
        name = Path(item['input']).name
        if len(name) > 50:
            name = '...' + name[-47:]
        print(
            f'{name:50} '
            f'{item["input_images"]:>8}->{item["output_images"]:<8} '
            f'{item["input_annotations"]:>8}->{item["output_annotations"]:<8} '
            f'{item["keep_ratio"] * 100:>7.2f}% '
            f'{item["max_ratio"]:>8.4f} '
            f'{item["output"]}')

    total_keep_ratio = total_out_images / total_in_images if total_in_images else 0.0
    print('-' * 120)
    print(
        f'{"TOTAL":50} '
        f'{total_in_images:>8}->{total_out_images:<8} '
        f'{total_in_anns:>8}->{total_out_anns:<8} '
        f'{total_keep_ratio * 100:>7.2f}%')


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

    pairs = list(zip(ann_paths, out_paths))
    stats = []
    for ann_path, out_path in tqdm(pairs, desc='COCO files', unit='file'):
        stats.append(
            filter_coco(
                ann_path=ann_path,
                out_path=out_path,
                threshold=args.threshold,
                keep_only_large_anns=args.keep_only_large_anns))
    print_summary(stats)


if __name__ == '__main__':
    main()
