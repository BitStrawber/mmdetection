#!/usr/bin/env python3
"""Convert CoralSCOP per-image JSON annotations to COCO and filter large boxes."""

import argparse
import glob
import json
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMG_SUFFIXES = ('', '.jpg', '.jpeg', '.png', '.bmp', '.webp')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/media/HDD1/XCX/exp_2/CoralSCOP')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test'],
        help='Dataset splits to convert.')
    return parser.parse_args()


def find_jsons(data_dir, split):
    split_dir = Path(data_dir) / split
    jsons = sorted((split_dir / 'jsons').glob('*.json'))
    if not jsons:
        jsons = sorted(split_dir.glob('*.json'))
    return [str(path) for path in jsons]


def find_image_path(data_dir, split, json_path, file_name):
    candidates = []
    split_dir = Path(data_dir) / split
    json_dir = Path(json_path).parent
    for base in (split_dir / 'images', split_dir, json_dir):
        for suffix in IMG_SUFFIXES:
            name = file_name
            if suffix and not name.lower().endswith(suffix):
                name = f'{file_name}{suffix}'
            candidates.append(base / name)

    for path in candidates:
        if path.exists():
            return path
    return None


def valid_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def merge_coralscop(data_dir, split):
    jsons = find_jsons(data_dir, split)
    if not jsons:
        print(f'CoralSCOP {split}: no JSON files found')
        return None

    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    skipped_images = 0
    skipped_anns = 0

    for json_path in tqdm(jsons, desc=f'CoralSCOP {split} convert', unit='json'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_info = data.get('image')
        if not img_info and isinstance(data.get('images'), list) and data['images']:
            img_info = data['images'][0]
        if not isinstance(img_info, dict) or 'file_name' not in img_info:
            skipped_images += 1
            continue

        image_path = find_image_path(
            data_dir=data_dir,
            split=split,
            json_path=json_path,
            file_name=img_info['file_name'])
        if image_path is None:
            skipped_images += 1
            continue

        width = int(img_info.get('width') or 0)
        height = int(img_info.get('height') or 0)
        images.append({
            'id': img_id,
            'file_name': os.path.relpath(image_path, data_dir),
            'width': width,
            'height': height,
        })

        for ann in data.get('annotations', []):
            bbox = valid_bbox(ann.get('bbox'))
            if bbox is None:
                skipped_anns += 1
                continue
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': bbox,
                'area': float(ann.get('area') or bbox[2] * bbox[3]),
                'iscrowd': int(ann.get('iscrowd', 0)),
            })
            ann_id += 1
        img_id += 1

    coco = {
        'info': {'description': f'CoralSCOP {split}'},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'coral'}],
        'images': images,
        'annotations': annotations,
    }

    ann_dir = Path(data_dir) / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_path = ann_dir / f'instances_{split}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(
        f'CoralSCOP {split}: {len(images)} imgs, {len(annotations)} anns, '
        f'skipped_images={skipped_images}, skipped_anns={skipped_anns} -> {out_path}')
    return str(out_path)


def filter_coco(coco, threshold):
    img_map = {img['id']: img for img in coco.get('images', [])}
    img_max = {}
    for ann in tqdm(
            coco.get('annotations', []),
            desc='CoralSCOP filter annotations',
            unit='ann'):
        img_id = ann.get('image_id')
        bbox = valid_bbox(ann.get('bbox'))
        if img_id not in img_map or bbox is None:
            continue
        area = bbox[2] * bbox[3]
        if area > img_max.get(img_id, 0.0):
            img_max[img_id] = area

    keep = set()
    for img_id, max_area in tqdm(
            img_max.items(),
            desc='CoralSCOP select images',
            unit='img'):
        img = img_map[img_id]
        image_area = float(img.get('width') or 0) * float(img.get('height') or 0)
        if image_area > 0 and max_area / image_area >= threshold:
            keep.add(img_id)

    filtered = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': [img for img in coco.get('images', []) if img['id'] in keep],
        'annotations': [
            ann for ann in tqdm(
                coco.get('annotations', []),
                desc='CoralSCOP write filtered anns',
                unit='ann')
            if ann.get('image_id') in keep
        ],
    }
    return filtered, len(coco.get('images', [])), len(keep)


def main():
    args = parse_args()
    for split in args.splits:
        out_path = merge_coralscop(args.data_dir, split)
        if not out_path:
            continue

        with open(out_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        filtered, total, kept = filter_coco(coco, args.threshold)
        out_filtered = out_path.replace(
            '.json', f'_bbox{int(args.threshold * 100)}pct.json')
        with open(out_filtered, 'w', encoding='utf-8') as f:
            json.dump(filtered, f)
        ratio = kept / total * 100 if total else 0.0
        print(f'CoralSCOP {split} filter: {total} -> {kept} ({ratio:.1f}%)')
        print(f'  output: {out_filtered}')


if __name__ == '__main__':
    main()
