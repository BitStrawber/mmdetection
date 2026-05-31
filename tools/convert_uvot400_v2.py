#!/usr/bin/env python3
"""Convert UVOT400 tracking annotations to COCO and filter large boxes."""

import argparse
import glob
import json
import os
import re
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMG_PATTERNS = (
    '*.jpg', '*.jpeg', '*.png', '*.bmp',
    '*.JPG', '*.JPEG', '*.PNG', '*.BMP',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/media/HDD1/XCX/exp_2/UVOT400')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate instances_*.json even if it already exists.')
    return parser.parse_args()


def collect_sequences(data_dir):
    seq_dirs = []
    for root, _, files in os.walk(data_dir):
        if 'groundtruth_rect.txt' in files:
            seq_dirs.append(root)
    return sorted(seq_dirs)


def collect_images(seq_dir):
    candidate_dirs = [
        Path(seq_dir) / 'imgs',
        Path(seq_dir) / 'images',
        Path(seq_dir) / 'img',
        Path(seq_dir),
    ]
    image_files = []
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        for pattern in IMG_PATTERNS:
            image_files.extend(glob.glob(str(directory / '**' / pattern), recursive=True))
    return sorted(set(image_files))


def parse_bbox_line(line):
    text = line.strip().replace('(', ' ').replace(')', ' ')
    parts = [p for p in re.split(r'[,\s]+', text) if p]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in parts[:4]]
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 1920, 1080


def convert_split(base_dir, split, force=False):
    data_dir = Path(base_dir) / split
    if not data_dir.is_dir():
        print(f'[skip] {data_dir} does not exist')
        return None

    out_json = data_dir / f'instances_{split}.json'
    if out_json.exists() and not force:
        print(f'[exists] {split}: {out_json}')
        return str(out_json)

    seq_dirs = collect_sequences(data_dir)
    print(f'UVOT400 {split}: found {len(seq_dirs)} sequences')

    images = []
    anns = []
    img_id = 0
    ann_id = 0
    skipped_lines = 0
    skipped_sequences = 0

    for seq_dir in tqdm(seq_dirs, desc=f'UVOT400 {split} sequences', unit='seq'):
        gt_file = Path(seq_dir) / 'groundtruth_rect.txt'
        img_files = collect_images(seq_dir)
        if not img_files:
            skipped_sequences += 1
            continue

        with open(gt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]

        frame_count = min(len(lines), len(img_files))
        seq_name = os.path.basename(seq_dir)
        for i in tqdm(
                range(frame_count),
                desc=f'UVOT400 {split} {seq_name}',
                unit='frame',
                leave=False):
            bbox = parse_bbox_line(lines[i])
            if bbox is None:
                skipped_lines += 1
                continue

            width, height = image_size(img_files[i])
            images.append({
                'id': img_id,
                'file_name': os.path.relpath(img_files[i], data_dir),
                'width': width,
                'height': height,
            })
            anns.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
            })
            img_id += 1
            ann_id += 1

    if not images:
        print(f'UVOT400 {split}: no valid data')
        return None

    coco = {
        'info': {'description': f'UVOT400_{split}'},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'object'}],
        'images': images,
        'annotations': anns,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(
        f'UVOT400 {split}: {len(images)} imgs, {len(anns)} anns, '
        f'skipped_sequences={skipped_sequences}, skipped_lines={skipped_lines} -> {out_json}')
    return str(out_json)


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


def filter_coco(out_json, threshold):
    with open(out_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_map = {img['id']: img for img in coco.get('images', [])}
    img_max = {}
    for ann in tqdm(coco.get('annotations', []), desc='UVOT400 filter annotations', unit='ann'):
        img_id = ann.get('image_id')
        bbox = valid_bbox(ann.get('bbox'))
        if img_id not in img_map or bbox is None:
            continue
        area = bbox[2] * bbox[3]
        if area > img_max.get(img_id, 0.0):
            img_max[img_id] = area

    keep = set()
    for img_id, max_area in tqdm(img_max.items(), desc='UVOT400 select images', unit='img'):
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
                desc='UVOT400 write filtered anns',
                unit='ann')
            if ann.get('image_id') in keep
        ],
    }

    out_filtered = out_json.replace('.json', f'_bbox{int(threshold * 100)}pct.json')
    with open(out_filtered, 'w', encoding='utf-8') as f:
        json.dump(filtered, f)

    total = len(coco.get('images', []))
    kept = len(filtered['images'])
    ratio = kept / total * 100 if total else 0.0
    print(f'UVOT400 filter: {total} -> {kept} ({ratio:.1f}%)')
    print(f'  output: {out_filtered}')


def main():
    args = parse_args()
    for split in args.splits:
        out_json = convert_split(args.base, split, force=args.force)
        if out_json:
            filter_coco(out_json, args.threshold)


if __name__ == '__main__':
    main()
