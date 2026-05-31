#!/usr/bin/env python3
"""Convert UOT100 tracking annotations to one COCO detection JSON.

UOT100 layout expected by this converter:

    UOT100/
      SequenceName1/
        img/
          1.jpg
          2.jpg
          ...
        groundtruth_rect.txt
      SequenceName2/
        img/
        groundtruth_rect.txt

Each line in ``groundtruth_rect.txt`` is interpreted as ``x y w h`` and mapped
to the numerically corresponding frame. By default, sequence names ending in a
repeat index are merged into one category, e.g. ``ArmyDiver1`` and
``ArmyDiver2`` both become ``ArmyDiver``.
"""

import argparse
import json
import os
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        required=True,
        help='Root directory containing UOT100 sequence folders.')
    parser.add_argument(
        '--out',
        required=True,
        help='Output COCO json path, e.g. annotations/instances_all.json.')
    parser.add_argument(
        '--keep-trailing-digits',
        action='store_true',
        help='Keep sequence names such as ArmyDiver1 as separate categories.')
    parser.add_argument(
        '--skip-invalid',
        action='store_true',
        default=True,
        help='Skip invalid or zero-area boxes.')
    return parser.parse_args()


def natural_key(path):
    stem = Path(path).stem
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', stem)]


def category_name_from_sequence(sequence_name, keep_trailing_digits=False):
    if keep_trailing_digits:
        return sequence_name
    return re.sub(r'\d+$', '', sequence_name)


def parse_bbox_line(line):
    parts = [x for x in re.split(r'[\s,]+', line.strip()) if x]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in parts[:4]]
    except ValueError:
        return None
    return [x, y, w, h]


def read_bboxes(path):
    bboxes = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            bbox = parse_bbox_line(line)
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def list_frames(img_dir):
    frames = []
    for path in Path(img_dir).iterdir():
        if path.is_file() and path.suffix.lower() in IMG_SUFFIXES:
            frames.append(path)
    return sorted(frames, key=natural_key)


def image_size(path):
    with Image.open(path) as img:
        return img.width, img.height


def clamp_bbox(bbox, width, height):
    x, y, w, h = bbox
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + max(0.0, w))
    y2 = min(float(height), y + max(0.0, h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def main():
    args = parse_args()
    root = Path(args.root)
    out_path = Path(args.out)

    sequences = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / 'img').is_dir()
        and (path / 'groundtruth_rect.txt').is_file()
    ]
    if not sequences:
        raise RuntimeError(f'No UOT100 sequence folders found under {root}')

    category_names = sorted({
        category_name_from_sequence(seq.name, args.keep_trailing_digits)
        for seq in sequences
    })
    cat_id_by_name = {name: idx + 1 for idx, name in enumerate(category_names)}

    coco = {
        'info': {'description': 'UOT100 tracking converted to COCO detection'},
        'licenses': [],
        'categories': [
            {'id': cat_id_by_name[name], 'name': name}
            for name in category_names
        ],
        'images': [],
        'annotations': [],
    }

    image_id = 1
    ann_id = 1
    skipped_invalid = 0
    skipped_mismatch = 0

    for seq in tqdm(sequences, desc='convert UOT100'):
        category_name = category_name_from_sequence(seq.name, args.keep_trailing_digits)
        category_id = cat_id_by_name[category_name]
        frames = list_frames(seq / 'img')
        bboxes = read_bboxes(seq / 'groundtruth_rect.txt')
        pair_count = min(len(frames), len(bboxes))
        skipped_mismatch += abs(len(frames) - len(bboxes))

        for frame_path, bbox in zip(frames[:pair_count], bboxes[:pair_count]):
            width, height = image_size(frame_path)
            clipped_bbox = clamp_bbox(bbox, width, height)
            area = clipped_bbox[2] * clipped_bbox[3]
            if args.skip_invalid and area <= 0:
                skipped_invalid += 1
                continue

            rel_file = f'{seq.name}/img/{frame_path.name}'
            coco['images'].append({
                'id': image_id,
                'file_name': rel_file.replace('\\', '/'),
                'width': width,
                'height': height,
                'sequence': seq.name,
            })
            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [round(float(v), 3) for v in clipped_bbox],
                'area': round(float(area), 3),
                'iscrowd': 0,
            })
            image_id += 1
            ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f'Sequences: {len(sequences)}')
    print(f'Categories: {len(coco["categories"])}')
    print(f'Images: {len(coco["images"])}')
    print(f'Annotations: {len(coco["annotations"])}')
    print(f'Skipped invalid boxes: {skipped_invalid}')
    print(f'Frame/bbox count mismatches skipped: {skipped_mismatch}')
    print(f'Output: {out_path}')


if __name__ == '__main__':
    main()
