#!/usr/bin/env python3
"""Create a deterministic RUOD image sample for fixed-GT CAM analysis."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import atomic_write_json, atomic_write_jsonl, existing_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--minimum-instances', type=int, default=1)
    parser.add_argument(
        '--materialize', choices=('none', 'symlink', 'copy'), default='none')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.minimum_instances <= 0:
        raise ValueError('samples and minimum-instances must be positive')
    annotation_file = existing_file(args.annotation_file)
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with annotation_file.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)
    annotations: Dict[int, List[dict]] = defaultdict(list)
    for annotation in coco.get('annotations', []):
        x, y, width, height = [float(value) for value in annotation['bbox']]
        if width > 0 and height > 0 and not int(annotation.get('iscrowd', 0)):
            annotations[int(annotation['image_id'])].append(annotation)
    candidates = []
    for image in coco.get('images', []):
        image_id = int(image['id'])
        if len(annotations.get(image_id, [])) < args.minimum_instances:
            continue
        source = (image_root / image['file_name']).resolve()
        if source.is_file() and source.stat().st_size > 0:
            candidates.append((image, source))
    candidates.sort(key=lambda pair: (int(pair[0]['id']), pair[0]['file_name']))
    if len(candidates) < args.samples:
        raise RuntimeError(f'Only {len(candidates)} valid images; requested {args.samples}')
    selected = random.Random(args.seed).sample(candidates, args.samples)
    selected.sort(key=lambda pair: (int(pair[0]['id']), pair[0]['file_name']))
    selected_ids = {int(image['id']) for image, _ in selected}
    rows = []
    for sample_index, (image, source) in enumerate(selected):
        image_id = int(image['id'])
        destination = out_dir / 'images' / Path(image['file_name'])
        if args.materialize == 'copy':
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            selected_path = destination.resolve()
        elif args.materialize == 'symlink':
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            destination.symlink_to(source)
            selected_path = destination.absolute()
        else:
            selected_path = source
        with Image.open(source) as opened:
            actual_width, actual_height = opened.size
        rows.append({
            'sample_index': sample_index,
            'image_id': image_id,
            'file_name': str(image['file_name']),
            'image_path': str(selected_path),
            'source_path': str(source),
            'width': int(image.get('width', actual_width)),
            'height': int(image.get('height', actual_height)),
            'actual_width': actual_width,
            'actual_height': actual_height,
            'annotation_ids': sorted(
                int(item['id']) for item in annotations[image_id]),
            'annotation_count': len(annotations[image_id]),
        })
    subset = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': [image for image, _ in selected],
        'annotations': [
            item for item in coco.get('annotations', [])
            if int(item['image_id']) in selected_ids
        ],
    }
    atomic_write_jsonl(out_dir / 'manifest.jsonl', rows)
    atomic_write_json(out_dir / 'annotations.coco.json', subset)
    atomic_write_json(out_dir / 'sampling.json', {
        'annotation_file': str(annotation_file),
        'image_root': str(image_root),
        'samples': args.samples,
        'seed': args.seed,
        'minimum_instances': args.minimum_instances,
        'valid_candidates': len(candidates),
        'materialize': args.materialize,
        'selected_annotation_count': len(subset['annotations']),
    })
    print(f'Sampled {len(rows)} RUOD images into {out_dir}')
    print(f'Manifest: {out_dir / "manifest.jsonl"}')


if __name__ == '__main__':
    main()
