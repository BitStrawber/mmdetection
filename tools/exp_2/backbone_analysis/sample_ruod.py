#!/usr/bin/env python3
"""Create a deterministic annotated RUOD sample and reusable manifest."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .common import ensure_empty_or_create, existing_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-file', required=True, help='COCO annotation JSON')
    parser.add_argument('--image-root', required=True, help='Root used by image file_name')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--require-annotations', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--materialize', choices=('copy', 'symlink', 'none'), default='copy',
        help='How selected images are represented below OUT_DIR/images')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_image(image_root: Path, file_name: str) -> Path:
    path = (image_root / file_name).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return path


def materialize(source: Path, destination: Path, mode: str) -> Path:
    if mode == 'none':
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == 'copy':
        shutil.copy2(str(source), str(destination))
    else:
        destination.symlink_to(source)
    return destination.resolve() if mode == 'copy' else destination.absolute()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError('--samples must be positive')
    annotation_file = existing_file(args.annotation_file)
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)

    with annotation_file.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)
    annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
    for annotation in coco.get('annotations', []):
        image_id = int(annotation['image_id'])
        if float(annotation.get('area', 0.0)) > 0:
            annotations_by_image[image_id].append(annotation)

    candidates = []
    missing = []
    for image in coco.get('images', []):
        image_id = int(image['id'])
        annotations = annotations_by_image.get(image_id, [])
        if args.require_annotations and not annotations:
            continue
        try:
            path = resolve_image(image_root, image['file_name'])
        except FileNotFoundError:
            missing.append(image['file_name'])
            continue
        candidates.append((image, annotations, path))
    candidates.sort(key=lambda item: (int(item[0]['id']), item[0]['file_name']))
    if len(candidates) < args.samples:
        raise RuntimeError(
            f'Only {len(candidates)} valid candidate images; requested {args.samples}')

    selected = random.Random(args.seed).sample(candidates, args.samples)
    selected.sort(key=lambda item: (int(item[0]['id']), item[0]['file_name']))
    rows = []
    subset_images = []
    subset_annotations = []
    for sample_index, (image, annotations, source) in enumerate(selected):
        relative_name = Path(image['file_name'])
        destination = out_dir / 'images' / relative_name
        selected_path = materialize(source, destination, args.materialize)
        with Image.open(source) as opened:
            actual_width, actual_height = opened.size
        boxes = []
        labels = []
        annotation_ids = []
        for annotation in annotations:
            x, y, width, height = [float(value) for value in annotation['bbox']]
            boxes.append([x, y, x + width, y + height])
            labels.append(int(annotation['category_id']))
            annotation_ids.append(int(annotation['id']))
            subset_annotations.append(annotation)
        rows.append({
            'sample_index': sample_index,
            'image_id': int(image['id']),
            'file_name': image['file_name'],
            'source_path': str(source),
            'image_path': str(selected_path),
            'width': int(image.get('width', actual_width)),
            'height': int(image.get('height', actual_height)),
            'actual_width': actual_width,
            'actual_height': actual_height,
            'annotation_ids': annotation_ids,
            'class_ids': sorted(set(labels)),
            'boxes_xyxy': boxes,
            'materialization': args.materialize,
        })
        subset_images.append(image)

    subset_coco = {
        key: coco.get(key, []) for key in ('info', 'licenses', 'categories')
    }
    subset_coco['images'] = subset_images
    subset_coco['annotations'] = subset_annotations
    write_jsonl(out_dir / 'manifest.jsonl', rows)
    write_json(out_dir / 'annotations.coco.json', subset_coco)
    write_json(out_dir / 'sampling.json', {
        'annotation_file': str(annotation_file),
        'image_root': str(image_root),
        'samples': args.samples,
        'seed': args.seed,
        'require_annotations': args.require_annotations,
        'materialize': args.materialize,
        'valid_candidates': len(candidates),
        'missing_images': len(missing),
    })
    print(f'Sampled {len(rows)} RUOD images into {out_dir}')
    print(f'Manifest: {out_dir / "manifest.jsonl"}')
    print(f'COCO subset: {out_dir / "annotations.coco.json"}')


if __name__ == '__main__':
    main()
