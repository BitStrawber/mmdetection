#!/usr/bin/env python3
"""Compose Fixed-GT image-level 5x4 panels from existing aggregate PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    compose_grid,
    draw_box,
    labeled_tile,
    load_rgb,
    parse_csv,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate-root', required=True)
    parser.add_argument(
        '--cam-root', default='',
        help=(
            'Fixed-GT raw index directory. Defaults to the sibling raw/ '
            'directory next to image_aggregate/.'),
    )
    parser.add_argument('--reference-model', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument(
        '--strategies', default='independent_p1_p99,imagenet_reference_p1_p99')
    parser.add_argument('--tile-width', type=int, default=480)
    parser.add_argument('--tile-height', type=int, default=360)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def find_source_image(root: Path, image_id: int, model: str, layer: str,
                      strategy: str) -> Path:
    path = root / strategy / model / layer / f'image_{image_id:08d}.png'
    if not path.is_file():
        raise FileNotFoundError(f'Missing aggregate image: {path}')
    return path


def main() -> None:
    args = parse_args()
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError('Tile dimensions must be positive')
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('PNG compression level must be in [0, 9]')

    root = Path(args.aggregate_root).expanduser().resolve()
    cam_root = (Path(args.cam_root).expanduser().resolve()
                if args.cam_root else root.parent / 'raw')
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    strategies = parse_csv(args.strategies)
    if not models or not layers or not strategies:
        raise ValueError('Models, layers and strategies must not be empty')
    if args.reference_model not in models:
        raise ValueError('--reference-model must be present in --models')

    raw_index = cam_root / 'raw_cam_index.jsonl'
    records = read_jsonl(raw_index)
    source_by_image = {}
    for row in records:
        if str(row.get('model')) != args.reference_model:
            continue
        image_id = int(row['image_id'])
        source_by_image.setdefault(image_id, {
            'image_path': str(row['image_path']),
            'boxes': row.get('all_gt_boxes_xyxy_original', []),
        })

    # Use the first completed model/layer directory as the canonical image set.
    anchor = root / strategies[0] / models[0] / layers[0]
    image_ids: List[int] = []
    for path in sorted(anchor.glob('image_*.png')):
        try:
            image_ids.append(int(path.stem.split('_')[1]))
        except (IndexError, ValueError):
            continue
    if not image_ids:
        raise RuntimeError(f'No image-level aggregate PNGs found in {anchor}')

    panel_total = 0
    skipped: Dict[str, int] = {strategy: 0 for strategy in strategies}
    for strategy in strategies:
        for position, image_id in enumerate(image_ids, 1):
            rows = []
            try:
                source = source_by_image[image_id]
                source_image = load_rgb(source['image_path'])
                source_with_gt = source_image
                for index, box in enumerate(source['boxes'], 1):
                    source_with_gt = draw_box(source_with_gt, box, f'GT {index}')
                for layer in layers:
                    image_paths = [
                        find_source_image(root, image_id, model, layer, strategy)
                        for model in models
                    ]
                    row = [labeled_tile(
                        source_with_gt, f'Input + all GT | {layer}',
                        args.tile_width, args.tile_height)]
                    row.extend(
                        labeled_tile(load_rgb(path), model,
                                     args.tile_width, args.tile_height)
                        for model, path in zip(models, image_paths))
                    rows.append(row)
            except (FileNotFoundError, KeyError) as exc:
                skipped[strategy] += 1
                print(f'SKIP image={image_id}: {exc}', flush=True)
                continue

            destination = (
                root / 'panels_5x4' / strategy /
                f'image_{image_id:08d}.png')
            if args.overwrite or not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                compose_grid(rows).save(
                    destination, format='PNG',
                    compress_level=args.png_compress_level)
            panel_total += 1
            print(f'[{strategy}] {position}/{len(image_ids)} image={image_id}',
                  flush=True)

    print('Fixed-GT image-level 5x4 panels complete')
    print(f'aggregate root: {root}')
    print(f'raw CAM index: {raw_index}')
    print(f'panels written or reused: {panel_total}')
    print(f'skipped: {skipped}')


if __name__ == '__main__':
    main()
