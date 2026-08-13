#!/usr/bin/env python3
"""Compose prediction-CAM 5x4 panels from existing legacy aggregate PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import compose_grid, labeled_tile, load_rgb, parse_csv, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate-root', required=True)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--styles', default='legacy_jet,legacy_turbo_gamma05')
    parser.add_argument('--tile-width', type=int, default=480)
    parser.add_argument('--tile-height', type=int, default=360)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError('Tile dimensions must be positive')
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('PNG compression level must be in [0, 9]')
    root = Path(args.aggregate_root).expanduser().resolve()
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    styles = parse_csv(args.styles)
    records = read_jsonl(Path(args.cam_root).expanduser().resolve() /
                         'raw_cam_index.jsonl')
    source_by_image = {}
    for row in records:
        image_id = int(row['image_id'])
        source_by_image.setdefault(image_id, str(row['image_path']))

    anchor = root / styles[0] / models[0] / layers[0]
    image_ids = []
    for path in sorted(anchor.glob('image_*.png')):
        try:
            image_ids.append(int(path.stem.split('_')[1]))
        except (IndexError, ValueError):
            continue
    if not image_ids:
        raise RuntimeError(f'No aggregate prediction PNGs found in {anchor}')

    completed = 0
    skipped = {style: 0 for style in styles}
    for style in styles:
        for position, image_id in enumerate(image_ids, 1):
            try:
                source = load_rgb(source_by_image[image_id])
                rows = []
                for layer in layers:
                    paths = [
                        root / style / model / layer /
                        f'image_{image_id:08d}.png'
                        for model in models
                    ]
                    missing = [path for path in paths if not path.is_file()]
                    if missing:
                        raise FileNotFoundError(missing[0])
                    row = [labeled_tile(
                        source, f'Input | {layer}',
                        args.tile_width, args.tile_height)]
                    row.extend(
                        labeled_tile(load_rgb(path), model,
                                     args.tile_width, args.tile_height)
                        for model, path in zip(models, paths))
                    rows.append(row)
            except (FileNotFoundError, KeyError) as exc:
                skipped[style] += 1
                print(f'SKIP [{style}] image={image_id}: {exc}', flush=True)
                continue
            destination = root / 'panels_5x4' / style / f'image_{image_id:08d}.png'
            if args.overwrite or not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                compose_grid(rows).save(
                    destination, format='PNG',
                    compress_level=args.png_compress_level)
            completed += 1
            print(f'[{style}] {position}/{len(image_ids)} image={image_id}', flush=True)
    print(f'Prediction aggregate 5x4 panels complete: {completed}')
    print(f'Skipped: {skipped}')


if __name__ == '__main__':
    main()
