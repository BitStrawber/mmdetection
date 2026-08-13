#!/usr/bin/env python3
"""Compose bare-backbone 5x4 panels from existing activation PNGs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.exp_2.backbone_analysis.common import parse_csv, read_jsonl  # noqa: E402
from tools.exp_2.backbone_analysis.render_feature_activation import (  # noqa: E402
    labeled_panel_tile,
)
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--normalizations', required=True)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def image_path(row: dict) -> Path:
    value = row.get('image_path') or row.get('variants', {}).get(
        'clean', {}).get('image_path')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('PNG compression level must be in [0, 9]')
    render_root = Path(args.render_root).expanduser().resolve()
    rows = read_jsonl(args.manifest)
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    normalizations = parse_csv(args.normalizations)
    if not rows or not models or not layers or not normalizations:
        raise ValueError('Manifest, models, layers and normalizations must not be empty')

    completed = 0
    for normalization in normalizations:
        root = render_root / normalization
        for position, row in enumerate(rows):
            file_stem = f'{position:05d}_{int(row["image_id"])}'
            try:
                with Image.open(image_path(row)) as opened:
                    original = opened.convert('RGB')
                grid_rows = []
                for layer in layers:
                    tiles = [labeled_panel_tile(original, f'Input | {layer}')]
                    for model in models:
                        path = root / model / 'without_boxes' / layer / f'{file_stem}.png'
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        with Image.open(path) as opened:
                            tiles.append(labeled_panel_tile(opened.convert('RGB'), model))
                    grid_rows.append(tiles)
            except FileNotFoundError as exc:
                print(f'SKIP [{normalization}] {file_stem}: {exc}', flush=True)
                continue
            width = max(tile.width for grid_row in grid_rows for tile in grid_row)
            height = max(tile.height for grid_row in grid_rows for tile in grid_row)
            panel = Image.new('RGB', (width * len(grid_rows[0]), height * len(grid_rows)),
                              color=(255, 255, 255))
            for row_index, grid_row in enumerate(grid_rows):
                for col_index, tile in enumerate(grid_row):
                    panel.paste(tile, (col_index * width, row_index * height))
            destination = root / 'panels_5x4' / f'{file_stem}.png'
            if args.overwrite or not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                panel.save(destination, format='PNG', compress_level=args.png_compress_level)
            completed += 1
            print(f'[{normalization}] {position + 1}/{len(rows)} {row["file_name"]}', flush=True)
    print(f'Pretrained activation 5x4 panels complete: {completed}')


if __name__ == '__main__':
    main()
