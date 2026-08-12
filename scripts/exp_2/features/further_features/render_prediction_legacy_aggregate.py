#!/usr/bin/env python3
"""Render legacy-style prediction CAMs: one image per model, layer and style."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    atomic_write_json,
    clean_name,
    load_rgb,
    parse_csv,
    resize_map,
    save_rgb,
    write_tsv,
)
from render_prediction_xgradcam import load_cam, scan_records  # noqa: E402


STYLES = ('legacy_jet', 'legacy_turbo_gamma05')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument(
        '--aggregation', choices=('max', 'sum', 'mean'), default='sum',
        help='Old scripts defaulted to sum; max and mean remain available.')
    parser.add_argument(
        '--styles', default=','.join(STYLES),
        help='Comma-separated subset of legacy_jet,legacy_turbo_gamma05.')
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def aggregate_maps(values: Sequence[np.ndarray], method: str) -> np.ndarray:
    if not values:
        raise ValueError('Cannot aggregate an empty prediction CAM collection')
    stack = np.stack(values, axis=0).astype(np.float32, copy=False)
    if method == 'max':
        return np.max(stack, axis=0)
    if method == 'sum':
        return np.sum(stack, axis=0, dtype=np.float32)
    return np.mean(stack, axis=0, dtype=np.float32)


def minmax(value: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def colorize(normalized: np.ndarray, style: str) -> np.ndarray:
    value = normalized
    if style == 'legacy_turbo_gamma05':
        value = np.power(value, 0.5)
        colormap = cv2.COLORMAP_TURBO
    elif style == 'legacy_jet':
        colormap = cv2.COLORMAP_JET
    else:
        raise ValueError(f'Unsupported legacy style: {style}')
    bgr = cv2.applyColorMap(
        np.uint8(np.clip(value, 0.0, 1.0) * 255.0), colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def output_path(
    root: Path, style: str, model: str, layer: str, image_id: int,
) -> Path:
    return (
        root / style / clean_name(model) / clean_name(layer) /
        f'image_{image_id:08d}.png'
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be in [0, 9]')
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    styles = parse_csv(args.styles)
    if not models or not layers or not styles:
        raise ValueError('--models, --layers and --styles cannot be empty')
    unsupported = set(styles) - set(STYLES)
    if unsupported:
        raise ValueError(f'Unsupported legacy styles: {sorted(unsupported)}')

    cam_root = Path(args.cam_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    records = scan_records(cam_root, models, layers)

    grouped: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row['model']), int(row['image_id']))].append(row)

    inventory: List[Dict[str, Any]] = []
    generated = 0
    for position, ((model, image_id), rows) in enumerate(
            sorted(grouped.items()), 1):
        image = load_rgb(rows[0]['image_path'])
        height, width = image.shape[:2]
        for layer in layers:
            raw_maps = [
                resize_map(load_cam(row['layers'][layer]), width, height)
                for row in rows
            ]
            aggregate = aggregate_maps(raw_maps, args.aggregation)
            normalized = minmax(aggregate)
            for style in styles:
                destination = output_path(
                    out_dir, style, model, layer, image_id)
                if not destination.is_file() or args.overwrite:
                    save_rgb(
                        destination,
                        colorize(normalized, style),
                        args.png_compress_level,
                    )
                inventory.append({
                    'style': style,
                    'aggregation': args.aggregation,
                    'normalization': 'per-image-per-model-per-layer min-max',
                    'model': model,
                    'image_id': image_id,
                    'layer': layer,
                    'predictions_aggregated': len(rows),
                    'path': str(destination),
                })
                generated += 1
        print(
            f'[{position}/{len(grouped)}] {model} image={image_id} '
            f'predictions={len(rows)}', flush=True)

    if not inventory:
        raise RuntimeError('No legacy prediction CAM visualizations were rendered')
    write_tsv(out_dir / 'image_inventory.tsv', inventory)
    atomic_write_json(out_dir / 'render_summary.json', {
        'method': 'legacy-style image-level prediction XGradCAM aggregation',
        'cam_root': str(cam_root),
        'models': models,
        'layers': layers,
        'styles': styles,
        'aggregation': args.aggregation,
        'normalization': 'per-image-per-model-per-layer min-max after aggregation',
        'views': 'pure heatmap only',
        'model_image_groups': len(grouped),
        'generated_png': generated,
        'output_contract': 'One PNG per style, image, model and layer.',
        'comparability_warning': (
            'Independent min-max improves visual contrast but color magnitude '
            'must not be compared quantitatively across models.'),
    })
    atomic_write_json(out_dir / 'COMPLETE.json', {
        'status': 'complete',
        'generated_png': generated,
        'render_summary': str(out_dir / 'render_summary.json'),
    })
    print(f'Legacy prediction aggregation outputs: {out_dir}')
    print(f'Generated PNG: {generated}')


if __name__ == '__main__':
    main()
