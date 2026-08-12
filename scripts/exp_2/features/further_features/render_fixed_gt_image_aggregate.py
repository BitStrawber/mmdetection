#!/usr/bin/env python3
"""Render one aggregated fixed-GT CAM per image, model, layer and strategy."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    atomic_write_json,
    blue_yellow_rgb,
    clean_name,
    draw_box,
    finite_percentiles,
    load_rgb,
    normalize_with_limits,
    overlay_heatmap,
    parse_csv,
    read_jsonl,
    resize_map,
    save_rgb,
    write_tsv,
)
from render_fixed_gt_xgradcam import load_cam, percentile_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-model', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument(
        '--aggregation', choices=('max', 'sum', 'mean'), default='max')
    parser.add_argument(
        '--view', choices=('pure', 'overlay', 'with_gt'), default='pure',
        help='The one visualization saved for each output combination.')
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--overlay-alpha', type=float, default=0.48)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def aggregate_maps(values: Sequence[np.ndarray], method: str) -> np.ndarray:
    if not values:
        raise ValueError('Cannot aggregate an empty CAM collection')
    stack = np.stack(values, axis=0).astype(np.float32, copy=False)
    if method == 'max':
        return np.max(stack, axis=0)
    if method == 'sum':
        return np.sum(stack, axis=0, dtype=np.float32)
    return np.mean(stack, axis=0, dtype=np.float32)


def render_view(
    raw: np.ndarray,
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    low: float,
    high: float,
    gamma: float,
    overlay_alpha: float,
    view: str,
) -> np.ndarray:
    heat = blue_yellow_rgb(normalize_with_limits(raw, low, high), gamma)
    if view == 'pure':
        return heat
    rendered = overlay_heatmap(image, heat, overlay_alpha)
    if view == 'overlay':
        return rendered
    for index, box in enumerate(boxes, 1):
        rendered = draw_box(rendered, box, f'GT {index}')
    return rendered


def output_path(
    root: Path, strategy: str, model: str, layer: str, image_id: int,
) -> Path:
    return (
        root / strategy / clean_name(model) / clean_name(layer) /
        f'image_{image_id:08d}.png'
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError('Expected 0 <= low < high <= 100')
    if args.gamma <= 0:
        raise ValueError('--gamma must be positive')
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError('--overlay-alpha must be in [0, 1]')
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be in [0, 9]')

    cam_root = Path(args.cam_root).expanduser().resolve()
    records = read_jsonl(cam_root / 'raw_cam_index.jsonl')
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv(args.models)
    if not models:
        models = list(dict.fromkeys(str(row['model']) for row in records))
    layers = parse_csv(args.layers)
    available_models = {str(row['model']) for row in records}
    missing_models = set(models) - available_models
    if missing_models:
        raise ValueError(f'Models absent from raw CAM index: {sorted(missing_models)}')
    if args.reference_model not in models:
        raise ValueError('--reference-model must be included in --models')

    by_image: Dict[int, Dict[str, Dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict))
    for row in records:
        model = str(row['model'])
        if model in models:
            by_image[int(row['image_id'])][model][int(row['annotation_id'])] = row

    suffix = percentile_tag(args.low_percentile, args.high_percentile)
    independent_strategy = f'independent_{suffix}'
    reference_strategy = f'imagenet_reference_{suffix}'
    strategies = (independent_strategy, reference_strategy)
    normalization_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    generated = 0
    skipped_incomplete = 0

    for position, image_id in enumerate(sorted(by_image), 1):
        model_rows = by_image[image_id]
        if not all(model in model_rows for model in models):
            skipped_incomplete += 1
            continue
        annotation_ids = set(model_rows[models[0]])
        for model in models[1:]:
            annotation_ids &= set(model_rows[model])
        annotation_ids = sorted(annotation_ids)
        if not annotation_ids:
            skipped_incomplete += 1
            continue

        reference_row = model_rows[args.reference_model][annotation_ids[0]]
        image = load_rgb(reference_row['image_path'])
        height, width = image.shape[:2]
        boxes = reference_row.get('all_gt_boxes_xyxy_original', [])
        for layer in layers:
            aggregated: Dict[str, np.ndarray] = {}
            for model in models:
                maps = []
                for annotation_id in annotation_ids:
                    row = model_rows[model][annotation_id]
                    cam_path = row.get('layers', {}).get(layer)
                    if not cam_path:
                        raise KeyError(
                            f'{model}/image {image_id}/ann {annotation_id} '
                            f'has no layer {layer}')
                    maps.append(resize_map(load_cam(cam_path), width, height))
                aggregated[model] = aggregate_maps(maps, args.aggregation)

            reference_low, reference_high = finite_percentiles(
                aggregated[args.reference_model],
                args.low_percentile,
                args.high_percentile,
            )
            for model in models:
                independent_low, independent_high = finite_percentiles(
                    aggregated[model],
                    args.low_percentile,
                    args.high_percentile,
                )
                bounds = {
                    independent_strategy: (
                        independent_low, independent_high, model),
                    reference_strategy: (
                        reference_low, reference_high, args.reference_model),
                }
                for strategy, (low, high, reference_model) in bounds.items():
                    destination = output_path(
                        out_dir, strategy, model, layer, image_id)
                    if not destination.is_file() or args.overwrite:
                        save_rgb(
                            destination,
                            render_view(
                                aggregated[model], image, boxes, low, high,
                                args.gamma, args.overlay_alpha, args.view),
                            args.png_compress_level,
                        )
                    normalization_rows.append({
                        'strategy': strategy,
                        'aggregation': args.aggregation,
                        'view': args.view,
                        'model': model,
                        'reference_model': reference_model,
                        'image_id': image_id,
                        'layer': layer,
                        'instances_aggregated': len(annotation_ids),
                        'low_percentile': args.low_percentile,
                        'high_percentile': args.high_percentile,
                        'low_value': low,
                        'high_value': high,
                    })
                    generated += 1

        image_rows.append({
            'image_id': image_id,
            'image_path': reference_row['image_path'],
            'instances_aggregated': len(annotation_ids),
        })
        print(
            f'[{position}/{len(by_image)}] image={image_id} '
            f'instances={len(annotation_ids)}', flush=True)

    if not normalization_rows:
        raise RuntimeError('No complete image-level CAM groups were rendered')
    write_tsv(out_dir / 'normalization_scales.tsv', normalization_rows)
    write_tsv(out_dir / 'images.tsv', image_rows)
    atomic_write_json(out_dir / 'render_summary.json', {
        'method': 'image-level aggregation of fixed-GT XGradCAM',
        'cam_root': str(cam_root),
        'models': models,
        'layers': layers,
        'strategies': list(strategies),
        'reference_model': args.reference_model,
        'aggregation': args.aggregation,
        'view': args.view,
        'images': len(image_rows),
        'skipped_incomplete_images': skipped_incomplete,
        'generated_png': generated,
        'output_contract': (
            'Exactly one PNG per image, model, layer and strategy.'),
    })
    atomic_write_json(out_dir / 'COMPLETE.json', {
        'status': 'complete',
        'generated_png': generated,
        'render_summary': str(out_dir / 'render_summary.json'),
    })
    print(f'Fixed-GT image aggregation outputs: {out_dir}')
    print(f'Generated PNG: {generated}')


if __name__ == '__main__':
    main()
