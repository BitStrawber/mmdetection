#!/usr/bin/env python3
"""Render independent and ImageNet-reference fixed-GT XGradCAM products."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    atomic_write_json,
    blue_yellow_rgb,
    cam_metrics,
    clean_name,
    compose_grid,
    draw_box,
    finite_percentiles,
    labeled_tile,
    load_rgb,
    normalize_with_limits,
    overlay_heatmap,
    parse_csv,
    read_jsonl,
    resize_map,
    save_rgb,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-model', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--overlay-alpha', type=float, default=0.48)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--tile-width', type=int, default=480)
    parser.add_argument('--tile-height', type=int, default=360)
    parser.add_argument('--panel-limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def percentile_tag(low: float, high: float) -> str:
    def one(value: float) -> str:
        return ('{:g}'.format(value)).replace('.', 'p')
    return f'p{one(low)}_p{one(high)}'


def load_cam(path: Union[str, Path]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive['cam'], dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f'CAM must be HxW, got {value.shape}: {path}')
    return value


def size_bucket(area: float) -> str:
    if area < 32.0 ** 2:
        return 'small'
    if area < 96.0 ** 2:
        return 'medium'
    return 'large'


def output_paths(
    out_dir: Path,
    strategy: str,
    model: str,
    image_id: int,
    annotation_id: int,
    layer: str,
) -> Dict[str, Path]:
    stem = (
        out_dir / strategy / clean_name(model) / clean_name(layer) /
        f'image_{image_id:08d}_ann_{annotation_id:08d}')
    return {
        'pure': stem.with_name(stem.name + '_pure.png'),
        'overlay': stem.with_name(stem.name + '_overlay.png'),
        'with_gt': stem.with_name(stem.name + '_with_gt.png'),
    }


def render_one(
    cam: np.ndarray,
    image: np.ndarray,
    bbox: Sequence[float],
    label: str,
    low: float,
    high: float,
    gamma: float,
    overlay_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_with_limits(cam, low, high)
    normalized = resize_map(normalized, image.shape[1], image.shape[0])
    heat = blue_yellow_rgb(normalized, gamma)
    overlay = overlay_heatmap(image, heat, overlay_alpha)
    with_gt = draw_box(overlay, bbox, label)
    return normalized, heat, overlay, with_gt


def aggregate_rows(
    metrics: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    numeric_fields = [
        'energy_in_target_box', 'energy_in_any_gt_box',
        'target_to_background_ratio', 'pointing_game_hit',
        'top20_iou_with_target', 'top20_area_fraction',
        'normalized_entropy', 'peak_distance_over_box_diagonal',
        'target_probability', 'target_logit',
    ]
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        rows = grouped[key]
        record = {field: value for field, value in zip(group_fields, key)}
        record['instances'] = len(rows)
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            record[f'{field}_mean'] = float(np.mean(values))
            record[f'{field}_median'] = float(np.median(values))
            record[f'{field}_std'] = float(np.std(values))
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError('Expected 0 <= low < high <= 100')
    if args.gamma <= 0:
        raise ValueError('--gamma must be positive')
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError('--overlay-alpha must be in [0, 1]')
    cam_root = Path(args.cam_root).expanduser().resolve()
    index_path = cam_root / 'raw_cam_index.jsonl'
    records = read_jsonl(index_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv(args.models)
    if not models:
        models = list(dict.fromkeys(str(row['model']) for row in records))
    layers = parse_csv(args.layers)
    selected = [row for row in records if str(row['model']) in models]
    available_models = {str(row['model']) for row in selected}
    missing_models = set(models) - available_models
    if missing_models:
        raise ValueError(f'Models absent from raw CAM index: {sorted(missing_models)}')
    if args.reference_model not in available_models:
        raise ValueError(f'Reference model not found: {args.reference_model}')

    by_key: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_key[(int(row['image_id']), int(row['annotation_id']))][str(row['model'])] = row
    complete_keys = [key for key, value in by_key.items() if set(models) <= set(value)]
    incomplete = len(by_key) - len(complete_keys)
    if not complete_keys:
        raise RuntimeError('No GT instances are complete across all selected models')

    normalization_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    panel_inputs: Dict[Tuple[str, int, int, str, str], Path] = {}
    image_cache: Dict[str, np.ndarray] = {}
    rendered_instances = 0
    percentile_suffix = percentile_tag(
        args.low_percentile, args.high_percentile)
    independent_strategy = f'independent_{percentile_suffix}'
    reference_strategy = f'imagenet_reference_{percentile_suffix}'
    for instance_position, key in enumerate(sorted(complete_keys), 1):
        image_id, annotation_id = key
        model_rows = by_key[key]
        reference = model_rows[args.reference_model]
        image_path = str(reference['image_path'])
        if image_path not in image_cache:
            image_cache[image_path] = load_rgb(image_path)
        image = image_cache[image_path]
        bbox = [float(value) for value in reference['bbox_xyxy_original']]
        all_boxes = reference['all_gt_boxes_xyxy_original']
        category_name = str(reference['category_name'])

        for layer in layers:
            reference_cam_path = reference.get('layers', {}).get(layer)
            if not reference_cam_path:
                raise KeyError(f'Reference instance {key} has no layer {layer}')
            reference_cam = load_cam(reference_cam_path)
            reference_low, reference_high = finite_percentiles(
                reference_cam, args.low_percentile, args.high_percentile)

            for model in models:
                row = model_rows[model]
                cam_path = row.get('layers', {}).get(layer)
                if not cam_path:
                    raise KeyError(f'{model} instance {key} has no layer {layer}')
                cam = load_cam(cam_path)
                independent_low, independent_high = finite_percentiles(
                    cam, args.low_percentile, args.high_percentile)
                strategies = {
                    independent_strategy: (independent_low, independent_high),
                    reference_strategy: (
                        reference_low, reference_high),
                }
                label = f'{category_name} | ann {annotation_id}'
                for strategy, (low, high) in strategies.items():
                    _, heat, overlay, with_gt = render_one(
                        cam, image, bbox, label, low, high,
                        args.gamma, args.overlay_alpha)
                    paths = output_paths(
                        out_dir, strategy, model, image_id, annotation_id, layer)
                    save_rgb(paths['pure'], heat, args.png_compress_level)
                    save_rgb(paths['overlay'], overlay, args.png_compress_level)
                    save_rgb(paths['with_gt'], with_gt, args.png_compress_level)
                    panel_inputs[(strategy, image_id, annotation_id, layer, model)] = (
                        paths['with_gt'])
                    normalization_rows.append({
                        'strategy': strategy,
                        'model': model,
                        'reference_model': (
                            args.reference_model
                            if strategy == reference_strategy else model),
                        'image_id': image_id,
                        'annotation_id': annotation_id,
                        'category_name': category_name,
                        'layer': layer,
                        'low_percentile': args.low_percentile,
                        'high_percentile': args.high_percentile,
                        'low_value': low,
                        'high_value': high,
                        'gamma_for_display_only': args.gamma,
                    })

                cam_original = resize_map(cam, image.shape[1], image.shape[0])
                measurements = cam_metrics(cam_original, bbox, all_boxes)
                metric_rows.append({
                    'model': model,
                    'image_id': image_id,
                    'annotation_id': annotation_id,
                    'category_id': int(row['category_id']),
                    'category_name': category_name,
                    'size_bucket_coco_pixels': size_bucket(float(row['area'])),
                    'area': float(row['area']),
                    'layer': layer,
                    'cascade_stage_zero_based': int(row['cascade_stage_zero_based']),
                    'target_logit': float(row['target_logit']),
                    'target_probability': float(row['target_probability']),
                    **measurements,
                })
        rendered_instances += 1
        print(
            f'[{instance_position}/{len(complete_keys)}] '
            f'image={image_id} ann={annotation_id}', flush=True)

    write_tsv(out_dir / 'normalization_scales.tsv', normalization_rows)
    write_tsv(out_dir / 'metrics' / 'instance_layer_metrics.tsv', metric_rows)
    write_tsv(
        out_dir / 'metrics' / 'model_layer_summary.tsv',
        aggregate_rows(metric_rows, ('model', 'layer')))
    write_tsv(
        out_dir / 'metrics' / 'model_layer_category_summary.tsv',
        aggregate_rows(metric_rows, ('model', 'layer', 'category_name')))
    write_tsv(
        out_dir / 'metrics' / 'model_layer_size_summary.tsv',
        aggregate_rows(metric_rows, ('model', 'layer', 'size_bucket_coco_pixels')))

    panel_count = 0
    panel_5x4_count = 0
    panel_keys = sorted(complete_keys)
    if args.panel_limit > 0:
        panel_keys = panel_keys[:args.panel_limit]
    for strategy in (independent_strategy, reference_strategy):
        for image_id, annotation_id in panel_keys:
            reference = by_key[(image_id, annotation_id)][args.reference_model]
            original = load_rgb(reference['image_path'])
            original_with_gt = draw_box(
                original,
                reference['bbox_xyxy_original'],
                f'{reference["category_name"]} | ann {annotation_id}')
            grid_rows = []
            for layer in layers:
                layer_row = [labeled_tile(
                    original_with_gt, f'Input + fixed GT | {layer}',
                    args.tile_width, args.tile_height)]
                for model in models:
                    rendered = load_rgb(panel_inputs[
                        (strategy, image_id, annotation_id, layer, model)])
                    layer_row.append(labeled_tile(
                        rendered, model,
                        args.tile_width, args.tile_height))
                grid_rows.append(layer_row)
            panel = compose_grid(grid_rows)
            panel_path = (
                out_dir / 'panels_5x4' / strategy /
                f'image_{image_id:08d}_ann_{annotation_id:08d}.png')
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(panel_path, format='PNG', compress_level=args.png_compress_level)
            panel_count += 1
            panel_5x4_count += 1

    atomic_write_json(out_dir / 'render_summary.json', {
        'cam_root': str(cam_root),
        'reference_model': args.reference_model,
        'models': models,
        'layers': layers,
        'complete_instances': len(complete_keys),
        'incomplete_instances_excluded': incomplete,
        'rendered_instances': rendered_instances,
        'normalization_strategies': {
            independent_strategy: (
                'Each model, GT instance and layer uses its own percentiles.'),
            reference_strategy: (
                'All models for the same GT instance and layer use the '
                'reference detector raw-CAM percentiles.'),
        },
        'low_percentile': args.low_percentile,
        'high_percentile': args.high_percentile,
        'gamma_for_display_only': args.gamma,
        'panels': panel_count,
        'panels_5x4': panel_5x4_count,
        'panel_layout': (
            'Rows=res2,res3,res4,res5; columns=input-with-fixed-GT plus '
            'the selected detector models.'),
        'metrics_source': 'unnormalized nonnegative raw XGradCAM',
        'primary_metric_note': (
            'Prefer scale-invariant spatial metrics such as energy fraction, '
            'FG/BG ratio, pointing game, top-response IoU and entropy.'),
    })
    atomic_write_json(out_dir / 'COMPLETE.json', {
        'status': 'complete',
        'raw_cam_index': str(index_path),
        'render_summary': str(out_dir / 'render_summary.json'),
    })
    print(f'Fixed-GT XGradCAM render and metrics complete: {out_dir}')


if __name__ == '__main__':
    main()
