#!/usr/bin/env python3
"""Index and render prediction-conditioned XGradCAM with prediction/GT QA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    blue_yellow_rgb,
    cam_metrics,
    clean_name,
    compose_grid,
    draw_box,
    finite_percentiles,
    jet_rgb,
    labeled_tile,
    load_rgb,
    normalize_with_limits,
    overlay_heatmap,
    parse_csv,
    read_json,
    resize_map,
    save_rgb,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--reference-model', default='')
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--overlay-alpha', type=float, default=0.48)
    parser.add_argument(
        '--overlay-original',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--colormap',
        choices=('jet', 'blue_yellow'),
        default='jet',
    )
    parser.add_argument(
        '--draw-boxes',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--strategy',
        choices=('both', 'reference', 'independent'),
        default='both',
    )
    parser.add_argument('--panel-limit', type=int, default=0)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument(
        '--save-prediction-images',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save per-prediction heatmap/overlay/with_boxes images.')
    parser.add_argument(
        '--compute-cam-metrics',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute full-resolution metrics for every prediction.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_cam(path: Union[str, Path]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = payload['cam'].astype(np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f'Invalid raw CAM: {path}, shape={value.shape}')
    return value


def scan_records(cam_root: Path, models: Sequence[str], layers: Sequence[str]):
    records: List[Dict[str, Any]] = []
    for model in models:
        base = cam_root / 'raw_cam' / clean_name(model)
        if not base.is_dir():
            raise FileNotFoundError(f'Missing prediction CAM model directory: {base}')
        for path in sorted(base.glob('image_*/pred_*/prediction.json')):
            row = read_json(path)
            missing = [layer for layer in layers if not Path(row['layers'].get(layer, '')).is_file()]
            if missing:
                raise FileNotFoundError(f'{path}: missing CAM layers {missing}')
            records.append(row)
    if not records:
        raise RuntimeError(f'No prediction.json files found under {cam_root}')
    return records


def render_map(
    cam: np.ndarray,
    image: np.ndarray,
    prediction_box: Sequence[float],
    prediction_label: str,
    matched_gt_box: Optional[Sequence[float]],
    low: float,
    high: float,
    gamma: float,
    alpha: float,
    colormap: str,
    draw_boxes: bool,
    overlay_original: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = resize_map(cam, image.shape[1], image.shape[0])
    normalized = normalize_with_limits(resized, low, high)
    heat = (
        jet_rgb(normalized, gamma)
        if colormap == 'jet'
        else blue_yellow_rgb(normalized, gamma)
    )
    overlay = (
        overlay_heatmap(image, heat, alpha)
        if overlay_original
        else heat
    )
    if not draw_boxes:
        return heat, overlay, overlay
    with_boxes = draw_box(
        overlay, prediction_box, prediction_label, color=(255, 230, 20), width=3)
    if matched_gt_box is not None:
        with_boxes = draw_box(
            with_boxes, matched_gt_box, 'best GT', color=(40, 245, 95), width=2)
    return heat, overlay, with_boxes


def output_path(
    root: Path, strategy: str, model: str, layer: str,
    image_id: int, rank: int, kind: str,
) -> Path:
    return (
        root / strategy / kind / model / layer /
        f'image_{image_id:08d}_pred_{rank:03d}.png'
    )


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    if not models or not layers:
        raise ValueError('--models and --layers cannot be empty')
    if args.reference_model and args.reference_model not in models:
        raise ValueError('--reference-model must be included in --models')
    cam_root = Path(args.cam_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    records = scan_records(cam_root, models, layers)
    records.sort(key=lambda row: (
        int(row['image_id']), str(row['model']), int(row['prediction_rank'])))
    atomic_write_jsonl(cam_root / 'raw_cam_index.jsonl', records)
    atomic_write_json(cam_root / 'raw_cam_index_summary.json', {
        'method': 'prediction-conditioned XGradCAM',
        'models': models,
        'layers': layers,
        'predictions': len(records),
        'images': len({int(row['image_id']) for row in records}),
    })

    # Optional common scale: one dataset-wide range per layer, estimated only
    # from the declared ImageNet-reference detector's prediction CAMs.
    reference_ranges: Dict[str, Tuple[float, float]] = {}
    if args.reference_model:
        for layer in layers:
            flattened = [
                load_cam(row['layers'][layer]).reshape(-1)
                for row in records if row['model'] == args.reference_model
            ]
            if flattened:
                reference_ranges[layer] = finite_percentiles(
                    np.concatenate(flattened),
                    args.low_percentile, args.high_percentile)

    normalization_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    rendered_for_union: Dict[
        Tuple[str, int, str, str],
        Tuple[np.ndarray, List[Dict[str, Any]]],
    ] = {}
    strategies = []
    if args.strategy in {'independent', 'both'}:
        strategies.append('independent_p1_p99')
    if args.strategy in {'reference', 'both'}:
        if not reference_ranges:
            raise ValueError(
                'Reference normalization requires --reference-model '
                'with valid reference predictions')
        strategies.append('imagenet_reference_dataset_p1_p99')

    for position, row in enumerate(records, 1):
        model = str(row['model'])
        image_id = int(row['image_id'])
        rank = int(row['prediction_rank'])
        image = None
        if args.save_prediction_images or args.compute_cam_metrics:
            image = load_rgb(row['image_path'])
        height, width = [int(value) for value in row['ori_shape']]
        pred_box = row['bbox_xyxy_original']
        gt_box = row.get('matched_bbox_xyxy_original')
        pred_label = (
            f'{row["prediction_category_name"]} '
            f'{float(row["prediction_score"]):.2f}')
        all_gt_boxes = row.get('all_gt_boxes_xyxy_original', [])
        for layer in layers:
            cam = load_cam(row['layers'][layer])
            independent = finite_percentiles(
                cam, args.low_percentile, args.high_percentile)
            bounds = {'independent_p1_p99': independent}
            if layer in reference_ranges:
                bounds['imagenet_reference_dataset_p1_p99'] = reference_ranges[layer]
            resized_union = None
            for strategy, (low, high) in bounds.items():
                if strategy not in strategies:
                    continue
                if args.save_prediction_images:
                    heat, overlay, with_boxes = render_map(
                        cam, image, pred_box, pred_label, gt_box,
                        low, high, args.gamma, args.overlay_alpha,
                        args.colormap, args.draw_boxes,
                        args.overlay_original)
                    save_rgb(output_path(
                        out_dir, strategy, model, layer, image_id, rank, 'heatmap'),
                        heat, args.png_compress_level)
                    save_rgb(output_path(
                        out_dir, strategy, model, layer, image_id, rank, 'overlay'),
                        overlay, args.png_compress_level)
                    save_rgb(output_path(
                        out_dir, strategy, model, layer, image_id, rank, 'with_boxes'),
                        with_boxes, args.png_compress_level)
                normalization_rows.append({
                    'strategy': strategy,
                    'model': model,
                    'reference_model': (
                        args.reference_model
                        if strategy.startswith('imagenet_reference') else model),
                    'image_id': image_id,
                    'prediction_rank': rank,
                    'layer': layer,
                    'low_value': low,
                    'high_value': high,
                    'low_percentile': args.low_percentile,
                    'high_percentile': args.high_percentile,
                })
                if resized_union is None:
                    resized_union = resize_map(cam, width, height)
                key = (strategy, image_id, model, layer)
                union_state = rendered_for_union.get(key)
                if union_state is None:
                    rendered_for_union[key] = (
                        resized_union.copy(),
                        [row],
                    )
                else:
                    np.maximum(
                        union_state[0],
                        resized_union,
                        out=union_state[0],
                    )
                    union_state[1].append(row)

            if args.compute_cam_metrics:
                resized = (
                    resized_union
                    if resized_union is not None
                    else resize_map(cam, width, height)
                )
                pred_metrics = cam_metrics(
                    resized, pred_box, list(all_gt_boxes) + [pred_box])
                metric = {
                    'model': model,
                    'image_id': image_id,
                    'prediction_rank': rank,
                    'layer': layer,
                    'prediction_category_name': row['prediction_category_name'],
                    'prediction_score': row['prediction_score'],
                    'best_gt_iou': row['best_gt_iou'],
                    'matched': row['matched'],
                    'class_correct': row['class_correct'],
                    'tp_at_match_iou': row['tp_at_match_iou'],
                    'matched_category_name': row.get('matched_category_name'),
                    'target_logit': row['target_logit'],
                    'target_probability_recomputed': row['target_probability_recomputed'],
                    **{f'prediction_box_{key}': value for key, value in pred_metrics.items()},
                }
                if bool(row['matched']) and gt_box is not None:
                    gt_metrics = cam_metrics(
                        resized, gt_box, all_gt_boxes)
                    metric.update({
                        f'matched_gt_{key}': value for key, value in gt_metrics.items()
                    })
                metric_rows.append(metric)
        print(
            f'[{position}/{len(records)}] {model} image={image_id} pred={rank}',
            flush=True)

    write_tsv(out_dir / 'normalization_scales.tsv', normalization_rows)
    if metric_rows:
        write_tsv(
            out_dir / 'metrics' / 'prediction_layer_metrics.tsv',
            metric_rows,
        )

    # One image-level map per model/layer: pixelwise maximum across predictions.
    union_paths: Dict[Tuple[str, int, str, str], Path] = {}
    union_source_paths: Dict[int, str] = {}
    cached_union_image_id = None
    cached_union_image = None
    for key, values in rendered_for_union.items():
        strategy, image_id, model, layer = key
        raw_union, union_rows = values
        sample_row = union_rows[0]
        union_source_paths[image_id] = str(sample_row['image_path'])
        if cached_union_image_id != image_id:
            cached_union_image = load_rgb(sample_row['image_path'])
            cached_union_image_id = image_id
        image = cached_union_image
        if strategy == 'independent_p1_p99':
            low, high = finite_percentiles(
                raw_union, args.low_percentile, args.high_percentile)
        else:
            low, high = reference_ranges[layer]
        normalized_union = normalize_with_limits(raw_union, low, high)
        heat = (
            jet_rgb(normalized_union, args.gamma)
            if args.colormap == 'jet'
            else blue_yellow_rgb(normalized_union, args.gamma)
        )
        overlay = (
            overlay_heatmap(image, heat, args.overlay_alpha)
            if args.overlay_original
            else heat
        )
        if args.draw_boxes:
            for row in union_rows:
                overlay = draw_box(
                    overlay,
                    row['bbox_xyxy_original'],
                    f'{int(row["prediction_rank"])}:{row["prediction_category_name"]}',
                    color=(255, 230, 20), width=2)
        destination = (
            out_dir / strategy / 'image_union' / model / layer /
            f'image_{image_id:08d}.png')
        save_rgb(destination, overlay, args.png_compress_level)
        union_paths[key] = destination

    panel_images = sorted({key[1] for key in union_paths})
    if args.panel_limit > 0:
        panel_images = panel_images[:args.panel_limit]
    panel_count = 0
    for strategy in strategies:
        for image_id in panel_images:
            if not all((strategy, image_id, model, layer) in union_paths
                       for model in models for layer in layers):
                continue
            grid = []
            for layer in layers:
                row = [labeled_tile(
                    load_rgb(union_source_paths[image_id]), f'Input | {layer}',
                    440, 330)]
                for model in models:
                    row.append(labeled_tile(
                        load_rgb(union_paths[(strategy, image_id, model, layer)]),
                        model, 440, 330))
                grid.append(row)
            panel = compose_grid(grid)
            path = (
                out_dir / 'panels_5x4' / strategy /
                f'image_{image_id:08d}.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(path, format='PNG', compress_level=args.png_compress_level)
            panel_count += 1

    atomic_write_json(out_dir / 'render_summary.json', {
        'method': 'prediction-conditioned XGradCAM',
        'models': models,
        'layers': layers,
        'reference_model': args.reference_model or None,
        'strategies': strategies,
        'prediction_layer_rows': len(metric_rows),
        'save_prediction_images': args.save_prediction_images,
        'compute_cam_metrics': args.compute_cam_metrics,
        'colormap': args.colormap,
        'draw_boxes': args.draw_boxes,
        'overlay_original': args.overlay_original,
        'strategy': args.strategy,
        'panels': panel_count,
        'panels_5x4': panel_count,
        'panel_layout': (
            'Rows=res2,res3,res4,res5; columns=input plus the selected '
            'detector models. Prediction unions remain model-specific.'),
        'interpretation': (
            'Prediction boxes/classes are model-specific. This is behavioral/error '
            'analysis and is not a controlled substitute for fixed-GT XGradCAM.'),
    })
    atomic_write_json(out_dir / 'COMPLETE.json', {'status': 'complete'})
    print(f'Prediction XGradCAM rendered: {out_dir}')


if __name__ == '__main__':
    main()
