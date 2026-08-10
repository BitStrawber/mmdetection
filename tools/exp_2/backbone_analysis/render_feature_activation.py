#!/usr/bin/env python3
"""Render shared-scale blue-yellow activation maps and compute FG/BG statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .common import ensure_empty_or_create, parse_csv, read_jsonl, write_json


PALETTE = (
    (0.00, (5, 16, 72)),
    (0.28, (0, 80, 190)),
    (0.55, (0, 190, 220)),
    (0.78, (225, 225, 30)),
    (1.00, (255, 250, 180)),
)
RESAMPLE = getattr(Image, 'Resampling', Image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument(
        '--normalization-mode',
        choices=(
            'shared-per-sample',
            'reference-per-sample-layer',
            'reference-dataset-per-layer',
        ),
        default='shared-per-sample',
        help=(
            'Color-scale normalization. reference-per-sample-layer computes '
            'one range from the reference model for each sample/layer. '
            'reference-dataset-per-layer computes one range per layer from '
            'the reference model over the complete manifest.'))
    parser.add_argument(
        '--normalization-reference-model', default='',
        help='Reference model ID required by either reference mode.')
    parser.add_argument('--box-width', type=int, default=3)
    parser.add_argument(
        '--png-compress-level', type=int, default=6,
        help='Lossless PNG compression level from 0 (fastest) to 9 (smallest)')
    parser.add_argument(
        '--skip-raw-activation', action='store_true',
        help=(
            'Do not duplicate the aggregated 2D activation arrays as .npy '
            'files. Rendering and quantitative statistics still use the '
            'activation computed from the existing spatial feature store.'))
    parser.add_argument(
        '--model-workers', type=int, default=1,
        help=(
            'Render models in separate processes after reference bounds are '
            'computed. Values greater than 1 require a reference normalization '
            'mode. Set this to the model count for one process per backbone.'))
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def spatial_path(
    root: Path, model: str, variant: str, sample_index: int,
    image_id: int, layer: str,
) -> Path:
    return (
        root / 'spatial' / model / variant /
        f'{sample_index:05d}_{image_id}' / f'{layer}.npz')


def load_activation(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing spatial feature {path}. Re-run extraction with --save-spatial.')
    with np.load(path, allow_pickle=False) as payload:
        feature = payload['feature'].astype(np.float32)
    if feature.ndim != 3 or not np.isfinite(feature).all():
        raise ValueError(f'Invalid CHW spatial feature: {path}, shape={feature.shape}')
    return np.abs(feature).mean(axis=0).astype(np.float32)


def normalize_shared(
    values: Sequence[np.ndarray], low_percentile: float, high_percentile: float,
) -> Tuple[List[np.ndarray], float, float]:
    flattened = np.concatenate([value.reshape(-1) for value in values])
    low = float(np.percentile(flattened, low_percentile))
    high = float(np.percentile(flattened, high_percentile))
    scale = max(high - low, 1e-12)
    return [
        np.clip((value - low) / scale, 0.0, 1.0).astype(np.float32)
        for value in values
    ], low, high


def normalize_with_bounds(
    value: np.ndarray, low: float, high: float,
) -> np.ndarray:
    scale = max(high - low, 1e-12)
    return np.clip((value - low) / scale, 0.0, 1.0).astype(np.float32)


def compute_reference_ranges(
    feature_root: Path, rows: Sequence[dict], reference_model: str,
    variant: str, layers: Sequence[str], low_percentile: float,
    high_percentile: float,
) -> Dict[str, Tuple[float, float]]:
    ranges = {}
    for layer in layers:
        flattened = []
        for position, row in enumerate(rows):
            path = spatial_path(
                feature_root, reference_model, variant, position,
                int(row['image_id']), layer)
            flattened.append(load_activation(path).reshape(-1))
        values = np.concatenate(flattened)
        low = float(np.percentile(values, low_percentile))
        high = float(np.percentile(values, high_percentile))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(
                f'Invalid reference range for {reference_model}/{layer}: '
                f'low={low}, high={high}')
        ranges[layer] = (low, high)
    return ranges


def colorize(value: np.ndarray) -> Image.Image:
    output = np.zeros(value.shape + (3,), dtype=np.float32)
    for index in range(len(PALETTE) - 1):
        start, start_color = PALETTE[index]
        end, end_color = PALETTE[index + 1]
        mask = (value >= start) & (value <= end if index == len(PALETTE) - 2 else value < end)
        alpha = np.clip((value - start) / max(end - start, 1e-12), 0.0, 1.0)
        first = np.asarray(start_color, dtype=np.float32)
        second = np.asarray(end_color, dtype=np.float32)
        interpolated = first + alpha[..., None] * (second - first)
        output[mask] = interpolated[mask]
    return Image.fromarray(np.rint(np.clip(output, 0, 255)).astype(np.uint8), mode='RGB')


def mask_from_boxes(
    boxes: Sequence[Sequence[float]], width: int, height: int,
    original_width: int, original_height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        left = max(0, min(width, int(math.floor(x1 * width / original_width))))
        top = max(0, min(height, int(math.floor(y1 * height / original_height))))
        right = max(0, min(width, int(math.ceil(x2 * width / original_width))))
        bottom = max(0, min(height, int(math.ceil(y2 * height / original_height))))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    width: int,
    source_size: Tuple[int, int],
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    source_width = max(float(source_size[0]), 1.0)
    source_height = max(float(source_size[1]), 1.0)
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        scaled = (
            x1 * image.width / source_width,
            y1 * image.height / source_height,
            x2 * image.width / source_width,
            y2 * image.height / source_height,
        )
        draw.rectangle(scaled, outline=(255, 40, 40), width=width)
    return result


def image_path(row: dict) -> Path:
    value = row.get('image_path')
    if not value:
        variants = row.get('variants', {})
        value = variants.get('clean', {}).get('image_path')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compute_per_sample_reference_ranges(
    feature_root: Path, rows: Sequence[dict], reference_model: str,
    variant: str, layers: Sequence[str], low_percentile: float,
    high_percentile: float,
) -> Dict[Tuple[int, str], Tuple[float, float]]:
    ranges = {}
    for position, row in enumerate(rows):
        for layer in layers:
            path = spatial_path(
                feature_root, reference_model, variant, position,
                int(row['image_id']), layer)
            reference = load_activation(path)
            low = float(np.percentile(reference, low_percentile))
            high = float(np.percentile(reference, high_percentile))
            if high <= low:
                low = float(reference.min())
                high = float(reference.max())
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(
                    f'Invalid per-sample reference range for '
                    f'{reference_model}/{position}/{layer}: '
                    f'low={low}, high={high}')
            ranges[(position, layer)] = (low, high)
    return ranges


def render_model_dataset(config: dict) -> Tuple[str, List[dict], List[dict]]:
    model = config['model']
    feature_root = Path(config['feature_root'])
    out_dir = Path(config['out_dir'])
    rows = config['rows']
    layers = config['layers']
    variant = config['variant']
    dataset_ranges = config['dataset_ranges']
    sample_ranges = config['sample_ranges']
    statistics = []
    clipping_rows = []

    for position, row in enumerate(rows):
        source_path = image_path(row)
        with Image.open(source_path) as opened:
            original_size = opened.size
        original_width, original_height = original_size
        boxes = row.get('boxes_xyxy', [])
        file_stem = f'{position:05d}_{int(row["image_id"])}'

        for layer in layers:
            path = spatial_path(
                feature_root, model, variant, position,
                int(row['image_id']), layer)
            raw = load_activation(path)
            if dataset_ranges:
                low, high = dataset_ranges[layer]
            else:
                low, high = sample_ranges[(position, layer)]
            display = normalize_with_bounds(raw, low, high)

            model_root = out_dir / model
            if not config['skip_raw_activation']:
                raw_path = model_root / 'raw' / layer / f'{file_stem}.npy'
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, raw, allow_pickle=False)

            fg_mask = mask_from_boxes(
                boxes, raw.shape[1], raw.shape[0],
                original_width, original_height)
            bg_mask = ~fg_mask
            fg_mean = float(raw[fg_mask].mean()) if fg_mask.any() else float('nan')
            bg_mean = float(raw[bg_mask].mean()) if bg_mask.any() else float('nan')
            ratio = (
                fg_mean / max(bg_mean, 1e-12)
                if np.isfinite(fg_mean) else float('nan'))
            total_energy = float(raw.sum())
            energy_in_box = (
                float(raw[fg_mask].sum()) / max(total_energy, 1e-12)
                if fg_mask.any() else float('nan'))
            statistics.append({
                'sample_index': position,
                'image_id': int(row['image_id']),
                'model': model,
                'layer': layer,
                'fg_mean': fg_mean,
                'bg_mean': bg_mean,
                'fg_bg_ratio': ratio,
                'energy_in_boxes': energy_in_box,
                'foreground_pixels': int(fg_mask.sum()),
                'background_pixels': int(bg_mask.sum()),
            })
            clipping_rows.append({
                'sample_index': position,
                'image_id': int(row['image_id']),
                'model': model,
                'layer': layer,
                'low': low,
                'high': high,
                'below_low_fraction': float(np.mean(raw < low)),
                'above_high_fraction': float(np.mean(raw > high)),
                'raw_mean': float(raw.mean()),
                'raw_median': float(np.median(raw)),
                'raw_p01': float(np.percentile(raw, 1)),
                'raw_p99': float(np.percentile(raw, 99)),
                'normalized_mean': float(display.mean()),
                'normalized_std': float(display.std()),
            })

            rendered = colorize(display).resize(original_size, RESAMPLE.BICUBIC)
            no_box_path = (
                model_root / 'without_boxes' / layer / f'{file_stem}.png')
            with_box_path = (
                model_root / 'with_gt_boxes' / layer / f'{file_stem}.png')
            no_box_path.parent.mkdir(parents=True, exist_ok=True)
            with_box_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(
                no_box_path, compress_level=config['png_compress_level'])
            draw_boxes(
                rendered, boxes, config['box_width'],
                (
                    int(row.get('width', original_width)),
                    int(row.get('height', original_height)),
                ),
            ).save(
                with_box_path,
                compress_level=config['png_compress_level'])

        print(
            f'[{model}] {position + 1}/{len(rows)} {row["file_name"]}',
            flush=True)

    return model, statistics, clipping_rows


def render_combined_panels(
    out_dir: Path, rows: Sequence[dict], models: Sequence[str],
    layers: Sequence[str], png_compress_level: int,
) -> None:
    for position, row in enumerate(rows):
        source_path = image_path(row)
        with Image.open(source_path) as opened:
            original = opened.convert('RGB')
        file_stem = f'{position:05d}_{int(row["image_id"])}'
        for layer in layers:
            tiles = [original]
            for model in models:
                rendered_path = (
                    out_dir / model / 'without_boxes' / layer /
                    f'{file_stem}.png')
                if not rendered_path.is_file():
                    raise FileNotFoundError(rendered_path)
                with Image.open(rendered_path) as opened:
                    tiles.append(opened.convert('RGB'))
            panel = Image.new(
                'RGB', (sum(tile.width for tile in tiles), original.height),
                color=(255, 255, 255))
            left = 0
            for tile in tiles:
                panel.paste(tile, (left, 0))
                left += tile.width
            panel_path = out_dir / 'panels' / layer / f'{file_stem}.png'
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(panel_path, compress_level=png_compress_level)
        print(
            f'[panels] {position + 1}/{len(rows)} {row["file_name"]}',
            flush=True)


def main() -> None:
    args = parse_args()
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError('Percentiles must satisfy 0 <= low < high <= 100')
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be between 0 and 9')
    if args.model_workers < 1:
        raise ValueError('--model-workers must be at least 1')
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    if not models or not layers:
        raise ValueError('--models and --layers must not be empty')
    rows = read_jsonl(args.manifest)
    feature_root = Path(args.feature_root).expanduser().resolve()
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    statistics = []
    normalization_rows = []
    clipping_rows = []
    reference_ranges = {}
    if args.normalization_mode.startswith('reference-'):
        reference_model = args.normalization_reference_model.strip()
        if not reference_model:
            raise ValueError(
                '--normalization-reference-model is required for '
                'reference normalization')
        if reference_model not in models:
            raise ValueError(
                f'Normalization reference model {reference_model!r} is not '
                f'in --models: {models}')
        if args.normalization_mode == 'reference-dataset-per-layer':
            reference_ranges = compute_reference_ranges(
                feature_root, rows, reference_model, args.variant, layers,
                args.low_percentile, args.high_percentile)
            write_json(out_dir / 'reference_normalization.json', {
                'normalization_mode': args.normalization_mode,
                'reference_model': reference_model,
                'variant': args.variant,
                'sample_count': len(rows),
                'percentiles': [args.low_percentile, args.high_percentile],
                'scope': 'dataset-level per-layer',
                'layers': {
                    layer: {'low': bounds[0], 'high': bounds[1]}
                    for layer, bounds in reference_ranges.items()
                },
            })

    if args.model_workers > 1:
        if not args.normalization_mode.startswith('reference-'):
            raise ValueError(
                '--model-workers greater than 1 requires a reference '
                'normalization mode')
        sample_ranges = {}
        if args.normalization_mode == 'reference-per-sample-layer':
            sample_ranges = compute_per_sample_reference_ranges(
                feature_root, rows, reference_model, args.variant, layers,
                args.low_percentile, args.high_percentile)
        for position, row in enumerate(rows):
            for layer in layers:
                if reference_ranges:
                    low, high = reference_ranges[layer]
                else:
                    low, high = sample_ranges[(position, layer)]
                normalization_rows.append({
                    'sample_index': position,
                    'image_id': int(row['image_id']),
                    'layer': layer,
                    'models': ','.join(models),
                    'normalization_mode': args.normalization_mode,
                    'reference_model': reference_model,
                    'low': low,
                    'high': high,
                })

        worker_count = min(args.model_workers, len(models))
        futures = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for model in models:
                future = executor.submit(render_model_dataset, {
                    'model': model,
                    'feature_root': str(feature_root),
                    'out_dir': str(out_dir),
                    'rows': rows,
                    'layers': layers,
                    'variant': args.variant,
                    'dataset_ranges': reference_ranges,
                    'sample_ranges': sample_ranges,
                    'skip_raw_activation': args.skip_raw_activation,
                    'png_compress_level': args.png_compress_level,
                    'box_width': args.box_width,
                })
                futures[future] = model
            for future in as_completed(futures):
                model, model_statistics, model_clipping = future.result()
                statistics.extend(model_statistics)
                clipping_rows.extend(model_clipping)
                print(f'Completed model worker: {model}', flush=True)

        statistics.sort(
            key=lambda item: (
                item['sample_index'], item['layer'], item['model']))
        clipping_rows.sort(
            key=lambda item: (
                item['sample_index'], item['layer'], item['model']))
        render_combined_panels(
            out_dir, rows, models, layers, args.png_compress_level)

        with (out_dir / 'activation_statistics.tsv').open(
                'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(statistics[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(statistics)
        with (out_dir / 'shared_normalization.tsv').open(
                'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(normalization_rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(normalization_rows)
        with (out_dir / 'normalization_clipping_statistics.tsv').open(
                'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(clipping_rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(clipping_rows)
        if args.normalization_mode == 'reference-per-sample-layer':
            write_json(out_dir / 'reference_normalization.json', {
                'normalization_mode': args.normalization_mode,
                'reference_model': reference_model,
                'variant': args.variant,
                'sample_count': len(rows),
                'percentiles': [args.low_percentile, args.high_percentile],
                'scope': 'per-sample per-layer',
                'bounds_file': 'shared_normalization.tsv',
            })
        write_json(out_dir / 'activation_metadata.json', {
            'feature_root': str(feature_root),
            'models': models,
            'layers': layers,
            'variant': args.variant,
            'aggregation': 'mean(abs(feature), channel)',
            'foreground_definition': 'union of COCO GT bounding boxes',
            'normalization': args.normalization_mode,
            'normalization_reference_model': reference_model,
            'normalization_scope': (
                'dataset-level per-layer'
                if args.normalization_mode == 'reference-dataset-per-layer'
                else 'per-sample per-layer from reference model'),
            'percentiles': [args.low_percentile, args.high_percentile],
            'quantitative_statistics_source': (
                'raw activation before normalization'),
            'raw_activation_arrays_saved': not args.skip_raw_activation,
            'model_workers': worker_count,
            'png_compress_level': args.png_compress_level,
            'warning': 'These are feature activation maps, not Grad-CAM.',
        })
        print(f'Feature activation outputs: {out_dir}')
        return

    for position, row in enumerate(rows):
        source_path = image_path(row)
        with Image.open(source_path) as opened:
            original = opened.convert('RGB')
        original_width, original_height = original.size
        boxes = row.get('boxes_xyxy', [])
        for layer in layers:
            raw_values = []
            for model in models:
                path = spatial_path(
                    feature_root, model, args.variant, position,
                    int(row['image_id']), layer)
                raw_values.append(load_activation(path))
            if args.normalization_mode == 'reference-dataset-per-layer':
                low, high = reference_ranges[layer]
                normalized = [
                    normalize_with_bounds(raw, low, high)
                    for raw in raw_values
                ]
            elif args.normalization_mode == 'reference-per-sample-layer':
                reference = raw_values[models.index(reference_model)]
                low = float(np.percentile(
                    reference, args.low_percentile))
                high = float(np.percentile(
                    reference, args.high_percentile))
                if high <= low:
                    low = float(reference.min())
                    high = float(reference.max())
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    raise ValueError(
                        f'Invalid per-sample reference range for '
                        f'{reference_model}/{position}/{layer}: '
                        f'low={low}, high={high}')
                normalized = [
                    normalize_with_bounds(raw, low, high)
                    for raw in raw_values
                ]
            else:
                low, high = normalize_shared(
                    raw_values, args.low_percentile, args.high_percentile)
            normalization_rows.append({
                'sample_index': position,
                'image_id': int(row['image_id']),
                'layer': layer,
                'models': ','.join(models),
                'normalization_mode': args.normalization_mode,
                'reference_model': args.normalization_reference_model,
                'low': low,
                'high': high,
            })
            panel_tiles = [original]
            for model, raw, display in zip(models, raw_values, normalized):
                model_root = out_dir / model
                file_stem = f'{position:05d}_{int(row["image_id"])}'
                if not args.skip_raw_activation:
                    raw_path = model_root / 'raw' / layer / f'{file_stem}.npy'
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(raw_path, raw, allow_pickle=False)
                fg_mask = mask_from_boxes(
                    boxes, raw.shape[1], raw.shape[0],
                    original_width, original_height)
                bg_mask = ~fg_mask
                fg_mean = float(raw[fg_mask].mean()) if fg_mask.any() else float('nan')
                bg_mean = float(raw[bg_mask].mean()) if bg_mask.any() else float('nan')
                ratio = fg_mean / max(bg_mean, 1e-12) if np.isfinite(fg_mean) else float('nan')
                total_energy = float(raw.sum())
                energy_in_box = (
                    float(raw[fg_mask].sum()) / max(total_energy, 1e-12)
                    if fg_mask.any() else float('nan'))
                statistics.append({
                    'sample_index': position,
                    'image_id': int(row['image_id']),
                    'model': model,
                    'layer': layer,
                    'fg_mean': fg_mean,
                    'bg_mean': bg_mean,
                    'fg_bg_ratio': ratio,
                    'energy_in_boxes': energy_in_box,
                    'foreground_pixels': int(fg_mask.sum()),
                    'background_pixels': int(bg_mask.sum()),
                })
                clipping_rows.append({
                    'sample_index': position,
                    'image_id': int(row['image_id']),
                    'model': model,
                    'layer': layer,
                    'low': low,
                    'high': high,
                    'below_low_fraction': float(np.mean(raw < low)),
                    'above_high_fraction': float(np.mean(raw > high)),
                    'raw_mean': float(raw.mean()),
                    'raw_median': float(np.median(raw)),
                    'raw_p01': float(np.percentile(raw, 1)),
                    'raw_p99': float(np.percentile(raw, 99)),
                    'normalized_mean': float(display.mean()),
                    'normalized_std': float(display.std()),
                })
                rendered = colorize(display).resize(original.size, RESAMPLE.BICUBIC)
                no_box_path = (
                    model_root / 'without_boxes' / layer / f'{file_stem}.png')
                with_box_path = (
                    model_root / 'with_gt_boxes' / layer / f'{file_stem}.png')
                no_box_path.parent.mkdir(parents=True, exist_ok=True)
                with_box_path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(
                    no_box_path, compress_level=args.png_compress_level)
                draw_boxes(
                    rendered,
                    boxes,
                    args.box_width,
                    (
                        int(row.get('width', original_width)),
                        int(row.get('height', original_height)),
                    ),
                ).save(
                    with_box_path, compress_level=args.png_compress_level)
                panel_tiles.append(rendered)
            panel = Image.new(
                'RGB', (sum(tile.width for tile in panel_tiles), original.height),
                color=(255, 255, 255))
            left = 0
            for tile in panel_tiles:
                panel.paste(tile, (left, 0))
                left += tile.width
            panel_path = out_dir / 'panels' / layer / f'{position:05d}_{int(row["image_id"])}.png'
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(
                panel_path, compress_level=args.png_compress_level)
        print(f'[{position + 1}/{len(rows)}] {row["file_name"]}', flush=True)

    with (out_dir / 'activation_statistics.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(statistics)
    with (out_dir / 'shared_normalization.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(normalization_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(normalization_rows)
    with (out_dir / 'normalization_clipping_statistics.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(clipping_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(clipping_rows)
    if args.normalization_mode == 'reference-per-sample-layer':
        write_json(out_dir / 'reference_normalization.json', {
            'normalization_mode': args.normalization_mode,
            'reference_model': reference_model,
            'variant': args.variant,
            'sample_count': len(rows),
            'percentiles': [args.low_percentile, args.high_percentile],
            'scope': 'per-sample per-layer',
            'bounds_file': 'shared_normalization.tsv',
        })
    write_json(out_dir / 'activation_metadata.json', {
        'feature_root': str(feature_root),
        'models': models,
        'layers': layers,
        'variant': args.variant,
        'aggregation': 'mean(abs(feature), channel)',
        'foreground_definition': 'union of COCO GT bounding boxes',
        'normalization': args.normalization_mode,
        'normalization_reference_model': (
            args.normalization_reference_model or None),
        'normalization_scope': (
            'dataset-level per-layer'
            if args.normalization_mode == 'reference-dataset-per-layer'
            else (
                'per-sample per-layer from reference model'
                if args.normalization_mode == 'reference-per-sample-layer'
                else 'per-sample per-layer across selected models')),
        'percentiles': [args.low_percentile, args.high_percentile],
        'quantitative_statistics_source': 'raw activation before normalization',
        'raw_activation_arrays_saved': not args.skip_raw_activation,
        'png_compress_level': args.png_compress_level,
        'warning': 'These are feature activation maps, not Grad-CAM.',
    })
    print(f'Feature activation outputs: {out_dir}')


if __name__ == '__main__':
    main()
