#!/usr/bin/env python
"""Render fixed-size, paper-ready feature maps from saved feature tensors."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


RESAMPLE = getattr(Image, 'Resampling', Image)
PALETTE_STOPS = (
    (0.00, (5, 16, 72)),
    (0.28, (0, 80, 190)),
    (0.55, (0, 190, 220)),
    (0.78, (225, 225, 30)),
    (1.00, (255, 250, 180)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render blue-yellow ImageNet/RUOD feature maps.')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--cascade-config', required=True)
    parser.add_argument('--cascade-checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-threshold', type=float, default=0.30)
    parser.add_argument('--max-boxes', type=int, default=20)
    parser.add_argument('--tile-width', type=int, default=640)
    parser.add_argument('--tile-height', type=int, default=480)
    parser.add_argument('--label-height', type=int, default=42)
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument('--expected-samples', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_manifest(path: Path) -> List[dict]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def find_input(sample_dir: Path) -> Path:
    candidates = sorted(path for path in sample_dir.glob('input.*') if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(
            'Expected one input image in {}, found {}'.format(
                sample_dir, len(candidates)))
    return candidates[0]


def load_features(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location='cpu')
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError('Invalid feature tensor payload: {}'.format(path))
    return payload


def aggregate_feature(
    feature: torch.Tensor,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    value = feature.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError('Expected CHW feature, got {}'.format(tuple(value.shape)))
    activation = value.abs().mean(dim=0).numpy()
    finite = np.isfinite(activation)
    if not finite.any():
        return np.zeros(activation.shape, dtype=np.float32)
    valid = activation[finite]
    low = float(np.percentile(valid, low_percentile))
    high = float(np.percentile(valid, high_percentile))
    if high <= low:
        low = float(valid.min())
        high = float(valid.max())
    normalized = (activation - low) / max(high - low, 1e-12)
    return np.nan_to_num(np.clip(normalized, 0.0, 1.0)).astype(np.float32)


def colorize_blue_yellow(normalized: np.ndarray) -> Image.Image:
    output = np.zeros(normalized.shape + (3,), dtype=np.float32)
    for stop_index in range(len(PALETTE_STOPS) - 1):
        start_value, start_color = PALETTE_STOPS[stop_index]
        end_value, end_color = PALETTE_STOPS[stop_index + 1]
        mask = (normalized >= start_value) & (normalized <= end_value)
        alpha = np.clip(
            (normalized - start_value) / max(end_value - start_value, 1e-12),
            0.0,
            1.0,
        )
        start = np.asarray(start_color, dtype=np.float32)
        end = np.asarray(end_color, dtype=np.float32)
        mixed = start + alpha[..., None] * (end - start)
        output[mask] = mixed[mask]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode='RGB')


def fit_fixed(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail(size, RESAMPLE.LANCZOS)
    canvas = Image.new('RGB', size, (245, 245, 245))
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    color: Tuple[int, int, int],
) -> Image.Image:
    result = image.copy().convert('RGB')
    draw = ImageDraw.Draw(result)
    line_width = max(2, int(round(min(result.size) / 180.0)))
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = max(0.0, min(x1, result.width - 1.0))
        y1 = max(0.0, min(y1, result.height - 1.0))
        x2 = max(0.0, min(x2, result.width - 1.0))
        y2 = max(0.0, min(y2, result.height - 1.0))
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    return result


def labeled_tile(image: Image.Image, label: str, size: Tuple[int, int], label_height: int) -> Image.Image:
    tile = fit_fixed(image, size)
    output = Image.new('RGB', (size[0], size[1] + label_height), 'white')
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, size[0], label_height), fill=(238, 238, 238))
    draw.text((12, 13), label, fill=(15, 15, 15))
    output.paste(tile, (0, label_height))
    return output


def join_tiles(tiles: Iterable[Image.Image]) -> Image.Image:
    values = list(tiles)
    panel = Image.new(
        'RGB',
        (sum(tile.width for tile in values), max(tile.height for tile in values)),
        'white',
    )
    offset = 0
    for tile in values:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    return panel


def predict_boxes(model, image_path: Path, threshold: float, max_boxes: int):
    from mmdet.apis import inference_detector

    prediction = inference_detector(model, str(image_path))
    instances = prediction.pred_instances.to('cpu')
    if len(instances) == 0:
        return [], [], []
    scores = instances.scores.numpy()
    order = np.argsort(-scores)
    order = [int(index) for index in order if scores[index] >= threshold][:max_boxes]
    if not order:
        return [], [], []
    boxes = instances.bboxes.numpy()[order].tolist()
    labels = instances.labels.numpy()[order].astype(int).tolist()
    selected_scores = scores[order].tolist()
    return boxes, labels, selected_scores


def write_subset_readme(
    subset_root: Path,
    subset: str,
    samples: int,
    with_boxes: bool,
    args: argparse.Namespace,
) -> None:
    lines = [
        'subset: {}'.format(subset),
        'samples: {}'.format(samples),
        'feature aggregation: mean(abs(feature), channel)',
        'normalization percentiles: {} to {}'.format(
            args.low_percentile, args.high_percentile),
        'palette: deep blue -> blue -> cyan -> yellow',
        'fixed image size: {}x{}'.format(args.tile_width, args.tile_height),
        'five-panel order: Original | Stage 1 | Stage 2 | Stage 3 | Stage 4',
        'prediction boxes: {}'.format('enabled' if with_boxes else 'disabled'),
    ]
    if with_boxes:
        lines.extend([
            'box source: J2 Cascade R-CNN predictions',
            'score threshold: {}'.format(args.score_threshold),
            'maximum boxes per image: {}'.format(args.max_boxes),
        ])
    (subset_root / 'README.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def prepare_variant(root: Path) -> Dict[str, Path]:
    paths = {
        'originals': root / 'originals',
        'feature_maps': root / 'feature_maps',
        'panels': root / 'five_panels',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_dir).resolve()
    manifest_path = source_root / 'manifest.tsv'
    if not manifest_path.is_file():
        raise FileNotFoundError('Missing source manifest: {}'.format(manifest_path))
    if args.tile_width <= 0 or args.tile_height <= 0 or args.label_height <= 0:
        raise ValueError('Tile and label dimensions must be positive')
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError('Invalid normalization percentiles')
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError('Output directory already exists: {}'.format(out_root))
        if out_root == Path(out_root.anchor) or source_root == out_root:
            raise RuntimeError('Refusing to reset unsafe output path: {}'.format(out_root))
        shutil.rmtree(str(out_root))
    out_root.mkdir(parents=True)

    records = read_manifest(manifest_path)
    grouped = {
        subset: sorted(
            (record for record in records if record['subset'] == subset),
            key=lambda row: int(row['index']),
        )
        for subset in ('imagenet', 'ruod')
    }
    for subset, rows in grouped.items():
        if len(rows) != args.expected_samples:
            raise RuntimeError(
                '{}: expected {} samples, found {}'.format(
                    subset, args.expected_samples, len(rows)))

    from mmdet.apis import init_detector
    detector = init_detector(
        args.cascade_config, args.cascade_checkpoint, device=args.device)
    detector.eval()

    variant_specs = {
        'imagenet': [('imagenet_resnet50_blue_yellow_no_box', False)],
        'ruod': [
            ('ruod_cascade_resnet50_blue_yellow_no_box', False),
            ('ruod_cascade_resnet50_blue_yellow_with_box', True),
        ],
    }
    output_rows: Dict[str, List[dict]] = {
        name: []
        for specs in variant_specs.values()
        for name, _ in specs
    }
    variant_paths = {
        name: prepare_variant(out_root / name)
        for specs in variant_specs.values()
        for name, _ in specs
    }
    size = (args.tile_width, args.tile_height)

    for subset, rows in grouped.items():
        for row in rows:
            index = int(row['index'])
            source_path = Path(row['source_path'])
            sample_dir = resolve_path(row['output_dir'], repo_root)
            input_path = find_input(sample_dir)
            features_path = sample_dir / 'features.pt'
            if not features_path.is_file():
                raise FileNotFoundError('Missing features: {}'.format(features_path))
            features = load_features(features_path)
            if len(features) != 4:
                raise RuntimeError(
                    '{}: expected four feature stages, found {}'.format(
                        features_path, len(features)))
            with Image.open(str(input_path)) as opened:
                original = ImageOps.exif_transpose(opened).convert('RGB')

            boxes: List[List[float]] = []
            labels: List[int] = []
            scores: List[float] = []
            if subset == 'ruod':
                inference_path = source_path if source_path.is_file() else input_path
                boxes, labels, scores = predict_boxes(
                    detector,
                    inference_path,
                    args.score_threshold,
                    args.max_boxes,
                )

            stage_images = []
            stage_names = []
            for stage_index, (stage_name, feature) in enumerate(features.items(), start=1):
                normalized = aggregate_feature(
                    feature, args.low_percentile, args.high_percentile)
                heatmap = colorize_blue_yellow(normalized)
                heatmap = heatmap.resize(original.size, RESAMPLE.BICUBIC)
                stage_images.append(heatmap)
                stage_names.append(stage_name)

            sample_name = '{:02d}_{}'.format(index, input_path.stem)
            for variant_name, with_boxes in variant_specs[subset]:
                paths = variant_paths[variant_name]
                variant_original = draw_boxes(original, boxes, (255, 230, 0)) \
                    if with_boxes else original
                fixed_original = fit_fixed(variant_original, size)
                original_output = paths['originals'] / '{}_original.png'.format(sample_name)
                fixed_original.save(str(original_output))

                rendered_stages = []
                feature_outputs = []
                sample_feature_dir = paths['feature_maps'] / sample_name
                sample_feature_dir.mkdir(parents=True, exist_ok=True)
                for stage_index, (stage_name, heatmap) in enumerate(
                        zip(stage_names, stage_images), start=1):
                    rendered = draw_boxes(heatmap, boxes, (255, 40, 40)) \
                        if with_boxes else heatmap
                    rendered_stages.append(rendered)
                    output = sample_feature_dir / 'stage{}_{}.png'.format(
                        stage_index, stage_name)
                    fit_fixed(rendered, size).save(str(output))
                    feature_outputs.append(str(output.relative_to(out_root / variant_name)))

                tiles = [labeled_tile(variant_original, 'Original', size, args.label_height)]
                tiles.extend(
                    labeled_tile(image, 'Stage {}'.format(stage_index), size, args.label_height)
                    for stage_index, image in enumerate(rendered_stages, start=1)
                )
                panel = join_tiles(tiles)
                panel_output = paths['panels'] / '{}_five_panel.png'.format(sample_name)
                panel.save(str(panel_output))

                output_rows[variant_name].append({
                    'subset': subset,
                    'index': index,
                    'source_path': str(source_path),
                    'input_path': str(input_path),
                    'features_path': str(features_path),
                    'with_boxes': int(with_boxes),
                    'box_count': len(boxes) if with_boxes else 0,
                    'box_scores': json.dumps(scores if with_boxes else []),
                    'box_labels': json.dumps(labels if with_boxes else []),
                    'original_file': str(original_output.relative_to(out_root / variant_name)),
                    'feature_map_files': ';'.join(feature_outputs),
                    'five_panel_file': str(panel_output.relative_to(out_root / variant_name)),
                    'layers': ','.join(stage_names),
                })

    fields = [
        'subset', 'index', 'source_path', 'input_path', 'features_path',
        'with_boxes', 'box_count', 'box_scores', 'box_labels',
        'original_file', 'feature_map_files', 'five_panel_file', 'layers',
    ]
    for subset, specs in variant_specs.items():
        for variant_name, with_boxes in specs:
            variant_root = out_root / variant_name
            with (variant_root / 'manifest.tsv').open(
                    'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                writer.writerows(output_rows[variant_name])
            write_subset_readme(
                variant_root,
                subset,
                len(output_rows[variant_name]),
                with_boxes,
                args,
            )
            image_count = sum(
                1 for path in variant_root.rglob('*')
                if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
            expected_images = args.expected_samples * 6
            if image_count != expected_images:
                raise RuntimeError(
                    '{}: expected {} images, found {}'.format(
                        variant_name, expected_images, image_count))
            print('{}: {} images'.format(variant_name, image_count))

    print('Paper feature-map export completed: {}'.format(out_root))


if __name__ == '__main__':
    main()
