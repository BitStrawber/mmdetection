#!/usr/bin/env python
"""Compare RUOD ResNet-50 detector feature maps on one fixed image set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
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
MODEL_SPECS = (
    ('imagenet_r50_ruod', 'ImageNet DINO R50 -> RUOD'),
    ('realuw_r50_ruod', 'RealUW DINO R50 -> RUOD'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Extract and compare four-stage feature maps from two RUOD '
            'Cascade R-CNN ResNet-50 checkpoints.'))
    parser.add_argument('--imagenet-config', required=True)
    parser.add_argument('--imagenet-checkpoint', required=True)
    parser.add_argument('--realuw-config', required=True)
    parser.add_argument('--realuw-checkpoint', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--score-threshold', type=float, default=0.30)
    parser.add_argument('--max-boxes', type=int, default=20)
    parser.add_argument('--tile-width', type=int, default=640)
    parser.add_argument('--tile-height', type=int, default=480)
    parser.add_argument('--label-height', type=int, default=42)
    parser.add_argument('--low-percentile', type=float, default=1.0)
    parser.add_argument('--high-percentile', type=float, default=99.0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def require_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError('Required file is missing or empty: {}'.format(path))
    return path


def choose_samples(
    annotation_file: Path,
    image_dir: Path,
    count: int,
    seed: int,
) -> List[dict]:
    with annotation_file.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)

    annotated_ids = {
        int(annotation['image_id'])
        for annotation in coco.get('annotations', [])
        if float(annotation.get('area', 0.0)) > 0.0
    }
    candidates = []
    for record in coco.get('images', []):
        image_id = int(record['id'])
        if image_id not in annotated_ids:
            continue
        path = image_dir / record['file_name']
        if path.is_file() and path.stat().st_size > 0:
            candidates.append({
                'image_id': image_id,
                'file_name': record['file_name'],
                'path': path.resolve(),
            })

    candidates.sort(key=lambda item: (item['image_id'], item['file_name']))
    if len(candidates) < count:
        raise RuntimeError(
            'Only {} annotated RUOD images are available; requested {}'.format(
                len(candidates), count))
    rng = random.Random(seed)
    return sorted(
        rng.sample(candidates, count),
        key=lambda item: (item['image_id'], item['file_name']),
    )


def unpack_backbone_output(output) -> List[torch.Tensor]:
    if torch.is_tensor(output):
        values = [output]
    elif isinstance(output, dict):
        values = list(output.values())
    elif isinstance(output, (list, tuple)):
        values = list(output)
    else:
        raise TypeError(
            'Unsupported backbone output type: {}'.format(type(output).__name__))
    tensors = [value for value in values if torch.is_tensor(value)]
    if len(tensors) != 4:
        raise RuntimeError(
            'Expected four ResNet stages, captured {}'.format(len(tensors)))
    return tensors


def prediction_boxes(prediction, threshold: float, max_boxes: int) -> dict:
    instances = prediction.pred_instances.to('cpu')
    if len(instances) == 0:
        return {'boxes': [], 'scores': [], 'labels': []}
    scores = instances.scores.numpy()
    order = np.argsort(-scores)
    selected = [
        int(index) for index in order
        if float(scores[index]) >= threshold
    ][:max_boxes]
    if not selected:
        return {'boxes': [], 'scores': [], 'labels': []}
    return {
        'boxes': instances.bboxes.numpy()[selected].tolist(),
        'scores': scores[selected].astype(float).tolist(),
        'labels': instances.labels.numpy()[selected].astype(int).tolist(),
    }


def shape_pair(value, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return fallback
    return int(value[0]), int(value[1])


def aggregate_valid_feature(
    feature: torch.Tensor,
    img_shape: Tuple[int, int],
    pad_shape: Tuple[int, int],
) -> np.ndarray:
    value = feature.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError('Expected CHW feature, got {}'.format(tuple(value.shape)))

    feature_h, feature_w = int(value.shape[-2]), int(value.shape[-1])
    pad_h, pad_w = pad_shape
    img_h, img_w = img_shape
    valid_h = min(
        feature_h,
        max(1, int(math.ceil(img_h * feature_h / float(max(pad_h, 1))))),
    )
    valid_w = min(
        feature_w,
        max(1, int(math.ceil(img_w * feature_w / float(max(pad_w, 1))))),
    )
    value = value[:, :valid_h, :valid_w]
    return value.abs().mean(dim=0).numpy().astype(np.float32)


def extract_model(
    config_path: Path,
    checkpoint_path: Path,
    samples: Sequence[dict],
    device: str,
    score_threshold: float,
    max_boxes: int,
) -> Dict[int, dict]:
    from mmengine.config import Config
    from mmdet.apis import inference_detector, init_detector

    config = Config.fromfile(str(config_path))
    if config.model.get('backbone', {}).get('init_cfg') is not None:
        config.model.backbone.init_cfg = None
    model = init_detector(config, str(checkpoint_path), device=device)
    model.eval()

    capture: Dict[str, object] = {}

    def hook(_module, _inputs, output):
        capture['features'] = unpack_backbone_output(output)

    handle = model.backbone.register_forward_hook(hook)
    records: Dict[int, dict] = {}
    try:
        for position, sample in enumerate(samples, start=1):
            capture.clear()
            prediction = inference_detector(model, str(sample['path']))
            if 'features' not in capture:
                raise RuntimeError(
                    'Backbone hook captured no features for {}'.format(sample['path']))

            metadata = prediction.metainfo
            img_shape = shape_pair(metadata.get('img_shape'), (1, 1))
            pad_shape = shape_pair(
                metadata.get('pad_shape', metadata.get('batch_input_shape')),
                img_shape,
            )
            activations = [
                aggregate_valid_feature(feature, img_shape, pad_shape)
                for feature in capture['features']
            ]
            records[int(sample['image_id'])] = {
                'activations': activations,
                'boxes': prediction_boxes(
                    prediction, score_threshold, max_boxes),
                'img_shape': list(img_shape),
                'pad_shape': list(pad_shape),
            }
            print(
                '[{}/{}] {} {}'.format(
                    position, len(samples), checkpoint_path.name,
                    sample['file_name']),
                flush=True,
            )
    finally:
        handle.remove()
        capture.clear()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def shared_normalize(
    first: np.ndarray,
    second: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    first_valid = first[np.isfinite(first)]
    second_valid = second[np.isfinite(second)]
    combined = np.concatenate((first_valid, second_valid))
    if combined.size == 0:
        zeros_first = np.zeros(first.shape, dtype=np.float32)
        zeros_second = np.zeros(second.shape, dtype=np.float32)
        return zeros_first, zeros_second, 0.0, 0.0

    low = float(np.percentile(combined, low_percentile))
    high = float(np.percentile(combined, high_percentile))
    if high <= low:
        low = float(combined.min())
        high = float(combined.max())
    scale = max(high - low, 1e-12)

    def normalize(value: np.ndarray) -> np.ndarray:
        result = (value - low) / scale
        return np.nan_to_num(np.clip(result, 0.0, 1.0)).astype(np.float32)

    return normalize(first), normalize(second), low, high


def colorize(normalized: np.ndarray) -> Image.Image:
    output = np.zeros(normalized.shape + (3,), dtype=np.float32)
    for index in range(len(PALETTE_STOPS) - 1):
        start_value, start_color = PALETTE_STOPS[index]
        end_value, end_color = PALETTE_STOPS[index + 1]
        if index == len(PALETTE_STOPS) - 2:
            mask = (normalized >= start_value) & (normalized <= end_value)
        else:
            mask = (normalized >= start_value) & (normalized < end_value)
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


def labeled_tile(
    image: Image.Image,
    label: str,
    size: Tuple[int, int],
    label_height: int,
) -> Image.Image:
    tile = fit_fixed(image, size)
    output = Image.new('RGB', (size[0], size[1] + label_height), 'white')
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, size[0], label_height), fill=(238, 238, 238))
    draw.text((12, 13), label, fill=(15, 15, 15))
    output.paste(tile, (0, label_height))
    return output


def join_horizontal(tiles: Iterable[Image.Image]) -> Image.Image:
    values = list(tiles)
    output = Image.new(
        'RGB',
        (sum(tile.width for tile in values), max(tile.height for tile in values)),
        'white',
    )
    offset = 0
    for tile in values:
        output.paste(tile, (offset, 0))
        offset += tile.width
    return output


def join_vertical(rows: Iterable[Image.Image]) -> Image.Image:
    values = list(rows)
    output = Image.new(
        'RGB',
        (max(row.width for row in values), sum(row.height for row in values)),
        'white',
    )
    offset = 0
    for row in values:
        output.paste(row, (0, offset))
        offset += row.height
    return output


def prepare_model_dirs(root: Path) -> Dict[str, Dict[str, Path]]:
    result = {}
    for variant in ('no_box', 'with_box'):
        result[variant] = {
            'originals': root / variant / 'originals',
            'feature_maps': root / variant / 'feature_maps',
            'panels': root / variant / 'five_panels',
        }
        for path in result[variant].values():
            path.mkdir(parents=True, exist_ok=True)
    return result


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError('--samples must be positive')
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError('Invalid percentile range')

    imagenet_config = require_file(args.imagenet_config)
    imagenet_checkpoint = require_file(args.imagenet_checkpoint)
    realuw_config = require_file(args.realuw_config)
    realuw_checkpoint = require_file(args.realuw_checkpoint)
    image_dir = Path(args.image_dir).resolve()
    annotation_file = require_file(args.annotation_file)
    if not image_dir.is_dir():
        raise NotADirectoryError('RUOD image directory not found: {}'.format(image_dir))

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                'Output directory already exists; use --overwrite: {}'.format(out_dir))
        if out_dir == Path(out_dir.anchor):
            raise RuntimeError('Refusing to reset filesystem root')
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    samples = choose_samples(
        annotation_file, image_dir, args.samples, args.seed)
    model_inputs = {
        'imagenet_r50_ruod': (imagenet_config, imagenet_checkpoint),
        'realuw_r50_ruod': (realuw_config, realuw_checkpoint),
    }
    extracted = {}
    for model_key, _model_label in MODEL_SPECS:
        config, checkpoint = model_inputs[model_key]
        print('Extracting model: {}'.format(model_key), flush=True)
        extracted[model_key] = extract_model(
            config,
            checkpoint,
            samples,
            args.device,
            args.score_threshold,
            args.max_boxes,
        )

    model_dirs = {
        key: prepare_model_dirs(out_dir / key)
        for key, _label in MODEL_SPECS
    }
    comparison_dirs = {
        variant: out_dir / 'comparison_panels' / variant
        for variant in ('no_box', 'with_box')
    }
    for path in comparison_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    selected_originals = out_dir / 'selected_originals'
    selected_originals.mkdir(parents=True)

    size = (args.tile_width, args.tile_height)
    manifest_rows = []
    normalization_rows = []

    for sample_index, sample in enumerate(samples, start=1):
        image_id = int(sample['image_id'])
        with Image.open(str(sample['path'])) as opened:
            original = ImageOps.exif_transpose(opened).convert('RGB')
        sample_name = '{:02d}_id{}_{}'.format(
            sample_index, image_id, Path(sample['file_name']).stem)
        fit_fixed(original, size).save(
            str(selected_originals / '{}_original.png'.format(sample_name)))

        heatmaps: Dict[str, List[Image.Image]] = {
            key: [] for key, _label in MODEL_SPECS
        }
        for stage_index in range(4):
            first = extracted['imagenet_r50_ruod'][image_id]['activations'][stage_index]
            second = extracted['realuw_r50_ruod'][image_id]['activations'][stage_index]
            first_normalized, second_normalized, low, high = shared_normalize(
                first,
                second,
                args.low_percentile,
                args.high_percentile,
            )
            normalization_rows.append({
                'sample': sample_name,
                'image_id': image_id,
                'stage': stage_index + 1,
                'shared_low': low,
                'shared_high': high,
            })
            for model_key, normalized in (
                ('imagenet_r50_ruod', first_normalized),
                ('realuw_r50_ruod', second_normalized),
            ):
                heatmap = colorize(normalized).resize(
                    original.size, RESAMPLE.BICUBIC)
                heatmaps[model_key].append(heatmap)

        comparison_rows = {'no_box': [], 'with_box': []}
        for model_key, model_label in MODEL_SPECS:
            prediction = extracted[model_key][image_id]['boxes']
            boxes = prediction['boxes']
            for variant in ('no_box', 'with_box'):
                paths = model_dirs[model_key][variant]
                use_boxes = variant == 'with_box'
                rendered_original = (
                    draw_boxes(original, boxes, (255, 230, 0))
                    if use_boxes else original
                )
                fixed_original = fit_fixed(rendered_original, size)
                fixed_original.save(str(
                    paths['originals'] / '{}_original.png'.format(sample_name)))

                sample_map_dir = paths['feature_maps'] / sample_name
                sample_map_dir.mkdir(parents=True)
                rendered_maps = []
                for stage_index, heatmap in enumerate(
                        heatmaps[model_key], start=1):
                    rendered = (
                        draw_boxes(heatmap, boxes, (255, 40, 40))
                        if use_boxes else heatmap
                    )
                    rendered_maps.append(rendered)
                    fit_fixed(rendered, size).save(str(
                        sample_map_dir / 'stage{}.png'.format(stage_index)))

                tiles = [
                    labeled_tile(
                        rendered_original,
                        '{} | Original'.format(model_label),
                        size,
                        args.label_height,
                    )
                ]
                tiles.extend(
                    labeled_tile(
                        heatmap,
                        '{} | Stage {}'.format(model_label, stage_index),
                        size,
                        args.label_height,
                    )
                    for stage_index, heatmap in enumerate(
                        rendered_maps, start=1)
                )
                row = join_horizontal(tiles)
                row.save(str(
                    paths['panels'] / '{}_five_panel.png'.format(sample_name)))
                comparison_rows[variant].append(row)

            manifest_rows.append({
                'sample': sample_name,
                'image_id': image_id,
                'file_name': sample['file_name'],
                'source_path': str(sample['path']),
                'model': model_key,
                'checkpoint': str(model_inputs[model_key][1]),
                'box_count': len(boxes),
                'scores': json.dumps(prediction['scores']),
                'labels': json.dumps(prediction['labels']),
                'img_shape': json.dumps(
                    extracted[model_key][image_id]['img_shape']),
                'pad_shape': json.dumps(
                    extracted[model_key][image_id]['pad_shape']),
            })

        for variant, rows in comparison_rows.items():
            join_vertical(rows).save(str(
                comparison_dirs[variant] /
                '{}_two_model_comparison.png'.format(sample_name)))
        print(
            'Rendered [{}/{}] {}'.format(
                sample_index, len(samples), sample['file_name']),
            flush=True,
        )

    manifest_fields = [
        'sample', 'image_id', 'file_name', 'source_path', 'model',
        'checkpoint', 'box_count', 'scores', 'labels', 'img_shape', 'pad_shape',
    ]
    with (out_dir / 'manifest.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=manifest_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (out_dir / 'shared_normalization.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'sample', 'image_id', 'stage', 'shared_low', 'shared_high'],
            delimiter='\t',
        )
        writer.writeheader()
        writer.writerows(normalization_rows)

    summary = {
        'samples': args.samples,
        'seed': args.seed,
        'image_dir': str(image_dir),
        'annotation_file': str(annotation_file),
        'device': args.device,
        'aggregation': 'mean(abs(feature), channel)',
        'normalization': (
            'shared between the two models for each image and stage'),
        'normalization_percentiles': [
            args.low_percentile, args.high_percentile],
        'tile_size': [args.tile_width, args.tile_height],
        'score_threshold': args.score_threshold,
        'max_boxes': args.max_boxes,
        'models': {
            model_key: {
                'label': model_label,
                'config': str(model_inputs[model_key][0]),
                'checkpoint': str(model_inputs[model_key][1]),
            }
            for model_key, model_label in MODEL_SPECS
        },
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'README.txt').write_text(
        '\n'.join([
            'RUOD ResNet-50 detector feature-map comparison',
            'The same {} annotated validation images are used for both models.'.format(
                args.samples),
            'Feature stages: ResNet-50 layer1, layer2, layer3, layer4.',
            'Aggregation: channel-wise mean absolute activation.',
            (
                'Normalization: shared percentile bounds across both models '
                'for each image and stage.'),
            'Palette: deep blue -> blue -> cyan -> yellow.',
            'no_box: feature maps without predictions.',
            'with_box: each detector uses its own predicted boxes.',
            'comparison_panels: ImageNet-pretrained model above RealUW-pretrained model.',
        ]) + '\n',
        encoding='utf-8',
    )

    expected_model_images = args.samples * 6
    for model_key, _model_label in MODEL_SPECS:
        for variant in ('no_box', 'with_box'):
            root = out_dir / model_key / variant
            count = sum(
                1 for path in root.rglob('*.png') if path.is_file())
            if count != expected_model_images:
                raise RuntimeError(
                    '{} {}: expected {} PNG files, found {}'.format(
                        model_key, variant, expected_model_images, count))
    for variant in ('no_box', 'with_box'):
        count = sum(
            1 for path in comparison_dirs[variant].glob('*.png')
            if path.is_file())
        if count != args.samples:
            raise RuntimeError(
                'comparison {}: expected {}, found {}'.format(
                    variant, args.samples, count))

    print('Feature-map comparison completed: {}'.format(out_dir))


if __name__ == '__main__':
    main()
