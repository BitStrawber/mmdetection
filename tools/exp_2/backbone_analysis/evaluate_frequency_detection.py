#!/usr/bin/env python3
"""Evaluate complete detectors on clean and frequency-degraded RUOD samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .common import ensure_empty_or_create, read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency-manifest', required=True)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--models-config', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--variants', default='clean,remove_low,remove_mid,remove_high')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--allow-index-category-map', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def variant_kind(name: str) -> Tuple[str, str]:
    if name.startswith('remove_'):
        return 'band_stop', name[len('remove_'):]
    if name == 'clean':
        return 'clean', 'clean'
    return 'band_pass', name


def category_mapping(
    model: object, annotation: Mapping[str, object], allow_index: bool,
) -> Tuple[Dict[int, int], str]:
    categories = list(annotation['categories'])
    by_name = {str(item['name']): int(item['id']) for item in categories}
    classes = list(getattr(model, 'dataset_meta', {}).get('classes', []))
    if classes and all(str(name) in by_name for name in classes):
        return {index: by_name[str(name)] for index, name in enumerate(classes)}, 'class_name'
    if allow_index and len(classes) == len(categories):
        ordered = sorted(int(item['id']) for item in categories)
        return {index: category_id for index, category_id in enumerate(ordered)}, 'index'
    raise RuntimeError(
        'Could not map model labels to COCO category IDs by class name. '
        'Inspect model.dataset_meta or use --allow-index-category-map only '
        'when the configured class order is known to match the annotation.')


def predictions_for_variant(
    model: object, rows: Sequence[dict], variant: str,
    label_to_category: Mapping[int, int],
) -> List[dict]:
    from mmdet.apis import inference_detector

    output = []
    for position, row in enumerate(rows, start=1):
        path = row['variants'][variant]['image_path']
        prediction = inference_detector(model, path)
        instances = prediction.pred_instances.cpu()
        boxes = instances.bboxes.numpy()
        scores = instances.scores.numpy()
        labels = instances.labels.numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(value) for value in box]
            output.append({
                'image_id': int(row['image_id']),
                'category_id': int(label_to_category[int(label)]),
                'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                'score': float(score),
            })
        print(f'[{variant} {position}/{len(rows)}] {row["file_name"]}', flush=True)
    return output


def coco_metrics(
    coco_gt: object, detections: Sequence[dict], image_ids: Sequence[int],
) -> Mapping[str, float]:
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(list(detections))
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75',
             'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l')
    return {name: float(evaluator.stats[index]) for index, name in enumerate(names)}


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.frequency_manifest)
    variants = parse_csv(args.variants)
    selected_models = set(parse_csv(args.models))
    models_config = read_json(args.models_config)
    specs = [spec for spec in models_config['models'] if spec.get('kind') == 'detector']
    if selected_models:
        specs = [spec for spec in specs if str(spec['id']) in selected_models]
    if not specs:
        raise ValueError('No complete detector entries selected from models-config')
    annotation = read_json(args.annotation_file)
    image_ids = [int(row['image_id']) for row in rows]
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)

    from mmdet.apis import init_detector
    from pycocotools.coco import COCO

    coco_gt = COCO(args.annotation_file)
    metric_rows = []
    mapping_reports = {}
    for spec in specs:
        model_id = str(spec['id'])
        model = init_detector(spec['config'], spec['checkpoint'], device=args.device)
        model.eval()
        label_map, strategy = category_mapping(
            model, annotation, args.allow_index_category_map)
        mapping_reports[model_id] = {
            'strategy': strategy,
            'label_to_category_id': label_map,
        }
        model_metrics = {}
        prediction_root = out_dir / 'predictions' / model_id
        prediction_root.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            detections = predictions_for_variant(model, rows, variant, label_map)
            with (prediction_root / f'{variant}.json').open('w', encoding='utf-8') as handle:
                json.dump(detections, handle)
            model_metrics[variant] = coco_metrics(coco_gt, detections, image_ids)
        clean = model_metrics['clean']
        for variant in variants:
            kind, band = variant_kind(variant)
            metrics = model_metrics[variant]
            metric_rows.append({
                'model': model_id,
                'variant': variant,
                'variant_kind': kind,
                'band': band,
                **metrics,
                'clean_bbox_mAP': clean['bbox_mAP'],
                'bbox_mAP_drop': clean['bbox_mAP'] - metrics['bbox_mAP'],
                'bbox_mAP_retention': (
                    metrics['bbox_mAP'] / max(clean['bbox_mAP'], 1e-12)),
                'clean_bbox_mAP_50': clean['bbox_mAP_50'],
                'bbox_mAP_50_drop': clean['bbox_mAP_50'] - metrics['bbox_mAP_50'],
                'bbox_mAP_50_retention': (
                    metrics['bbox_mAP_50'] / max(clean['bbox_mAP_50'], 1e-12)),
            })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_tsv(out_dir / 'frequency_detection_metrics.tsv', metric_rows)
    write_json(out_dir / 'frequency_detection_metadata.json', {
        'frequency_manifest': str(Path(args.frequency_manifest).resolve()),
        'annotation_file': str(Path(args.annotation_file).resolve()),
        'models_config': str(Path(args.models_config).resolve()),
        'models': [str(spec['id']) for spec in specs],
        'variants': variants,
        'sample_image_ids': image_ids,
        'sample_count': len(rows),
        'category_mapping': mapping_reports,
        'warning': (
            'Subset COCO AP can be noisy. Use the complete validation set for '
            'the primary robustness result.'),
    })
    print(f'Frequency detection evaluation: {out_dir}')


if __name__ == '__main__':
    main()
