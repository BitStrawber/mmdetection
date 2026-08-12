#!/usr/bin/env python3
"""Extract prediction-conditioned XGradCAM from RUOD Cascade R-CNNs."""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from mmcv.transforms import Compose


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import (  # noqa: E402
    atomic_save_npz,
    atomic_write_json,
    clean_name,
    clip_box,
    existing_file,
    parse_csv,
    read_json,
    read_jsonl,
)
from extract_fixed_gt_xgradcam import (  # noqa: E402
    build_label_mapping,
    crop_valid_map,
    load_instances,
    prepare_image,
    scale_factor_xy,
    shape_pair,
    tensor_from_output,
    xgradcam_raw,
)
from tools.exp_2.backbone_analysis.model_adapter import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--models-config', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cascade-stage', type=int, default=-1)
    parser.add_argument('--score-threshold', type=float, default=0.30)
    parser.add_argument('--max-predictions-per-image', type=int, default=10)
    parser.add_argument('--minimum-box-area', type=float, default=4.0)
    parser.add_argument('--match-iou-threshold', type=float, default=0.50)
    parser.add_argument(
        '--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def prediction_dir(
    root: Path, model_id: str, image_id: int, rank: int,
) -> Path:
    return (
        root / 'raw_cam' / clean_name(model_id) /
        f'image_{int(image_id):08d}' / f'pred_{int(rank):03d}'
    )


def prediction_cam_path(
    root: Path, model_id: str, image_id: int, rank: int, layer: str,
) -> Path:
    return prediction_dir(root, model_id, image_id, rank) / f'{clean_name(layer)}.npz'


def prediction_metadata_path(
    root: Path, model_id: str, image_id: int, rank: int,
) -> Path:
    return prediction_dir(root, model_id, image_id, rank) / 'prediction.json'


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def best_gt_match(
    prediction_box: Sequence[float],
    prediction_label: int,
    gt_instances: Sequence[Mapping[str, Any]],
    category_to_label: Mapping[int, int],
    threshold: float,
) -> Dict[str, Any]:
    if not gt_instances:
        return {
            'matched': False,
            'best_gt_iou': 0.0,
            'matched_annotation_id': None,
            'matched_category_id': None,
            'matched_category_name': None,
            'matched_bbox_xyxy_original': None,
            'class_correct': False,
            'tp_at_match_iou': False,
        }
    scored = [
        (box_iou(prediction_box, row['bbox_xyxy']), row)
        for row in gt_instances
    ]
    iou, matched = max(scored, key=lambda item: item[0])
    matched_label = category_to_label[int(matched['category_id'])]
    class_correct = int(prediction_label) == int(matched_label)
    return {
        'matched': bool(iou >= threshold),
        'best_gt_iou': float(iou),
        'matched_annotation_id': int(matched['annotation_id']),
        'matched_category_id': int(matched['category_id']),
        'matched_bbox_xyxy_original': list(matched['bbox_xyxy']),
        'class_correct': bool(class_correct),
        'tp_at_match_iou': bool(class_correct and iou >= threshold),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError('--score-threshold must be in [0, 1]')
    if args.max_predictions_per_image <= 0:
        raise ValueError('--max-predictions-per-image must be positive')
    if not 0.0 <= args.match_iou_threshold <= 1.0:
        raise ValueError('--match-iou-threshold must be in [0, 1]')

    manifest_path = existing_file(args.manifest)
    annotation_path = existing_file(args.annotation_file)
    models_config_path = existing_file(args.models_config)
    manifest = read_jsonl(manifest_path)
    model_config = read_json(models_config_path)
    specs = list(model_config.get('models', []))
    selected_models = set(parse_csv(args.models))
    if selected_models:
        specs = [spec for spec in specs if str(spec['id']) in selected_models]
    if not specs:
        raise ValueError('No detector models selected')
    for spec in specs:
        if str(spec.get('kind', 'detector')) != 'detector':
            raise ValueError(
                f'{spec.get("id")}: prediction XGradCAM requires a complete detector')
    selected_layers = parse_csv(args.layers)
    if not selected_layers:
        raise ValueError('At least one layer is required')

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not (
            args.resume or args.overwrite):
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [int(row['image_id']) for row in manifest]
    _, all_gt_by_image, categories = load_instances(
        annotation_path, image_ids, 1.0, True, 0, 'annotation-id')
    category_names = {
        int(item['id']): str(item['name']) for item in categories
    }

    from mmdet.apis import inference_detector
    from mmdet.utils import get_test_pipeline_cfg

    reports: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    for model_position, spec in enumerate(specs, 1):
        loaded = load_model(spec, args.device)
        model = loaded.model
        model_id = loaded.model_id
        reports[model_id] = loaded.load_report
        prediction_count = 0
        image_count = 0
        try:
            if not hasattr(model, 'roi_head') or not hasattr(model.roi_head, '_bbox_forward'):
                raise TypeError(f'{model_id}: model has no compatible ROI bbox head')
            unknown_layers = set(selected_layers) - set(loaded.layers)
            if unknown_layers:
                raise ValueError(f'{model_id}: unavailable layers {sorted(unknown_layers)}')
            num_stages = int(getattr(model.roi_head, 'num_stages', 1))
            stage = args.cascade_stage if args.cascade_stage >= 0 else num_stages - 1
            if not 0 <= stage < num_stages:
                raise ValueError(f'{model_id}: invalid Cascade stage {stage}/{num_stages}')
            label_map, _, mapping_method = build_label_mapping(categories, model)
            label_to_category = {label: category for category, label in label_map.items()}
            model_classes = list((getattr(model, 'dataset_meta', {}) or {}).get('classes', []))

            captures: Dict[str, torch.Tensor] = {}
            handles = []
            for layer_id in selected_layers:
                def hook(_module, _inputs, output, key=layer_id):
                    captures[key] = tensor_from_output(output)
                handles.append(loaded.layers[layer_id].register_forward_hook(hook))
            pipeline = Compose(get_test_pipeline_cfg(model.cfg))
            try:
                for image_position, row in enumerate(manifest, 1):
                    image_id = int(row['image_id'])
                    image_path = existing_file(row['image_path'])

                    # Detector inference chooses the actual post-NMS boxes/classes.
                    with torch.no_grad():
                        result = inference_detector(model, str(image_path))
                    predicted = result.pred_instances.cpu()
                    boxes = predicted.bboxes.numpy().astype(np.float32)
                    scores = predicted.scores.numpy().astype(np.float32)
                    labels = predicted.labels.numpy().astype(np.int64)
                    keep = np.flatnonzero(scores >= args.score_threshold)
                    if keep.size:
                        keep = keep[np.argsort(-scores[keep], kind='stable')]
                    keep = keep[:args.max_predictions_per_image]
                    selected = []
                    for index in keep.tolist():
                        box = clip_box(boxes[index], int(row['width']), int(row['height']))
                        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                        if area >= args.minimum_box_area:
                            selected.append((index, box, area))

                    summary_path = (
                        out_dir / 'raw_cam' / clean_name(model_id) /
                        f'image_{image_id:08d}' / 'image_predictions.json')
                    expected_complete = all(
                        prediction_metadata_path(
                            out_dir, model_id, image_id, rank).is_file() and
                        all(prediction_cam_path(
                            out_dir, model_id, image_id, rank, layer).is_file()
                            for layer in selected_layers)
                        for rank in range(len(selected)))
                    if (args.resume and summary_path.is_file() and expected_complete
                            and not args.overwrite):
                        prediction_count += len(selected)
                        image_count += 1
                        print(
                            f'REUSE [{model_id}] {image_position}/{len(manifest)} '
                            f'{row["file_name"]}: {len(selected)} predictions', flush=True)
                        continue

                    if not selected:
                        atomic_write_json(summary_path, {
                            'model': model_id,
                            'image_id': image_id,
                            'image_path': str(image_path),
                            'file_name': str(row['file_name']),
                            'score_threshold': args.score_threshold,
                            'max_predictions_per_image': args.max_predictions_per_image,
                            'prediction_count': 0,
                            'predictions': [],
                        })
                        image_count += 1
                        print(
                            f'[{model_id}] {image_position}/{len(manifest)} '
                            f'{row["file_name"]}: no predictions', flush=True)
                        continue

                    captures.clear()
                    model.zero_grad(set_to_none=True)
                    batch_inputs, data_samples = prepare_image(
                        model, pipeline, image_path, image_id)
                    features = model.extract_feat(batch_inputs)
                    missing = set(selected_layers) - set(captures)
                    if missing:
                        raise RuntimeError(f'{model_id}: hooks missed layers {sorted(missing)}')
                    activations = [captures[layer] for layer in selected_layers]
                    non_differentiable = [
                        layer for layer, activation in zip(
                            selected_layers, activations)
                        if not activation.requires_grad
                    ]
                    if non_differentiable:
                        raise RuntimeError(
                            f'{model_id}: target activations do not require gradients: '
                            f'{non_differentiable}. Ensure CAM inference is not wrapped in '
                            'torch.no_grad() and batch inputs require gradients.')
                    metadata = data_samples[0].metainfo
                    ori_shape = shape_pair(
                        metadata.get('ori_shape'),
                        (int(row['height']), int(row['width'])))
                    img_shape = shape_pair(metadata.get('img_shape'), ori_shape)
                    pad_shape = shape_pair(
                        metadata.get('pad_shape', metadata.get('batch_input_shape')),
                        img_shape)
                    scale_x, scale_y = scale_factor_xy(metadata)
                    all_gt = all_gt_by_image.get(image_id, [])
                    all_gt_boxes = [item['bbox_xyxy'] for item in all_gt]
                    image_prediction_rows: List[Dict[str, Any]] = []

                    for rank, (source_index, original_box, area) in enumerate(selected):
                        prediction_label = int(labels[source_index])
                        prediction_score = float(scores[source_index])
                        category_id = label_to_category.get(prediction_label)
                        category_name = (
                            category_names.get(category_id, f'label_{prediction_label}')
                            if category_id is not None else
                            (str(model_classes[prediction_label])
                             if prediction_label < len(model_classes)
                             else f'label_{prediction_label}'))
                        match = best_gt_match(
                            original_box, prediction_label, all_gt,
                            label_map, args.match_iou_threshold)
                        if match['matched_category_id'] is not None:
                            match['matched_category_name'] = category_names.get(
                                int(match['matched_category_id']), 'unknown')
                        network_box = clip_box([
                            original_box[0] * scale_x,
                            original_box[1] * scale_y,
                            original_box[2] * scale_x,
                            original_box[3] * scale_y,
                        ], img_shape[1], img_shape[0])
                        rois = batch_inputs.new_tensor([[
                            0.0, network_box[0], network_box[1],
                            network_box[2], network_box[3],
                        ]])
                        bbox_result = model.roi_head._bbox_forward(stage, features, rois)
                        cls_score = bbox_result['cls_score']
                        if prediction_label >= cls_score.shape[1] - 1:
                            raise IndexError(
                                f'{model_id}: predicted label {prediction_label} is invalid '
                                f'for cls_score shape {tuple(cls_score.shape)}')
                        target_logit = cls_score[0, prediction_label]
                        gradients = torch.autograd.grad(
                            target_logit, activations, retain_graph=True,
                            create_graph=False, allow_unused=False)
                        recomputed_probability = float(
                            torch.softmax(cls_score, dim=1)[0, prediction_label])
                        layer_records = {}
                        for layer_id, activation, gradient in zip(
                                selected_layers, activations, gradients):
                            cam = xgradcam_raw(activation, gradient)
                            cam_valid, valid_shape = crop_valid_map(
                                cam, img_shape, pad_shape)
                            destination = prediction_cam_path(
                                out_dir, model_id, image_id, rank, layer_id)
                            atomic_save_npz(
                                destination,
                                cam=cam_valid.detach().float().cpu().numpy().astype(np.float32),
                                image_id=np.int64(image_id),
                                prediction_rank=np.int64(rank),
                                prediction_label=np.int64(prediction_label),
                                prediction_score=np.float32(prediction_score),
                                target_logit=np.float32(target_logit.detach().cpu()),
                                target_probability_recomputed=np.float32(
                                    recomputed_probability),
                                bbox_xyxy_original=np.asarray(original_box, dtype=np.float32),
                                bbox_xyxy_network_input=np.asarray(network_box, dtype=np.float32),
                                ori_shape=np.asarray(ori_shape, dtype=np.int32),
                                img_shape=np.asarray(img_shape, dtype=np.int32),
                                pad_shape=np.asarray(pad_shape, dtype=np.int32),
                                valid_feature_shape=np.asarray(valid_shape, dtype=np.int32),
                            )
                            layer_records[layer_id] = str(destination)
                        record = {
                            'model': model_id,
                            'model_position': model_position,
                            'image_id': image_id,
                            'prediction_rank': rank,
                            'prediction_label': prediction_label,
                            'prediction_category_id': category_id,
                            'prediction_category_name': category_name,
                            'prediction_score': prediction_score,
                            'target_logit': float(target_logit.detach().cpu()),
                            'target_probability_recomputed': recomputed_probability,
                            'class_mapping_method': mapping_method,
                            'image_path': str(image_path),
                            'file_name': str(row['file_name']),
                            'bbox_xyxy_original': original_box,
                            'bbox_xyxy_network_input': network_box,
                            'area': float(area),
                            'all_gt_boxes_xyxy_original': all_gt_boxes,
                            'cascade_stage_zero_based': stage,
                            'match_iou_threshold': args.match_iou_threshold,
                            **match,
                            'ori_shape': list(ori_shape),
                            'img_shape': list(img_shape),
                            'pad_shape': list(pad_shape),
                            'scale_factor_xy': [scale_x, scale_y],
                            'layers': layer_records,
                        }
                        atomic_write_json(
                            prediction_metadata_path(
                                out_dir, model_id, image_id, rank), record)
                        image_prediction_rows.append(record)
                        prediction_count += 1
                    atomic_write_json(summary_path, {
                        'model': model_id,
                        'image_id': image_id,
                        'image_path': str(image_path),
                        'file_name': str(row['file_name']),
                        'score_threshold': args.score_threshold,
                        'max_predictions_per_image': args.max_predictions_per_image,
                        'prediction_count': len(image_prediction_rows),
                        'predictions': [
                            {
                                'prediction_rank': item['prediction_rank'],
                                'prediction_category_name': item['prediction_category_name'],
                                'prediction_score': item['prediction_score'],
                                'bbox_xyxy_original': item['bbox_xyxy_original'],
                                'best_gt_iou': item['best_gt_iou'],
                                'class_correct': item['class_correct'],
                                'tp_at_match_iou': item['tp_at_match_iou'],
                            }
                            for item in image_prediction_rows
                        ],
                    })
                    image_count += 1
                    print(
                        f'[{model_id}] {image_position}/{len(manifest)} '
                        f'{row["file_name"]}: {len(selected)} predictions', flush=True)
                    del features, batch_inputs, data_samples
                    captures.clear()
                    gc.collect()
            finally:
                for handle in handles:
                    handle.remove()
                captures.clear()
            summaries[model_id] = {
                'images': image_count,
                'predictions': prediction_count,
                'score_threshold': args.score_threshold,
                'max_predictions_per_image': args.max_predictions_per_image,
                'match_iou_threshold': args.match_iou_threshold,
                'cascade_stage_zero_based': stage,
            }
        finally:
            loaded.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    worker_name = clean_name('_'.join(str(spec['id']) for spec in specs))
    atomic_write_json(
        out_dir / 'model_load_reports' / f'{worker_name}.json', reports)
    atomic_write_json(
        out_dir / 'extraction_summaries' / f'{worker_name}.json', {
            'method': 'prediction-conditioned XGradCAM',
            'raw_cam_state': 'ReLU applied; no display normalization',
            'manifest': str(manifest_path),
            'annotation_file': str(annotation_path),
            'models_config': str(models_config_path),
            'models': [str(spec['id']) for spec in specs],
            'layers': selected_layers,
            'device': args.device,
            'model_summaries': summaries,
            'warning': (
                'Predicted ROIs/classes differ between detectors. Use this branch for '
                'behavior/error analysis, not as the controlled replacement for fixed-GT CAM.'),
        })
    print(f'Prediction-conditioned XGradCAM complete: {out_dir}')


if __name__ == '__main__':
    main()
