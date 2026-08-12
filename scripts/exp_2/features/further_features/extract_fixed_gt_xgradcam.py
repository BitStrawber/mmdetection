#!/usr/bin/env python3
"""Extract unnormalized fixed-GT XGradCAM from MMDetection Cascade R-CNNs."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
    atomic_write_jsonl,
    clean_name,
    clip_box,
    existing_file,
    instance_metadata_path,
    parse_csv,
    raw_cam_path,
    read_json,
    read_jsonl,
)
from tools.exp_2.backbone_analysis.model_adapter import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--models-config', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', default='', help='Optional model ID subset')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--cascade-stage', type=int, default=-1,
        help='Zero-based ROI bbox stage; -1 means the last stage')
    parser.add_argument(
        '--max-instances-per-image', type=int, default=0,
        help='0 keeps every valid non-crowd GT instance')
    parser.add_argument('--minimum-box-area', type=float, default=1.0)
    parser.add_argument(
        '--instance-order', choices=('annotation-id', 'area-desc'),
        default='annotation-id')
    parser.add_argument(
        '--exclude-crowd', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f'Expected one tensor from target layer, got {type(output)}')


def shape_pair(value: Any, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return fallback
    return int(value[0]), int(value[1])


def crop_valid_map(
    value: torch.Tensor,
    img_shape: Tuple[int, int],
    pad_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = int(value.shape[-2]), int(value.shape[-1])
    img_h, img_w = img_shape
    pad_h, pad_w = pad_shape
    valid_h = min(height, max(1, int(math.ceil(img_h * height / max(pad_h, 1)))))
    valid_w = min(width, max(1, int(math.ceil(img_w * width / max(pad_w, 1)))))
    return value[:valid_h, :valid_w], (valid_h, valid_w)


def xgradcam_raw(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Return the nonnegative XGradCAM before any min-max normalization."""
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise ValueError(
            f'Invalid activation/gradient shapes: {activation.shape}, {gradient.shape}')
    numerator = (gradient * activation).sum(dim=(-2, -1))
    denominator = activation.sum(dim=(-2, -1)) + 2e-7
    weights = numerator / denominator
    cam = (weights[..., None, None] * activation).sum(dim=1)
    return torch.relu(cam)[0]


def scale_factor_xy(metadata: Mapping[str, Any]) -> Tuple[float, float]:
    factor = np.asarray(metadata.get('scale_factor', [1.0, 1.0]), dtype=np.float32)
    factor = factor.reshape(-1)
    if factor.size < 2:
        return float(factor[0]), float(factor[0])
    return float(factor[0]), float(factor[1])


def build_label_mapping(
    categories: Sequence[Mapping[str, Any]],
    model: Any,
) -> Tuple[Dict[int, int], Dict[int, str], str]:
    category_names = {int(item['id']): str(item['name']) for item in categories}
    num_classes = int(model.roi_head.bbox_head[-1].num_classes)
    model_classes = list((getattr(model, 'dataset_meta', {}) or {}).get('classes', []))
    model_name_to_label = {str(name): index for index, name in enumerate(model_classes)}
    if category_names and all(name in model_name_to_label for name in category_names.values()):
        mapping = {
            category_id: model_name_to_label[name]
            for category_id, name in category_names.items()
        }
        method = 'category-name-to-checkpoint-dataset-meta'
    elif len(categories) == num_classes:
        mapping = {
            int(category['id']): index for index, category in enumerate(categories)
        }
        method = 'annotation-category-order'
    else:
        raise RuntimeError(
            f'Cannot map {len(categories)} annotation categories to '
            f'{num_classes} detector classes; checkpoint classes={model_classes}')
    if max(mapping.values(), default=-1) >= num_classes:
        raise RuntimeError(f'Class mapping exceeds bbox head size: {mapping}')
    return mapping, category_names, method


def load_instances(
    annotation_file: Path,
    selected_image_ids: Sequence[int],
    minimum_area: float,
    exclude_crowd: bool,
    max_per_image: int,
    instance_order: str,
) -> Tuple[
    Dict[int, List[Dict[str, Any]]],
    Dict[int, List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    coco = read_json(annotation_file)
    selected = set(int(value) for value in selected_image_ids)
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get('annotations', []):
        image_id = int(annotation['image_id'])
        if image_id not in selected:
            continue
        if exclude_crowd and int(annotation.get('iscrowd', 0)):
            continue
        x, y, width, height = [float(value) for value in annotation['bbox']]
        area = float(annotation.get('area', width * height))
        if width <= 0 or height <= 0 or area < minimum_area:
            continue
        grouped[image_id].append({
            'annotation_id': int(annotation['id']),
            'image_id': image_id,
            'category_id': int(annotation['category_id']),
            'bbox_xyxy': [x, y, x + width, y + height],
            'area': area,
            'iscrowd': int(annotation.get('iscrowd', 0)),
        })
    all_valid = {
        image_id: [dict(item) for item in rows]
        for image_id, rows in grouped.items()
    }
    for image_id, rows in grouped.items():
        if instance_order == 'area-desc':
            rows.sort(key=lambda item: (-item['area'], item['annotation_id']))
        else:
            rows.sort(key=lambda item: item['annotation_id'])
        if max_per_image > 0:
            grouped[image_id] = rows[:max_per_image]
    return grouped, all_valid, list(coco.get('categories', []))


def prepare_image(model: Any, pipeline: Compose, path: Path, image_id: int):
    data = pipeline(dict(img_path=str(path), img_id=int(image_id)))
    data['inputs'] = [data['inputs']]
    data['data_samples'] = [data['data_samples']]
    processed = model.data_preprocessor(data, training=False)
    return processed['inputs'], processed['data_samples']


def main() -> None:
    args = parse_args()
    if args.max_instances_per_image < 0:
        raise ValueError('--max-instances-per-image cannot be negative')
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
        raise ValueError('No models selected')
    for spec in specs:
        if str(spec.get('kind', 'detector')) != 'detector':
            raise ValueError(
                f'{spec.get("id")}: fixed-GT category XGradCAM requires a '
                'complete RUOD-trained detector, not a bare pretrained backbone')
    selected_layers = parse_csv(args.layers)
    if not selected_layers:
        raise ValueError('At least one layer is required')
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not (
            args.resume or args.overwrite):
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    image_by_id = {int(row['image_id']): row for row in manifest}
    instances_by_image, all_valid_instances_by_image, categories = load_instances(
        annotation_path,
        list(image_by_id),
        args.minimum_box_area,
        args.exclude_crowd,
        args.max_instances_per_image,
        args.instance_order,
    )
    total_instances = sum(len(rows) for rows in instances_by_image.values())
    if total_instances == 0:
        raise RuntimeError('No valid GT instances selected')

    from mmdet.utils import get_test_pipeline_cfg

    reports = {}
    for model_position, spec in enumerate(specs, 1):
        loaded = load_model(spec, args.device)
        model = loaded.model
        model_id = loaded.model_id
        reports[model_id] = loaded.load_report
        try:
            if not hasattr(model, 'roi_head') or not hasattr(model.roi_head, '_bbox_forward'):
                raise TypeError(f'{model_id}: model has no compatible ROI bbox head')
            num_stages = int(getattr(model.roi_head, 'num_stages', 1))
            stage = args.cascade_stage if args.cascade_stage >= 0 else num_stages - 1
            if not 0 <= stage < num_stages:
                raise ValueError(f'{model_id}: invalid Cascade stage {stage}/{num_stages}')
            unknown_layers = set(selected_layers) - set(loaded.layers)
            if unknown_layers:
                raise ValueError(f'{model_id}: unavailable layers {sorted(unknown_layers)}')
            label_map, category_names, mapping_method = build_label_mapping(
                categories, model)
            captures: Dict[str, torch.Tensor] = {}
            handles = []
            for layer_id in selected_layers:
                def hook(_module, _inputs, output, key=layer_id):
                    captures[key] = tensor_from_output(output)
                handles.append(loaded.layers[layer_id].register_forward_hook(hook))
            pipeline = Compose(get_test_pipeline_cfg(model.cfg))
            completed = 0
            try:
                for image_position, row in enumerate(manifest, 1):
                    image_id = int(row['image_id'])
                    instances = instances_by_image.get(image_id, [])
                    if not instances:
                        continue
                    image_path = existing_file(row['image_path'])
                    all_complete = True
                    for instance in instances:
                        metadata_path = instance_metadata_path(
                            out_dir, model_id, image_id, instance['annotation_id'])
                        layer_complete = all(
                            raw_cam_path(
                                out_dir, model_id, image_id,
                                instance['annotation_id'], layer_id).is_file()
                            for layer_id in selected_layers
                        )
                        if not (metadata_path.is_file() and layer_complete):
                            all_complete = False
                            break
                    if args.resume and all_complete and not args.overwrite:
                        completed += len(instances)
                        print(
                            f'REUSE [{model_id}] image {image_position}/{len(manifest)} '
                            f'{row["file_name"]}: {len(instances)} instances', flush=True)
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
                    metadata = data_samples[0].metainfo
                    ori_shape = shape_pair(
                        metadata.get('ori_shape'),
                        (int(row['height']), int(row['width'])))
                    img_shape = shape_pair(metadata.get('img_shape'), ori_shape)
                    pad_shape = shape_pair(
                        metadata.get('pad_shape', metadata.get('batch_input_shape')),
                        img_shape)
                    scale_x, scale_y = scale_factor_xy(metadata)
                    all_boxes = [
                        item['bbox_xyxy']
                        for item in all_valid_instances_by_image.get(image_id, instances)
                    ]

                    for instance_position, instance in enumerate(instances, 1):
                        annotation_id = int(instance['annotation_id'])
                        category_id = int(instance['category_id'])
                        if category_id not in label_map:
                            raise KeyError(f'Unknown category ID {category_id}')
                        class_label = int(label_map[category_id])
                        original_box = clip_box(
                            instance['bbox_xyxy'], ori_shape[1], ori_shape[0])
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
                        if class_label >= cls_score.shape[1] - 1:
                            raise IndexError(
                                f'{model_id}: class label {class_label} is invalid for '
                                f'cls_score shape {tuple(cls_score.shape)}')
                        target_logit = cls_score[0, class_label]
                        gradients = torch.autograd.grad(
                            target_logit,
                            activations,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=False,
                        )
                        probability = float(torch.softmax(cls_score, dim=1)[0, class_label])
                        layer_records = {}
                        for layer_id, activation, gradient in zip(
                                selected_layers, activations, gradients):
                            cam = xgradcam_raw(activation, gradient)
                            cam_valid, valid_shape = crop_valid_map(
                                cam, img_shape, pad_shape)
                            destination = raw_cam_path(
                                out_dir, model_id, image_id, annotation_id, layer_id)
                            atomic_save_npz(
                                destination,
                                cam=cam_valid.detach().float().cpu().numpy().astype(np.float32),
                                image_id=np.int64(image_id),
                                annotation_id=np.int64(annotation_id),
                                category_id=np.int64(category_id),
                                class_label=np.int64(class_label),
                                target_logit=np.float32(target_logit.detach().cpu()),
                                target_probability=np.float32(probability),
                                bbox_xyxy_original=np.asarray(original_box, dtype=np.float32),
                                bbox_xyxy_network_input=np.asarray(network_box, dtype=np.float32),
                                ori_shape=np.asarray(ori_shape, dtype=np.int32),
                                img_shape=np.asarray(img_shape, dtype=np.int32),
                                pad_shape=np.asarray(pad_shape, dtype=np.int32),
                                valid_feature_shape=np.asarray(valid_shape, dtype=np.int32),
                                scale_factor=np.asarray([scale_x, scale_y], dtype=np.float32),
                            )
                            layer_records[layer_id] = str(destination)
                        instance_record = {
                            'model': model_id,
                            'model_position': model_position,
                            'image_id': image_id,
                            'annotation_id': annotation_id,
                            'category_id': category_id,
                            'category_name': category_names[category_id],
                            'class_label': class_label,
                            'class_mapping_method': mapping_method,
                            'image_path': str(image_path),
                            'file_name': str(row['file_name']),
                            'bbox_xyxy_original': original_box,
                            'bbox_xyxy_network_input': network_box,
                            'all_gt_boxes_xyxy_original': all_boxes,
                            'area': float(instance['area']),
                            'cascade_stage_zero_based': stage,
                            'target_logit': float(target_logit.detach().cpu()),
                            'target_probability': probability,
                            'ori_shape': list(ori_shape),
                            'img_shape': list(img_shape),
                            'pad_shape': list(pad_shape),
                            'scale_factor_xy': [scale_x, scale_y],
                            'layers': layer_records,
                        }
                        atomic_write_json(
                            instance_metadata_path(
                                out_dir, model_id, image_id, annotation_id),
                            instance_record,
                        )
                        completed += 1
                        print(
                            f'[{model_id}] image {image_position}/{len(manifest)} '
                            f'instance {instance_position}/{len(instances)} '
                            f'ann={annotation_id} class={category_names[category_id]}',
                            flush=True,
                        )
                    del features, batch_inputs, data_samples
                    captures.clear()
                    gc.collect()
                print(
                    f'DONE {model_id}: {completed}/{total_instances} instances',
                    flush=True)
            finally:
                for handle in handles:
                    handle.remove()
                captures.clear()
        finally:
            loaded.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed_rows = []
    for spec in specs:
        model_id = str(spec['id'])
        for path in sorted(
                (out_dir / 'raw_cam' / clean_name(model_id)).glob(
                    'image_*/ann_*/instance.json')):
            completed_rows.append(read_json(path))
    worker_name = clean_name('_'.join(str(spec['id']) for spec in specs))
    atomic_write_json(
        out_dir / 'model_load_reports' / f'{worker_name}.json', reports)
    atomic_write_json(out_dir / 'extraction_summaries' / f'{worker_name}.json', {
        'method': 'fixed-GT category-conditioned XGradCAM',
        'raw_cam_state': 'ReLU applied; no min-max/percentile normalization',
        'manifest': str(manifest_path),
        'annotation_file': str(annotation_path),
        'models_config': str(models_config_path),
        'models': [str(spec['id']) for spec in specs],
        'layers': selected_layers,
        'cascade_stage_requested': args.cascade_stage,
        'selected_images': len(manifest),
        'selected_instances': total_instances,
        'completed_model_instances': len(completed_rows),
        'max_instances_per_image': args.max_instances_per_image,
        'minimum_box_area': args.minimum_box_area,
        'instance_order': args.instance_order,
        'device': args.device,
        'warning': (
            'Raw XGradCAM magnitude is affected by parameter/logit scale. '
            'Use scale-invariant spatial metrics as primary quantitative evidence.'),
    })
    print(f'Raw fixed-GT XGradCAM complete: {out_dir}')


if __name__ == '__main__':
    main()
