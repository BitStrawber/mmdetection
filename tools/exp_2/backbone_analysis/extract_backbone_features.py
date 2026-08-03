#!/usr/bin/env python3
"""Extract aligned multi-layer features from configurable backbones/detectors."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from .common import (
    ensure_empty_or_create,
    existing_file,
    finite_summary,
    parse_csv,
    read_json,
    read_jsonl,
    sample_ids,
    validate_sample_order,
    write_json,
)
from .model_adapter import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--models-config', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', default='', help='Optional comma-separated model IDs')
    parser.add_argument(
        '--variants', default='clean',
        help='Comma-separated manifest variants; clean uses image_path when absent')
    parser.add_argument('--layers', default='', help='Optional comma-separated layer IDs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--pooling', choices=('avg', 'max', 'avgmax'), default='avg')
    parser.add_argument(
        '--save-spatial', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--spatial-samples', type=int, default=0,
        help='Maximum spatial samples per model/variant; 0 means all')
    parser.add_argument(
        '--spatial-dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        tensors = [value for value in output if torch.is_tensor(value)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f'Hook output is not one tensor: {type(output).__name__}')


def shape_pair(value: Any, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return fallback
    return int(value[0]), int(value[1])


def crop_valid_feature(
    feature: torch.Tensor,
    img_shape: Tuple[int, int],
    pad_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    value = feature.detach().float().cpu()
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f'Expected batch size 1, got {tuple(value.shape)}')
        value = value[0]
    if value.ndim == 3:
        feature_h, feature_w = int(value.shape[-2]), int(value.shape[-1])
        img_h, img_w = img_shape
        pad_h, pad_w = pad_shape
        valid_h = min(feature_h, max(
            1, int(math.ceil(img_h * feature_h / float(max(pad_h, 1))))))
        valid_w = min(feature_w, max(
            1, int(math.ceil(img_w * feature_w / float(max(pad_w, 1))))))
        return value[:, :valid_h, :valid_w], (valid_h, valid_w)
    if value.ndim == 2:
        return value, (int(value.shape[0]), 1)
    raise ValueError(f'Unsupported feature shape: {tuple(value.shape)}')


def pool_feature(value: torch.Tensor, method: str) -> torch.Tensor:
    if value.ndim == 2:
        average = value.mean(dim=0)
        maximum = value.max(dim=0).values
    elif value.ndim == 3:
        average = value.mean(dim=(-2, -1))
        maximum = value.amax(dim=(-2, -1))
    else:
        raise ValueError(f'Cannot pool feature shape {tuple(value.shape)}')
    if method == 'avg':
        return average
    if method == 'max':
        return maximum
    return torch.cat([average, maximum], dim=0)


def variant_path(row: Mapping[str, Any], variant: str) -> Path:
    if variant == 'clean' and 'variants' not in row:
        return existing_file(row['image_path'])
    variants = row.get('variants', {})
    if variant not in variants:
        if variant == 'clean':
            return existing_file(row['image_path'])
        raise KeyError(
            f'Sample {row["sample_index"]} has no frequency variant {variant}')
    return existing_file(variants[variant]['image_path'])


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    validate_sample_order(rows)
    models_config = read_json(args.models_config)
    specs = models_config.get('models', [])
    if not isinstance(specs, list) or not specs:
        raise ValueError('models-config must contain a non-empty models list')
    selected_models = set(parse_csv(args.models))
    if selected_models:
        specs = [spec for spec in specs if str(spec['id']) in selected_models]
        missing = selected_models - {str(spec['id']) for spec in specs}
        if missing:
            raise ValueError(f'Unknown model IDs: {sorted(missing)}')
    variants = parse_csv(args.variants)
    if not variants:
        raise ValueError('At least one --variants value is required')
    selected_layers = set(parse_csv(args.layers))
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    np.save(out_dir / 'sample_ids.npy', sample_ids(rows), allow_pickle=False)
    write_json(out_dir / 'source.json', {
        'manifest': str(existing_file(args.manifest)),
        'models_config': str(existing_file(args.models_config)),
        'models': [str(spec['id']) for spec in specs],
        'variants': variants,
        'pooling': args.pooling,
        'save_spatial': args.save_spatial,
        'spatial_samples': args.spatial_samples,
    })
    qa_rows = []
    extraction_reports = {}

    from mmdet.apis import inference_detector

    for spec in specs:
        loaded = load_model(spec, args.device)
        extraction_reports[loaded.model_id] = loaded.load_report
        layer_ids = list(loaded.layers)
        if selected_layers:
            unknown = selected_layers - set(layer_ids)
            if unknown:
                loaded.close()
                raise ValueError(
                    f'{loaded.model_id}: requested unavailable layers {sorted(unknown)}')
            layer_ids = [layer for layer in layer_ids if layer in selected_layers]
        captures: Dict[str, torch.Tensor] = {}
        handles = []
        for layer_id in layer_ids:
            def hook(_module, _inputs, output, key=layer_id):
                captures[key] = tensor_from_output(output)
            handles.append(loaded.layers[layer_id].register_forward_hook(hook))
        try:
            for variant in variants:
                pooled_by_layer: Dict[str, List[np.ndarray]] = {
                    layer: [] for layer in layer_ids
                }
                norm_by_layer: Dict[str, List[float]] = {
                    layer: [] for layer in layer_ids
                }
                metadata_rows = []
                for position, row in enumerate(rows):
                    captures.clear()
                    image_path = variant_path(row, variant)
                    prediction = inference_detector(loaded.model, str(image_path))
                    missing_layers = set(layer_ids) - set(captures)
                    if missing_layers:
                        raise RuntimeError(
                            f'{loaded.model_id}: hooks captured no outputs for '
                            f'{sorted(missing_layers)}')
                    metadata = prediction.metainfo
                    img_shape = shape_pair(metadata.get('img_shape'), (1, 1))
                    pad_shape = shape_pair(
                        metadata.get('pad_shape', metadata.get('batch_input_shape')),
                        img_shape)
                    layer_shapes = {}
                    for layer_id in layer_ids:
                        valid, valid_shape = crop_valid_feature(
                            captures[layer_id], img_shape, pad_shape)
                        pooled = pool_feature(valid, args.pooling)
                        pooled_by_layer[layer_id].append(
                            pooled.numpy().astype(np.float32, copy=False))
                        norm_by_layer[layer_id].append(float(torch.linalg.vector_norm(valid)))
                        layer_shapes[layer_id] = {
                            'full': list(captures[layer_id].shape),
                            'valid': list(valid.shape),
                        }
                        save_this_spatial = args.save_spatial and (
                            args.spatial_samples <= 0 or position < args.spatial_samples)
                        if save_this_spatial:
                            spatial_path = (
                                out_dir / 'spatial' / loaded.model_id / variant /
                                f'{position:05d}_{int(row["image_id"])}' /
                                f'{layer_id}.npz')
                            spatial_path.parent.mkdir(parents=True, exist_ok=True)
                            dtype = np.float16 if args.spatial_dtype == 'float16' else np.float32
                            np.savez_compressed(
                                spatial_path,
                                feature=valid.numpy().astype(dtype),
                                img_shape=np.asarray(img_shape, dtype=np.int32),
                                pad_shape=np.asarray(pad_shape, dtype=np.int32),
                                sample_index=np.int64(position),
                                image_id=np.int64(row['image_id']))
                    metadata_rows.append({
                        'sample_index': position,
                        'image_id': int(row['image_id']),
                        'image_path': str(image_path),
                        'variant': variant,
                        'img_shape': list(img_shape),
                        'pad_shape': list(pad_shape),
                        'ori_shape': list(shape_pair(
                            metadata.get('ori_shape'),
                            (int(row['height']), int(row['width'])))),
                        'scale_factor': [float(value) for value in np.asarray(
                            metadata.get('scale_factor', [1.0, 1.0])).reshape(-1)],
                        'layer_shapes': layer_shapes,
                    })
                    print(
                        f'[{loaded.model_id}:{variant}] '
                        f'{position + 1}/{len(rows)} {row["file_name"]}', flush=True)

                destination = out_dir / 'features' / loaded.model_id / variant
                destination.mkdir(parents=True, exist_ok=True)
                for layer_id in layer_ids:
                    matrix = np.stack(pooled_by_layer[layer_id]).astype(np.float32)
                    norms = np.asarray(norm_by_layer[layer_id], dtype=np.float32)
                    np.save(destination / f'{layer_id}.npy', matrix, allow_pickle=False)
                    np.save(destination / f'{layer_id}.spatial_norm.npy', norms,
                            allow_pickle=False)
                    summary = finite_summary(matrix)
                    qa_rows.append({
                        'model': loaded.model_id,
                        'variant': variant,
                        'layer': layer_id,
                        'samples': matrix.shape[0],
                        'dimensions': matrix.shape[1],
                        **summary,
                        'zero_norm_rows': int(
                            (np.linalg.norm(matrix, axis=1) == 0).sum()),
                    })
                write_json(
                    out_dir / 'metadata' / loaded.model_id / f'{variant}.json',
                    metadata_rows)
        finally:
            for handle in handles:
                handle.remove()
            captures.clear()
            loaded.close()
            gc.collect()

    write_json(out_dir / 'model_load_reports.json', extraction_reports)
    qa_path = out_dir / 'qa' / 'feature_summary.tsv'
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'model', 'variant', 'layer', 'samples', 'dimensions', 'shape', 'dtype',
        'nan_count', 'inf_count', 'finite_count', 'min', 'max', 'mean',
        'zero_norm_rows',
    ]
    with qa_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(qa_rows)
    print(f'Feature extraction completed: {out_dir}')
    print(f'QA summary: {qa_path}')


if __name__ == '__main__':
    main()
