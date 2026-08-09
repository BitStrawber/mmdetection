#!/usr/bin/env python3
"""Measure feature sensitivity to phase-averaged 2D Fourier perturbations."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .common import ensure_empty_or_create, parse_csv, read_json, read_jsonl, write_json
from .extract_backbone_features import (
    crop_valid_feature, pool_feature, shape_pair, tensor_from_output)
from .model_adapter import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--models-config', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--filter-config', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--grid-size', type=int, default=15)
    parser.add_argument('--maximum-axis-frequency', type=float, default=0.5)
    parser.add_argument('--amplitude', type=float, default=8 / 255)
    parser.add_argument('--phases', default='0,1.5707963267948966')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def capture_features(
    model: object, layers: Mapping[str, object], captures: Dict[str, torch.Tensor],
    image: np.ndarray,
) -> Mapping[str, np.ndarray]:
    from mmdet.apis import inference_detector

    captures.clear()
    prediction = inference_detector(model, image[:, :, ::-1].copy())
    metadata = prediction.metainfo
    img_shape = shape_pair(metadata.get('img_shape'), (1, 1))
    pad_shape = shape_pair(
        metadata.get('pad_shape', metadata.get('batch_input_shape')), img_shape)
    output = {}
    for layer in layers:
        valid, _ = crop_valid_feature(captures[layer], img_shape, pad_shape)
        output[layer] = pool_feature(valid, 'avg').numpy().astype(np.float64)
    return output


def perturbation(
    height: int, width: int, fx: float, fy: float, phase: float,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    return np.cos(2 * np.pi * (fx * xx + fy * yy) + phase).astype(np.float32)


def render_heatmaps(
    matrices: Mapping[Tuple[str, str], np.ndarray], frequencies: np.ndarray,
    cutoffs: Sequence[float], out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for (model, layer), matrix in matrices.items():
        figure, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        limit = float(max(abs(frequencies[0]), abs(frequencies[-1])))
        image = axis.imshow(
            matrix, origin='lower', cmap='cividis',
            extent=(-limit, limit, -limit, limit), aspect='equal')
        for cutoff in cutoffs:
            axis.add_patch(Circle(
                (0, 0), cutoff, fill=False, color='white', linestyle='--',
                linewidth=1.0))
        axis.set_xlabel('$f_x$ (cycles/pixel)')
        axis.set_ylabel('$f_y$ (cycles/pixel)')
        axis.set_title(f'Fourier basis sensitivity: {model} / {layer}')
        figure.colorbar(image, ax=axis, label='Median normalized feature shift')
        stem = out_dir / f'fourier_sensitivity_{model}_{layer}'
        for suffix in ('png', 'pdf'):
            figure.savefig(stem.with_suffix(f'.{suffix}'), dpi=240)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.grid_size < 3 or args.grid_size % 2 == 0:
        raise ValueError('--samples must be positive and --grid-size must be odd >= 3')
    if not 0 < args.maximum_axis_frequency <= 0.5:
        raise ValueError('--maximum-axis-frequency must satisfy 0 < value <= 0.5')
    if args.amplitude <= 0:
        raise ValueError('--amplitude must be positive')
    rows = read_jsonl(args.manifest)[:args.samples]
    config = read_json(args.models_config)
    selected = set(parse_csv(args.models))
    specs = config['models']
    if selected:
        specs = [spec for spec in specs if str(spec['id']) in selected]
    layers = parse_csv(args.layers)
    phases = [float(value) for value in parse_csv(args.phases)]
    frequencies = np.linspace(
        -args.maximum_axis_frequency, args.maximum_axis_frequency,
        args.grid_size, dtype=np.float64)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    per_sample_root = out_dir / 'per_sample'
    per_sample_root.mkdir(parents=True, exist_ok=True)
    matrices: Dict[Tuple[str, str], np.ndarray] = {}
    rows_out = []

    for spec in specs:
        loaded = load_model(spec, args.device)
        selected_layers = {layer: loaded.layers[layer] for layer in layers}
        captures: Dict[str, torch.Tensor] = {}
        handles = []
        for layer, module in selected_layers.items():
            def hook(_module, _inputs, output, key=layer):
                captures[key] = tensor_from_output(output)
            handles.append(module.register_forward_hook(hook))
        sample_values = {
            layer: np.zeros((len(rows), args.grid_size, args.grid_size), dtype=np.float32)
            for layer in layers
        }
        try:
            for sample_index, row in enumerate(rows):
                with Image.open(row['variants']['clean']['image_path']) as opened:
                    clean = np.asarray(opened.convert('RGB'), dtype=np.uint8)
                baseline = capture_features(
                    loaded.model, selected_layers, captures, clean)
                for y_index, fy in enumerate(frequencies):
                    for x_index, fx in enumerate(frequencies):
                        if abs(fx) < 1e-15 and abs(fy) < 1e-15:
                            continue
                        phase_shifts = {layer: [] for layer in layers}
                        for phase in phases:
                            wave = perturbation(
                                clean.shape[0], clean.shape[1], fx, fy, phase)
                            changed = np.clip(
                                clean.astype(np.float32) / 255.0 +
                                args.amplitude * wave[:, :, None], 0.0, 1.0)
                            changed_u8 = np.rint(changed * 255).astype(np.uint8)
                            features = capture_features(
                                loaded.model, selected_layers, captures, changed_u8)
                            for layer in layers:
                                shift = np.linalg.norm(features[layer] - baseline[layer])
                                shift /= max(np.linalg.norm(baseline[layer]), 1e-12)
                                phase_shifts[layer].append(float(shift))
                        for layer in layers:
                            sample_values[layer][sample_index, y_index, x_index] = (
                                float(np.mean(phase_shifts[layer])))
                print(
                    f'[{loaded.model_id} {sample_index + 1}/{len(rows)}] '
                    f'{row["file_name"]}', flush=True)
            for layer in layers:
                np.savez_compressed(
                    per_sample_root / f'{loaded.model_id}_{layer}.npz',
                    values=sample_values[layer], frequencies=frequencies)
                median = np.median(sample_values[layer], axis=0)
                matrices[(loaded.model_id, layer)] = median
                np.save(out_dir / f'{loaded.model_id}_{layer}.npy', median,
                        allow_pickle=False)
                for y_index, fy in enumerate(frequencies):
                    for x_index, fx in enumerate(frequencies):
                        values = sample_values[layer][:, y_index, x_index]
                        rows_out.append({
                            'model': loaded.model_id, 'layer': layer,
                            'fx_cpp': float(fx), 'fy_cpp': float(fy),
                            'radius_cpp': float(math.hypot(fx, fy)),
                            'mean_shift': float(values.mean()),
                            'median_shift': float(np.median(values)),
                            'p05': float(np.percentile(values, 5)),
                            'p95': float(np.percentile(values, 95)),
                        })
        finally:
            for handle in handles:
                handle.remove()
            loaded.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cutoffs = []
    filter_policy = None
    if args.filter_config:
        filter_config = read_json(args.filter_config)
        filter_policy = filter_config.get('band_policy')
        cutoffs = [
            float(band['high']) for band in filter_config['bands'][:-1]
            if band['high'] != 'max'
        ]
    render_heatmaps(matrices, frequencies, cutoffs, out_dir)
    with (out_dir / 'fourier_basis_sensitivity.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)
    write_json(out_dir / 'fourier_basis_metadata.json', {
        'manifest': str(Path(args.manifest).resolve()),
        'models_config': str(Path(args.models_config).resolve()),
        'models': [str(spec['id']) for spec in specs],
        'layers': layers,
        'samples': len(rows),
        'grid_size': args.grid_size,
        'maximum_axis_frequency_cpp': args.maximum_axis_frequency,
        'amplitude': args.amplitude,
        'phases': phases,
        'aggregation': 'phase mean per sample, then sample median',
        'metric': '||f(x+delta)-f(x)||_2 / ||f(x)||_2',
        'filter_config': args.filter_config or None,
        'filter_policy': filter_policy,
        'overlay_cutoffs_cpp': cutoffs,
        'warning': 'This is feature sensitivity, not task-loss sensitivity.',
    })
    print(f'Fourier basis sensitivity: {out_dir}')


if __name__ == '__main__':
    main()
