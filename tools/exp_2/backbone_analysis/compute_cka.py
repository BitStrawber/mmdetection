#!/usr/bin/env python3
"""Compute a cross-layer linear CKA similarity matrix from saved features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .common import ensure_empty_or_create, parse_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument(
        '--model-a', '--y-model', dest='model_a', default='',
        help='Single model shown on the heatmap y-axis (matrix rows).')
    parser.add_argument(
        '--y-models', default='',
        help='Comma-separated y-axis models. Overrides --y-model and stacks '
             'MODEL/LAYER rows in one reference heatmap.')
    parser.add_argument(
        '--model-b', '--x-model', dest='model_b', required=True,
        help='Model shown on the heatmap x-axis (matrix columns).')
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--layers-a', default='', help='Default: discover all .npy layers')
    parser.add_argument('--layers-b', default='')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def discover_layers(root: Path, requested: str) -> List[str]:
    values = parse_csv(requested)
    if values:
        return values
    return sorted(
        path.stem for path in root.glob('*.npy')
        if not path.name.endswith('.spatial_norm.npy'))


def load_feature(root: Path, layer: str) -> np.ndarray:
    path = root / f'{layer}.npy'
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    if value.ndim != 2:
        raise ValueError(f'{path}: expected N x D, got {value.shape}')
    if not np.isfinite(value).all():
        raise ValueError(f'{path}: contains NaN or Inf')
    return value


def linear_cka(first: np.ndarray, second: np.ndarray, eps: float) -> float:
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f'CKA sample count mismatch: {first.shape[0]} vs {second.shape[0]}')
    x = first - first.mean(axis=0, keepdims=True)
    y = second - second.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord='fro') ** 2
    first_norm = np.linalg.norm(x.T @ x, ord='fro')
    second_norm = np.linalg.norm(y.T @ y, ord='fro')
    denominator = first_norm * second_norm
    return float(cross / max(denominator, eps))


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    y_models = parse_csv(args.y_models)
    if not y_models:
        y_models = [args.model_a] if args.model_a else []
    if not y_models:
        raise ValueError('Set --y-model or --y-models')
    if len(set(y_models)) != len(y_models):
        raise ValueError(f'Duplicate y-axis models: {y_models}')
    root_b = feature_root / 'features' / args.model_b / args.variant
    if not root_b.is_dir():
        raise FileNotFoundError(f'Missing x-axis feature root: {root_b}')
    layers_b = discover_layers(root_b, args.layers_b)
    if not layers_b:
        raise ValueError('No layers were selected')
    features_b = {layer: load_feature(root_b, layer) for layer in layers_b}
    y_rows: List[Tuple[str, str, np.ndarray]] = []
    requested_y_layers = parse_csv(args.layers_a)
    for model in y_models:
        root_a = feature_root / 'features' / model / args.variant
        if not root_a.is_dir():
            raise FileNotFoundError(f'Missing y-axis feature root: {root_a}')
        layers_a = requested_y_layers or discover_layers(root_a, '')
        if not layers_a:
            raise ValueError(f'No layers were selected for {model}')
        for layer in layers_a:
            y_rows.append((model, layer, load_feature(root_a, layer)))
    sample_counts = {value.shape[0] for value in features_b.values()}
    sample_counts.update(value.shape[0] for _, _, value in y_rows)
    if len(sample_counts) != 1:
        raise ValueError(f'Feature files have inconsistent sample counts: {sample_counts}')
    matrix = np.empty((len(y_rows), len(layers_b)), dtype=np.float64)
    for row, (_, _, feature_a) in enumerate(y_rows):
        for column, layer_b in enumerate(layers_b):
            matrix[row, column] = linear_cka(
                feature_a, features_b[layer_b], args.eps)

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    np.save(out_dir / 'cka_matrix.npy', matrix.astype(np.float32), allow_pickle=False)
    with (out_dir / 'cka_matrix.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['y_model', 'y_layer'] + layers_b)
        for (model, layer, _), values in zip(y_rows, matrix):
            writer.writerow(
                [model, layer] + [f'{value:.8f}' for value in values])
    write_json(out_dir / 'cka_metadata.json', {
        'feature_root': str(feature_root),
        'model_a': y_models[0] if len(y_models) == 1 else None,
        'model_b': args.model_b,
        'y_axis_model': y_models[0] if len(y_models) == 1 else None,
        'y_axis_models': y_models,
        'y_axis_rows': [
            {'model': model, 'layer': layer}
            for model, layer, _ in y_rows
        ],
        'x_axis_model': args.model_b,
        'variant': args.variant,
        'layers_a': requested_y_layers or sorted({layer for _, layer, _ in y_rows}),
        'layers_b': layers_b,
        'samples': sample_counts.pop(),
        'method': 'linear CKA on column-centered pooled features',
    })
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(
            figsize=(1.4 * len(layers_b) + 2, 0.55 * len(y_rows) + 2.5))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap='viridis')
        axis.set_xticks(range(len(layers_b)))
        axis.set_xticklabels(layers_b, rotation=45, ha='right')
        axis.set_yticks(range(len(y_rows)))
        if len(y_models) == 1:
            y_labels = [layer for _, layer, _ in y_rows]
        else:
            y_labels = [f'{model} / {layer}' for model, layer, _ in y_rows]
        axis.set_yticklabels(y_labels)
        axis.set_xlabel(f'{args.model_b} layers')
        axis.set_ylabel(
            f'{y_models[0]} layers' if len(y_models) == 1
            else 'Comparison model layers')
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f'{matrix[row, column]:.3f}',
                          ha='center', va='center', color='white' if matrix[row, column] < .55 else 'black')
        figure.colorbar(image, ax=axis, label='Linear CKA')
        figure.tight_layout()
        figure.savefig(out_dir / 'cka_matrix.png', dpi=220)
        plt.close(figure)
    except ImportError:
        print('matplotlib is unavailable; skipped cka_matrix.png')
    print(f'CKA results: {out_dir}')


if __name__ == '__main__':
    main()
