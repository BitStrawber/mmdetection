#!/usr/bin/env python3
"""Compute a cross-layer linear CKA similarity matrix from saved features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from .common import ensure_empty_or_create, parse_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--model-a', required=True)
    parser.add_argument('--model-b', required=True)
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
    root_a = feature_root / 'features' / args.model_a / args.variant
    root_b = feature_root / 'features' / args.model_b / args.variant
    if not root_a.is_dir() or not root_b.is_dir():
        raise FileNotFoundError(f'Missing feature roots: {root_a}, {root_b}')
    layers_a = discover_layers(root_a, args.layers_a)
    layers_b = discover_layers(root_b, args.layers_b)
    if not layers_a or not layers_b:
        raise ValueError('No layers were selected')
    features_a = {layer: load_feature(root_a, layer) for layer in layers_a}
    features_b = {layer: load_feature(root_b, layer) for layer in layers_b}
    sample_counts = {
        value.shape[0] for value in list(features_a.values()) + list(features_b.values())
    }
    if len(sample_counts) != 1:
        raise ValueError(f'Feature files have inconsistent sample counts: {sample_counts}')
    matrix = np.empty((len(layers_a), len(layers_b)), dtype=np.float64)
    for row, layer_a in enumerate(layers_a):
        for column, layer_b in enumerate(layers_b):
            matrix[row, column] = linear_cka(
                features_a[layer_a], features_b[layer_b], args.eps)

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    np.save(out_dir / 'cka_matrix.npy', matrix.astype(np.float32), allow_pickle=False)
    with (out_dir / 'cka_matrix.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['layer'] + layers_b)
        for layer, values in zip(layers_a, matrix):
            writer.writerow([layer] + [f'{value:.8f}' for value in values])
    write_json(out_dir / 'cka_metadata.json', {
        'feature_root': str(feature_root),
        'model_a': args.model_a,
        'model_b': args.model_b,
        'variant': args.variant,
        'layers_a': layers_a,
        'layers_b': layers_b,
        'samples': sample_counts.pop(),
        'method': 'linear CKA on column-centered pooled features',
    })
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(1.4 * len(layers_b) + 2, 1.2 * len(layers_a) + 2))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap='viridis')
        axis.set_xticks(range(len(layers_b)))
        axis.set_xticklabels(layers_b, rotation=45, ha='right')
        axis.set_yticks(range(len(layers_a)))
        axis.set_yticklabels(layers_a)
        axis.set_xlabel(args.model_b)
        axis.set_ylabel(args.model_a)
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
