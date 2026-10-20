#!/usr/bin/env python3
"""Compute linear CKA from aligned, pooled backbone features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import ensure_empty_or_create, parse_csv, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--model-a', required=True)
    parser.add_argument('--model-b', required=True)
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--layers-a', default='')
    parser.add_argument('--layers-b', default='')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_ids(feature_root: Path, model: str, variant: str) -> np.ndarray:
    metadata_path = feature_root / 'metadata' / model / f'{variant}.json'
    rows = read_json(metadata_path)
    ids = np.asarray([int(row['image_id']) for row in rows], dtype=np.int64)
    positions = [int(row['sample_index']) for row in rows]
    if not len(ids) or positions != list(range(len(ids))) or len(set(ids.tolist())) != len(ids):
        raise ValueError(f'Invalid sample order or duplicate image IDs: {metadata_path}')
    return ids


def load_features(root: Path, requested: str, count: int) -> dict[str, np.ndarray]:
    layers = parse_csv(requested) or sorted(
        path.stem for path in root.glob('*.npy')
        if not path.name.endswith('.spatial_norm.npy'))
    if not layers:
        raise ValueError(f'No feature layers found in {root}')
    features = {}
    for layer in layers:
        path = root / f'{layer}.npy'
        value = np.load(path, allow_pickle=False)
        if value.ndim != 2 or value.shape[0] != count or not np.isfinite(value).all():
            raise ValueError(f'Invalid feature matrix: {path} shape={value.shape}')
        features[layer] = value.astype(np.float64, copy=False)
    return features


def centered_gram(value: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    centered = value - value.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    norm = float(np.linalg.norm(gram, ord='fro'))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError('Centered feature matrix has zero or invalid norm')
    return gram, norm


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    ids_a = load_ids(feature_root, args.model_a, args.variant)
    ids_b = load_ids(feature_root, args.model_b, args.variant)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError('Model features have different image IDs or sample order')

    root_a = feature_root / 'features' / args.model_a / args.variant
    root_b = feature_root / 'features' / args.model_b / args.variant
    features_a = load_features(root_a, args.layers_a, len(ids_a))
    features_b = load_features(root_b, args.layers_b, len(ids_b))
    grams_a = {name: centered_gram(value, args.eps) for name, value in features_a.items()}
    grams_b = {name: centered_gram(value, args.eps) for name, value in features_b.items()}
    layers_a, layers_b = list(features_a), list(features_b)
    matrix = np.empty((len(layers_a), len(layers_b)), dtype=np.float64)
    for row, layer_a in enumerate(layers_a):
        ka, norm_a = grams_a[layer_a]
        for col, layer_b in enumerate(layers_b):
            kb, norm_b = grams_b[layer_b]
            matrix[row, col] = np.sum(ka * kb) / (norm_a * norm_b)

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    np.save(out_dir / 'cka_matrix.npy', matrix.astype(np.float32), allow_pickle=False)
    with (out_dir / 'cka_matrix.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['model_a', 'layer_a'] + layers_b)
        for layer, values in zip(layers_a, matrix):
            writer.writerow([args.model_a, layer] + [f'{value:.8f}' for value in values])
    write_json(out_dir / 'cka_metadata.json', {
        'feature_root': str(feature_root),
        'model_a': args.model_a,
        'model_b': args.model_b,
        'variant': args.variant,
        'layers_a': layers_a,
        'layers_b': layers_b,
        'samples': len(ids_a),
        'method': 'linear CKA on column-centered pooled features',
        'implementation': 'sample Gram matrices; NumPy linear algebra',
        'sample_ids_verified': True,
    })
    print(f'CKA results: {out_dir}')


if __name__ == '__main__':
    main()
