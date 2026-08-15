#!/usr/bin/env python3
"""Compute only same-semantic-layer linear CKA against one reference model.

This deliberately excludes layer-crossed comparisons and, by default, the
reference model's self-comparison.  CKA(X, X) is mathematically one and is not
an experimental result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.exp_2.backbone_analysis.common import ensure_empty_or_create, parse_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--reference-model', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--include-reference', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_feature(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f'Invalid pooled feature matrix: {path}, shape={value.shape}')
    return value


def linear_cka(first: np.ndarray, second: np.ndarray, eps: float) -> float:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f'CKA sample mismatch: {first.shape} vs {second.shape}')
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(first.T @ second, ord='fro') ** 2
    denominator = (
        np.linalg.norm(first.T @ first, ord='fro') *
        np.linalg.norm(second.T @ second, ord='fro'))
    return float(numerator / max(float(denominator), eps))


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    if not models or not layers:
        raise ValueError('--models and --layers must not be empty')
    if len(models) != len(set(models)):
        raise ValueError(f'Duplicate model IDs: {models}')

    root = Path(args.feature_root).expanduser().resolve()
    output = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    references: Dict[str, np.ndarray] = {}
    for layer in layers:
        references[layer] = load_feature(
            root / 'features' / args.reference_model / args.variant / f'{layer}.npy')

    rows = []
    comparison_models = [
        model for model in models
        if args.include_reference or model != args.reference_model]
    if not comparison_models:
        raise ValueError('No non-self CKA rows remain')
    for model in comparison_models:
        values = []
        for layer in layers:
            feature_path = root / 'features' / model / args.variant / f'{layer}.npy'
            value = load_feature(feature_path)
            cka = linear_cka(value, references[layer], args.eps)
            values.append(cka)
            rows.append({
                'model': model,
                'reference_model': args.reference_model,
                'layer': layer,
                'variant': args.variant,
                'linear_cka': cka,
                'self_comparison': model == args.reference_model,
                'feature_path': str(feature_path),
                'reference_feature_path': str(
                    root / 'features' / args.reference_model / args.variant / f'{layer}.npy'),
            })

    with (output / 'same_layer_cka.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    matrix = np.asarray([
        [next(item['linear_cka'] for item in rows if item['model'] == model and item['layer'] == layer)
         for layer in layers]
        for model in comparison_models
    ], dtype=np.float32)
    np.save(output / 'same_layer_cka.npy', matrix, allow_pickle=False)
    write_json(output / 'metadata.json', {
        'feature_root': str(root), 'models': models,
        'reference_model': args.reference_model, 'layers': layers,
        'variant': args.variant, 'include_reference': args.include_reference,
        'method': 'linear CKA on column-centered pooled features; same semantic layer only',
        'self_comparison_policy': (
            'excluded by default because linear CKA(X, X) = 1 by definition'),
    })

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(
        figsize=(1.55 * len(layers) + 2.8, 0.58 * len(comparison_models) + 2.3))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
    axis.set_xticks(range(len(layers)), layers)
    axis.set_yticks(range(len(comparison_models)), comparison_models)
    axis.set_xlabel(f'Same layer of {args.reference_model}')
    axis.set_ylabel('Comparison model')
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            axis.text(column_index, row_index, f'{value:.3f}', ha='center', va='center',
                      color='white' if value < 0.55 else 'black', fontsize=9)
    figure.colorbar(image, ax=axis, label='Linear CKA')
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(output / f'same_layer_cka.{suffix}', dpi=240, bbox_inches='tight')
    plt.close(figure)
    print(f'Same-layer CKA: {output}')


if __name__ == '__main__':
    main()
