#!/usr/bin/env python3
"""Summarize per-band feature norms and shifts relative to clean inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import ensure_empty_or_create, parse_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='')
    parser.add_argument('--clean-variant', default='clean')
    parser.add_argument('--frequency-variants', default='low,mid,high')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def layer_names(root: Path, requested: List[str]) -> List[str]:
    if requested:
        return requested
    return sorted(
        path.stem for path in root.glob('*.npy')
        if not path.name.endswith('.spatial_norm.npy'))


def load_matrix(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f'Invalid feature matrix: {path}, shape={value.shape}')
    return value


def summary(values: np.ndarray) -> Dict[str, float]:
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
    }


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    variants = parse_csv(args.frequency_variants)
    requested_layers = parse_csv(args.layers)
    if not models or not variants:
        raise ValueError('--models and --frequency-variants must not be empty')
    root = Path(args.feature_root).expanduser().resolve()
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    per_sample_rows = []
    summary_rows = []

    for model in models:
        clean_root = root / 'features' / model / args.clean_variant
        layers = layer_names(clean_root, requested_layers)
        if not layers:
            raise ValueError(f'No feature layers found for {model}')
        for layer in layers:
            clean = load_matrix(clean_root / f'{layer}.npy')
            clean_norm = np.linalg.norm(clean, axis=1)
            clean_spatial = np.load(
                clean_root / f'{layer}.spatial_norm.npy', allow_pickle=False).astype(np.float64)
            for variant in variants:
                variant_root = root / 'features' / model / variant
                changed = load_matrix(variant_root / f'{layer}.npy')
                changed_spatial = np.load(
                    variant_root / f'{layer}.spatial_norm.npy', allow_pickle=False).astype(np.float64)
                if changed.shape != clean.shape or changed_spatial.shape != clean_spatial.shape:
                    raise ValueError(
                        f'{model}/{layer}/{variant}: clean and variant shapes differ')
                changed_norm = np.linalg.norm(changed, axis=1)
                norm_ratio = changed_norm / np.maximum(clean_norm, args.eps)
                feature_shift = np.linalg.norm(changed - clean, axis=1) / np.maximum(
                    clean_norm, args.eps)
                spatial_norm_ratio = changed_spatial / np.maximum(clean_spatial, args.eps)
                for index in range(clean.shape[0]):
                    per_sample_rows.append({
                        'sample_index': index,
                        'model': model,
                        'layer': layer,
                        'variant': variant,
                        'clean_pooled_norm': clean_norm[index],
                        'variant_pooled_norm': changed_norm[index],
                        'pooled_norm_ratio': norm_ratio[index],
                        'normalized_feature_shift': feature_shift[index],
                        'clean_spatial_norm': clean_spatial[index],
                        'variant_spatial_norm': changed_spatial[index],
                        'spatial_norm_ratio': spatial_norm_ratio[index],
                    })
                for metric, values in (
                    ('pooled_norm_ratio', norm_ratio),
                    ('normalized_feature_shift', feature_shift),
                    ('spatial_norm_ratio', spatial_norm_ratio),
                ):
                    summary_rows.append({
                        'model': model,
                        'layer': layer,
                        'variant': variant,
                        'metric': metric,
                        'samples': len(values),
                        **summary(values),
                    })

    per_fields = list(per_sample_rows[0])
    with (out_dir / 'frequency_response_per_sample.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(per_sample_rows)
    summary_fields = list(summary_rows[0])
    with (out_dir / 'frequency_response_summary.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(out_dir / 'frequency_response_metadata.json', {
        'feature_root': str(root),
        'models': models,
        'layers': requested_layers or 'discovered per model',
        'clean_variant': args.clean_variant,
        'frequency_variants': variants,
        'definitions': {
            'pooled_norm_ratio': '||f_band||_2 / max(||f_clean||_2, eps)',
            'normalized_feature_shift': (
                '||f_band - f_clean||_2 / max(||f_clean||_2, eps)'),
            'spatial_norm_ratio': (
                '||F_band||_F / max(||F_clean||_F, eps)'),
        },
    })
    print(f'Frequency response results: {out_dir}')


if __name__ == '__main__':
    main()
