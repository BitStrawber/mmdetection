#!/usr/bin/env python3
"""Measure band-pass response and band-stop sensitivity of backbone features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .common import (
    ensure_empty_or_create,
    existing_file,
    parse_csv,
    read_jsonl,
    validate_sample_order,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument(
        '--frequency-manifest', required=True,
        help='frequency_manifest.jsonl used to extract the feature variants')
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


def variant_path(row: Mapping[str, object], variant: str) -> Path:
    variants = row.get('variants')
    if not isinstance(variants, Mapping) or variant not in variants:
        raise KeyError(
            f'Sample {row.get("sample_index")} has no variant {variant}')
    payload = variants[variant]
    if not isinstance(payload, Mapping) or not payload.get('image_path'):
        raise ValueError(
            f'Sample {row.get("sample_index")} has invalid variant {variant}')
    return existing_file(str(payload['image_path']))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        value = np.asarray(opened.convert('RGB'), dtype=np.float64) / 255.0
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError(f'Invalid RGB frequency input: {path}, shape={value.shape}')
    return value


def centered(value: np.ndarray) -> np.ndarray:
    return value - value.mean(axis=(0, 1), keepdims=True)


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def input_metrics(
    rows: Sequence[Mapping[str, object]], clean_variant: str,
    variants: Sequence[str], eps: float,
) -> Mapping[str, Mapping[str, np.ndarray]]:
    result: Dict[str, Dict[str, List[float]]] = {
        variant: {
            'clean_centered_rms': [],
            'variant_centered_rms': [],
            'input_centered_rms_ratio': [],
            'input_relative_shift': [],
        }
        for variant in variants
    }
    for row in rows:
        clean = centered(load_rgb(variant_path(row, clean_variant)))
        clean_rms = rms(clean)
        for variant in variants:
            changed = centered(load_rgb(variant_path(row, variant)))
            if changed.shape != clean.shape:
                raise ValueError(
                    f'Sample {row.get("sample_index")} {variant}: input shapes '
                    f'differ, clean={clean.shape}, variant={changed.shape}')
            changed_rms = rms(changed)
            values = result[variant]
            values['clean_centered_rms'].append(clean_rms)
            values['variant_centered_rms'].append(changed_rms)
            values['input_centered_rms_ratio'].append(
                changed_rms / max(clean_rms, eps))
            values['input_relative_shift'].append(
                rms(changed - clean) / max(clean_rms, eps))
    return {
        variant: {
            metric: np.asarray(values, dtype=np.float64)
            for metric, values in metrics.items()
        }
        for variant, metrics in result.items()
    }


def cosine_similarity(
    first: np.ndarray, second: np.ndarray, eps: float,
) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.clip(numerator / np.maximum(denominator, eps), -1.0, 1.0)


def variant_kind(variant: str) -> Tuple[str, str]:
    if variant.startswith('remove_'):
        return 'band_stop', variant[len('remove_'):]
    return 'band_pass', variant


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    variants = parse_csv(args.frequency_variants)
    requested_layers = parse_csv(args.layers)
    if not models or not variants:
        raise ValueError('--models and --frequency-variants must not be empty')
    root = Path(args.feature_root).expanduser().resolve()
    manifest_path = existing_file(args.frequency_manifest)
    manifest_rows = read_jsonl(manifest_path)
    validate_sample_order(manifest_rows)
    inputs = input_metrics(
        manifest_rows, args.clean_variant, variants, args.eps)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    per_sample_rows = []
    summary_rows = []
    input_summary_rows = []

    for variant in variants:
        kind, band = variant_kind(variant)
        for metric in ('input_centered_rms_ratio', 'input_relative_shift'):
            values = inputs[variant][metric]
            input_summary_rows.append({
                'variant': variant,
                'variant_kind': kind,
                'band': band,
                'metric': metric,
                'samples': len(values),
                **summary(values),
            })

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
                kind, band = variant_kind(variant)
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
                similarity = cosine_similarity(clean, changed, args.eps)
                cosine_distance = 1.0 - similarity
                spatial_norm_ratio = changed_spatial / np.maximum(clean_spatial, args.eps)
                input_values = inputs[variant]
                if len(input_values['input_relative_shift']) != clean.shape[0]:
                    raise ValueError(
                        f'{variant}: manifest and feature sample counts differ')
                response_gain = norm_ratio / np.maximum(
                    input_values['input_centered_rms_ratio'], args.eps)
                shift_gain = feature_shift / np.maximum(
                    input_values['input_relative_shift'], args.eps)
                for index in range(clean.shape[0]):
                    per_sample_rows.append({
                        'sample_index': int(
                            manifest_rows[index]['sample_index']),
                        'image_id': int(manifest_rows[index]['image_id']),
                        'model': model,
                        'layer': layer,
                        'variant': variant,
                        'variant_kind': kind,
                        'band': band,
                        'clean_input_centered_rms': (
                            input_values['clean_centered_rms'][index]),
                        'variant_input_centered_rms': (
                            input_values['variant_centered_rms'][index]),
                        'input_centered_rms_ratio': (
                            input_values['input_centered_rms_ratio'][index]),
                        'input_relative_shift': (
                            input_values['input_relative_shift'][index]),
                        'clean_pooled_norm': clean_norm[index],
                        'variant_pooled_norm': changed_norm[index],
                        'pooled_norm_ratio': norm_ratio[index],
                        'pooled_relative_shift': feature_shift[index],
                        'normalized_feature_shift': feature_shift[index],
                        'pooled_cosine_similarity': similarity[index],
                        'pooled_cosine_distance': cosine_distance[index],
                        'input_normalized_response_gain': response_gain[index],
                        'input_normalized_shift_gain': shift_gain[index],
                        'clean_spatial_norm': clean_spatial[index],
                        'variant_spatial_norm': changed_spatial[index],
                        'spatial_norm_ratio': spatial_norm_ratio[index],
                    })
                for metric, values in (
                    ('pooled_norm_ratio', norm_ratio),
                    ('pooled_relative_shift', feature_shift),
                    ('normalized_feature_shift', feature_shift),
                    ('pooled_cosine_similarity', similarity),
                    ('pooled_cosine_distance', cosine_distance),
                    ('input_normalized_response_gain', response_gain),
                    ('input_normalized_shift_gain', shift_gain),
                    ('spatial_norm_ratio', spatial_norm_ratio),
                ):
                    summary_rows.append({
                        'model': model,
                        'layer': layer,
                        'variant': variant,
                        'variant_kind': kind,
                        'band': band,
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
    input_fields = list(input_summary_rows[0])
    with (out_dir / 'frequency_input_summary.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=input_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(input_summary_rows)
    write_json(out_dir / 'frequency_response_metadata.json', {
        'feature_root': str(root),
        'frequency_manifest': str(manifest_path),
        'models': models,
        'layers': requested_layers or 'discovered per model',
        'clean_variant': args.clean_variant,
        'frequency_variants': variants,
        'definitions': {
            'input_centered_rms_ratio': (
                'RMS(center(x_variant)) / max(RMS(center(x_clean)), eps)'),
            'input_relative_shift': (
                'RMS(center(x_variant) - center(x_clean)) / '
                'max(RMS(center(x_clean)), eps)'),
            'pooled_norm_ratio': '||f_band||_2 / max(||f_clean||_2, eps)',
            'pooled_relative_shift': (
                '||f_variant - f_clean||_2 / max(||f_clean||_2, eps)'),
            'normalized_feature_shift': (
                'Backward-compatible alias of pooled_relative_shift'),
            'pooled_cosine_similarity': (
                'cosine(f_variant, f_clean)'),
            'pooled_cosine_distance': (
                '1 - cosine(f_variant, f_clean)'),
            'input_normalized_response_gain': (
                'pooled_norm_ratio / max(input_centered_rms_ratio, eps)'),
            'input_normalized_shift_gain': (
                'pooled_relative_shift / max(input_relative_shift, eps)'),
            'spatial_norm_ratio': (
                '||F_band||_F / max(||F_clean||_F, eps)'),
        },
        'interpretation': {
            'band_pass': (
                'Use pooled_norm_ratio and input_normalized_response_gain to '
                'measure retained response to an isolated frequency band.'),
            'band_stop': (
                'Use pooled_relative_shift, pooled_cosine_distance and '
                'input_normalized_shift_gain to measure sensitivity when the '
                'named band is removed from the clean image.'),
        },
        'input_measurement': (
            'Metrics are computed from the actual quantized model-input PNGs; '
            'per-channel means are removed before RMS calculations.'),
    })
    print(f'Frequency response results: {out_dir}')


if __name__ == '__main__':
    main()
