#!/usr/bin/env python3
"""Render only input-normalized feature response and FG/BG response figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from tools.exp_2.backbone_analysis.common import ensure_empty_or_create, parse_csv, read_jsonl, write_json


VARIANTS = ('clean', 'low', 'mid', 'high', 'remove_low', 'remove_mid', 'remove_high')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#332288')
MARKERS = ('o', 's', '^', 'D', 'P', 'X', 'v', '<')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--frequency-manifest', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--pretrained-models', required=True)
    parser.add_argument('--detector-models', required=True)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def input_path(row: Mapping[str, object], variant: str) -> Path:
    variants = row.get('variants', {})
    if variant == 'clean' and variant not in variants:
        return Path(str(row['image_path'])).expanduser().resolve()
    payload = variants.get(variant)
    if not isinstance(payload, Mapping) or not payload.get('image_path'):
        raise KeyError(f'Missing {variant} image for sample {row.get("sample_index")}')
    return Path(str(payload['image_path'])).expanduser().resolve()


def centered_rms(path: Path) -> float:
    with Image.open(path) as opened:
        value = np.asarray(opened.convert('RGB'), dtype=np.float64) / 255.0
    value = value - value.mean(axis=(0, 1), keepdims=True)
    return float(np.sqrt(np.mean(np.square(value))))


def spatial_path(root: Path, model: str, variant: str, index: int, image_id: int, layer: str) -> Path:
    return root / 'spatial' / model / variant / f'{index:05d}_{image_id}' / f'{layer}.npz'


def load_feature(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Missing spatial feature: {path}')
    with np.load(path, allow_pickle=False) as payload:
        value = payload['feature'].astype(np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f'Invalid CHW feature: {path}, shape={value.shape}')
    return value


def mask_from_boxes(boxes: Sequence[Sequence[float]], height: int, width: int, original_height: int, original_width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        left = max(0, min(width, int(math.floor(x1 * width / max(original_width, 1)))))
        right = max(0, min(width, int(math.ceil(x2 * width / max(original_width, 1)))))
        top = max(0, min(height, int(math.floor(y1 * height / max(original_height, 1)))))
        bottom = max(0, min(height, int(math.ceil(y2 * height / max(original_height, 1)))))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def summarize(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {'samples': 0, 'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'), 'p05': float('nan'), 'p95': float('nan')}
    return {
        'samples': int(len(array)), 'mean': float(array.mean()), 'std': float(array.std()),
        'sem': float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        'p05': float(np.percentile(array, 5)), 'p95': float(np.percentile(array, 95)),
    }


def write_tsv(path: Path, rows: List[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: List[dict], models: List[str], layers: List[str], metric: str, ylabel: str, title: str, stem: Path) -> None:
    import matplotlib.pyplot as plt
    palette = {model: COLORS[index] for index, model in enumerate(models)}
    markers = {model: MARKERS[index] for index, model in enumerate(models)}
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True, constrained_layout=True)
    for axis, layer in zip(axes.flat, layers):
        for model in models:
            lookup = {row['variant']: row for row in rows if row['model'] == model and row['layer'] == layer}
            values = [float(lookup[variant][metric]) for variant in VARIANTS]
            errors = [float(lookup[variant]['sem']) for variant in VARIANTS]
            x = np.arange(len(VARIANTS))
            axis.plot(x, values, color=palette[model], marker=markers[model], linewidth=2.0, markersize=5, label=model)
            axis.fill_between(x, np.asarray(values) - np.asarray(errors), np.asarray(values) + np.asarray(errors), color=palette[model], alpha=0.14)
        axis.set_title(layer)
        axis.grid(axis='y', color='#d9d9d9', linewidth=0.7)
        axis.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=35, ha='right')
        axis.set_ylabel(ylabel)
        if metric == 'log_fg_bg_ratio':
            axis.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=min(4, len(models)), frameon=False)
    figure.suptitle(title, y=1.03)
    for suffix in ('png', 'pdf'):
        figure.savefig(stem.with_suffix(f'.{suffix}'), dpi=240, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    rows = read_jsonl(Path(args.frequency_manifest).expanduser().resolve())
    models, layers, variants = parse_csv(args.models), parse_csv(args.layers), parse_csv(args.variants)
    pretrained, detectors = parse_csv(args.pretrained_models), parse_csv(args.detector_models)
    if tuple(variants) != VARIANTS:
        raise ValueError(f'Variants must use the fixed paper order: {",".join(VARIANTS)}')
    if set(pretrained) | set(detectors) != set(models) or set(pretrained) & set(detectors):
        raise ValueError('Each --models ID must occur in exactly one analysis group')
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    input_rms = {(index, variant): centered_rms(input_path(row, variant)) for index, row in enumerate(rows) for variant in variants}
    per_sample: List[dict] = []
    for model in models:
        group = 'pretrained' if model in pretrained else 'detector'
        for index, row in enumerate(rows):
            image_id = int(row['image_id'])
            width, height = int(row['width']), int(row['height'])
            for layer in layers:
                for variant in variants:
                    feature = load_feature(spatial_path(root, model, variant, index, image_id, layer))
                    feature_rms = float(np.sqrt(np.mean(np.square(feature, dtype=np.float64))))
                    activation = np.abs(feature).mean(axis=0)
                    fg_mask = mask_from_boxes(row.get('boxes_xyxy', []), activation.shape[0], activation.shape[1], height, width)
                    bg_mask = ~fg_mask
                    fg = float(activation[fg_mask].mean()) if fg_mask.any() else float('nan')
                    bg = float(activation[bg_mask].mean()) if bg_mask.any() else float('nan')
                    ratio = fg / max(bg, args.eps) if np.isfinite(fg) else float('nan')
                    per_sample.append({
                        'sample_index': index, 'image_id': image_id, 'model': model, 'group': group,
                        'layer': layer, 'variant': variant, 'input_centered_rms': input_rms[(index, variant)],
                        'feature_rms': feature_rms, 'feature_input_norm': feature_rms / max(input_rms[(index, variant)], args.eps),
                        'fg_mean_abs_activation': fg, 'bg_mean_abs_activation': bg,
                        'fg_bg_ratio': ratio, 'log_fg_bg_ratio': float(np.log(max(ratio, args.eps))),
                    })
    write_tsv(out_dir / 'frequency_per_sample.tsv', per_sample)
    summary: List[dict] = []
    for group, group_models in (('pretrained', pretrained), ('detector', detectors)):
        for model in group_models:
            for layer in layers:
                for variant in variants:
                    selected = [row for row in per_sample if row['model'] == model and row['layer'] == layer and row['variant'] == variant]
                    for metric in ('feature_input_norm', 'fg_bg_ratio', 'log_fg_bg_ratio'):
                        summary.append({'group': group, 'model': model, 'layer': layer, 'variant': variant, 'metric': metric, **summarize(row[metric] for row in selected)})
    write_tsv(out_dir / 'frequency_summary.tsv', summary)
    for group, group_models in (('pretrained', pretrained), ('detector', detectors)):
        for metric, ylabel, title, name in (
            ('feature_input_norm', 'Feature RMS / input RMS', f'{group}: input-normalized feature response', 'feature_input_norm'),
            ('log_fg_bg_ratio', 'log(FG/BG response ratio)', f'{group}: foreground/background response', 'fg_bg_ratio'),
        ):
            selected = [row for row in summary if row['group'] == group and row['metric'] == metric]
            plot_metric(selected, group_models, layers, metric, ylabel, title, out_dir / f'{name}_{group}')
    write_json(out_dir / 'metadata.json', {
        'feature_root': str(root), 'frequency_manifest': str(Path(args.frequency_manifest).resolve()),
        'variants': list(VARIANTS), 'pretrained_models': pretrained, 'detector_models': detectors,
        'feature_input_norm': 'RMS(raw CHW feature) / RMS(channel-centered RGB input)',
        'fg_bg_ratio': 'mean(abs(feature), channels) in GT-box union / background complement',
        'log_fg_bg_ratio': 'natural log of FG/BG ratio; zero means equal foreground/background response',
        'visual_outputs': ['feature_input_norm_pretrained', 'feature_input_norm_detector', 'fg_bg_ratio_pretrained', 'fg_bg_ratio_detector'],
    })
    print(f'Frequency metrics and figures: {out_dir}')


if __name__ == '__main__':
    main()
