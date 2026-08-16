#!/usr/bin/env python3
"""Render only input-normalized feature response and FG/BG response figures."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from tools.exp_2.backbone_analysis.common import ensure_empty_or_create, parse_csv, read_jsonl, write_json


VARIANTS = ('clean', 'low', 'mid', 'high', 'remove_low', 'remove_mid', 'remove_high')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#332288')
MARKERS = ('o', 's', '^', 'D', 'P', 'X', 'v', '<')
DISPLAY_NAMES = {
    'imagenet_dino100e_backbone': 'ImageNet pretrain',
    'realuw_dino100e_backbone': 'RealUW pretrain',
    'synthetic5_dino100e_backbone': 'Synthetic5 pretrain',
    'imagenet_dino100e_dfui_backbone': 'ImageNet + DFUI pretrain',
    'imagenet_dino100e_ruod_cascade': 'ImageNet -> RUOD Cascade',
    'realuw_dino100e_ruod_cascade': 'RealUW -> RUOD Cascade',
    'synthetic5_dino100e_ruod_cascade': 'Synthetic5 -> RUOD Cascade',
    'imagenet_dino100e_dfui_ruod_cascade': 'ImageNet + DFUI -> RUOD Cascade',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--frequency-manifest', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--pretrained-models', required=True)
    parser.add_argument('--detector-models', required=True)
    parser.add_argument(
        '--pretrained-reference-model',
        default='imagenet_dino100e_ruod_cascade',
        help=(
            'Detector model appended to pretrained frequency plots as the '
            'ImageNet-to-RUOD reference curve; pass an empty string to disable.'))
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument(
        '--model-workers', type=int, default=1,
        help='Independent model processes used to read spatial features.')
    parser.add_argument(
        '--reuse-per-sample', default='',
        help=(
            'Existing frequency_per_sample.tsv generated with the former '
            'variant-input denominator. Rebase its saved feature RMS values '
            'to each sample clean-input RMS without re-reading spatial features.'))
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


def parse_float(value: object) -> float:
    return float(str(value))


def rebase_existing_per_sample(path: Path, eps: float) -> List[dict]:
    """Convert prior variant-normalized rows to clean-normalized rows."""
    with path.open('r', encoding='utf-8', newline='') as handle:
        source_rows = list(csv.DictReader(handle, delimiter='\t'))
    if not source_rows:
        raise ValueError(f'No rows in --reuse-per-sample: {path}')
    required = {'sample_index', 'variant', 'input_centered_rms', 'feature_rms'}
    missing = required - set(source_rows[0])
    if missing:
        raise ValueError(
            f'Cannot rebase {path}; missing expected columns: {sorted(missing)}')
    clean_rms = {
        int(row['sample_index']): parse_float(row['input_centered_rms'])
        for row in source_rows if row['variant'] == 'clean'
    }
    rebased = []
    for row in source_rows:
        sample_index = int(row['sample_index'])
        if sample_index not in clean_rms:
            raise ValueError(f'Missing clean RMS for sample_index={sample_index}')
        feature_rms = parse_float(row['feature_rms'])
        variant_rms = parse_float(row['input_centered_rms'])
        clean_value = clean_rms[sample_index]
        rebased.append({
            'sample_index': sample_index,
            'image_id': int(row['image_id']),
            'model': row['model'], 'group': row['group'],
            'layer': row['layer'], 'variant': row['variant'],
            'variant_input_centered_rms': variant_rms,
            'clean_input_centered_rms': clean_value,
            'feature_rms': feature_rms,
            'feature_clean_norm': feature_rms / max(clean_value, eps),
            'fg_mean_abs_activation': parse_float(row['fg_mean_abs_activation']),
            'bg_mean_abs_activation': parse_float(row['bg_mean_abs_activation']),
            'fg_bg_ratio': parse_float(row['fg_bg_ratio']),
            'log_fg_bg_ratio': parse_float(row['log_fg_bg_ratio']),
        })
    print(f'Rebased {len(rebased)} existing per-sample frequency rows from {path}', flush=True)
    return rebased


def plot_metric(rows: List[dict], models: List[str], layers: List[str], metric: str, ylabel: str, title: str, stem: Path) -> None:
    import matplotlib.pyplot as plt
    palette = {model: COLORS[index] for index, model in enumerate(models)}
    markers = {model: MARKERS[index] for index, model in enumerate(models)}
    figure, axes = plt.subplots(1, len(layers), figsize=(4.7 * len(layers), 5.7), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(VARIANTS))
    for axis, layer in zip(axes, layers):
        for model in models:
            lookup = {row['variant']: row for row in rows if row['model'] == model and row['layer'] == layer}
            values = [float(lookup[variant]['mean']) for variant in VARIANTS]
            errors = [float(lookup[variant]['sem']) for variant in VARIANTS]
            axis.plot(
                x, values, color=palette[model], marker=markers[model],
                linewidth=2.0, markersize=5, label=DISPLAY_NAMES.get(model, model))
            axis.fill_between(x, np.asarray(values) - np.asarray(errors), np.asarray(values) + np.asarray(errors), color=palette[model], alpha=0.14)
        axis.set_title(layer, pad=10)
        axis.grid(axis='y', color='#d9d9d9', linewidth=0.7)
        axis.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=35, ha='right')
        axis.set_xlabel('Input frequency variant')
        axis.set_ylabel(ylabel)
        if metric == 'log_fg_bg_ratio':
            axis.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.985, fontsize=15)
    figure.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.925),
        ncol=min(5, len(models)), frameon=False, columnspacing=1.4, handlelength=2.4)
    figure.tight_layout(rect=(0.01, 0.02, 0.99, 0.79))
    for suffix in ('png', 'pdf'):
        figure.savefig(stem.with_suffix(f'.{suffix}'), dpi=240, bbox_inches='tight')
    plt.close(figure)


def process_model(task: Mapping[str, object]) -> List[dict]:
    """Compute all sample/layer/variant metrics for one isolated model."""
    root = Path(str(task['feature_root']))
    model = str(task['model'])
    group = str(task['group'])
    rows = task['rows']
    layers = task['layers']
    variants = task['variants']
    input_rms = task['input_rms']
    eps = float(task['eps'])
    result: List[dict] = []
    for index, row in enumerate(rows):
        image_id = int(row['image_id'])
        width, height = int(row['width']), int(row['height'])
        for layer in layers:
            for variant in variants:
                feature = load_feature(spatial_path(root, model, variant, index, image_id, layer))
                feature_rms = float(np.sqrt(np.mean(np.square(feature, dtype=np.float64))))
                activation = np.abs(feature).mean(axis=0)
                fg_mask = mask_from_boxes(
                    row.get('boxes_xyxy', []), activation.shape[0], activation.shape[1], height, width)
                bg_mask = ~fg_mask
                fg = float(activation[fg_mask].mean()) if fg_mask.any() else float('nan')
                bg = float(activation[bg_mask].mean()) if bg_mask.any() else float('nan')
                ratio = fg / max(bg, eps) if np.isfinite(fg) else float('nan')
                variant_input_rms = float(input_rms[(index, variant)])
                clean_input_rms = float(input_rms[(index, 'clean')])
                result.append({
                    'sample_index': index, 'image_id': image_id, 'model': model, 'group': group,
                    'layer': layer, 'variant': variant,
                    'variant_input_centered_rms': variant_input_rms,
                    'clean_input_centered_rms': clean_input_rms,
                    'feature_rms': feature_rms,
                    'feature_clean_norm': feature_rms / max(clean_input_rms, eps),
                    'fg_mean_abs_activation': fg, 'bg_mean_abs_activation': bg,
                    'fg_bg_ratio': ratio, 'log_fg_bg_ratio': float(np.log(max(ratio, eps))),
                })
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root).expanduser().resolve()
    rows = read_jsonl(Path(args.frequency_manifest).expanduser().resolve())
    models, layers, variants = parse_csv(args.models), parse_csv(args.layers), parse_csv(args.variants)
    pretrained, detectors = parse_csv(args.pretrained_models), parse_csv(args.detector_models)
    if tuple(variants) != VARIANTS:
        raise ValueError(f'Variants must use the fixed paper order: {",".join(VARIANTS)}')
    if args.model_workers <= 0:
        raise ValueError('--model-workers must be positive')
    if set(pretrained) | set(detectors) != set(models) or set(pretrained) & set(detectors):
        raise ValueError('Each --models ID must occur in exactly one analysis group')
    reference_model = args.pretrained_reference_model.strip()
    if reference_model and reference_model not in detectors:
        raise ValueError('--pretrained-reference-model must be one of --detector-models')
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    if args.reuse_per_sample:
        per_sample = rebase_existing_per_sample(
            Path(args.reuse_per_sample).expanduser().resolve(), args.eps)
    else:
        input_rms = {
            (index, variant): centered_rms(input_path(row, variant))
            for index, row in enumerate(rows) for variant in variants}
        tasks = [{
            'feature_root': str(root), 'model': model,
            'group': 'pretrained' if model in pretrained else 'detector',
            'rows': rows, 'layers': layers, 'variants': variants,
            'input_rms': input_rms, 'eps': args.eps,
        } for model in models]
        per_sample = []
        if args.model_workers == 1:
            for task in tasks:
                per_sample.extend(process_model(task))
        else:
            with ProcessPoolExecutor(max_workers=min(args.model_workers, len(tasks))) as executor:
                futures = {executor.submit(process_model, task): str(task['model']) for task in tasks}
                for future in as_completed(futures):
                    model = futures[future]
                    per_sample.extend(future.result())
                    print(f'Frequency metrics completed model={model}', flush=True)
    per_sample.sort(key=lambda row: (
        str(row['group']), str(row['model']), int(row['sample_index']),
        str(row['layer']), VARIANTS.index(str(row['variant']))))
    write_tsv(out_dir / 'frequency_per_sample.tsv', per_sample)
    summary: List[dict] = []
    for group, group_models in (('pretrained', pretrained), ('detector', detectors)):
        for model in group_models:
            for layer in layers:
                for variant in variants:
                    selected = [row for row in per_sample if row['model'] == model and row['layer'] == layer and row['variant'] == variant]
                    for metric in ('feature_clean_norm', 'fg_bg_ratio', 'log_fg_bg_ratio'):
                        summary.append({'group': group, 'model': model, 'layer': layer, 'variant': variant, 'metric': metric, **summarize(row[metric] for row in selected)})
    write_tsv(out_dir / 'frequency_summary.tsv', summary)
    for group, group_models in (('pretrained', pretrained), ('detector', detectors)):
        for metric, ylabel, title, name in (
            ('feature_clean_norm', 'Feature RMS / clean-image RMS', f'{group}: clean-normalized feature response', 'feature_clean_norm'),
            ('log_fg_bg_ratio', 'log(FG/BG response ratio)', f'{group}: foreground/background response', 'fg_bg_ratio'),
        ):
            selected = [row for row in summary if row['group'] == group and row['metric'] == metric]
            plot_models = list(group_models)
            if group == 'pretrained' and reference_model:
                selected.extend(
                    row for row in summary
                    if row['model'] == reference_model and row['metric'] == metric)
                plot_models.append(reference_model)
                title = f'{title} (with ImageNet -> RUOD Cascade reference)'
            plot_metric(selected, plot_models, layers, metric, ylabel, title, out_dir / f'{name}_{group}')
    write_json(out_dir / 'metadata.json', {
        'feature_root': str(root), 'frequency_manifest': str(Path(args.frequency_manifest).resolve()),
        'variants': list(VARIANTS), 'pretrained_models': pretrained, 'detector_models': detectors,
        'model_workers': args.model_workers,
        'reuse_per_sample': str(Path(args.reuse_per_sample).resolve()) if args.reuse_per_sample else '',
        'feature_clean_norm': (
            'RMS(raw CHW feature) / RMS(channel-centered clean RGB input from the same sample)'),
        'variant_input_centered_rms': 'RMS of the current clean/band-stop/band input, retained for QA only',
        'fg_bg_ratio': 'mean(abs(feature), channels) in GT-box union / background complement',
        'log_fg_bg_ratio': 'natural log of FG/BG ratio; zero means equal foreground/background response',
        'pretrained_reference_model': reference_model,
        'visual_outputs': ['feature_clean_norm_pretrained', 'feature_clean_norm_detector', 'fg_bg_ratio_pretrained', 'fg_bg_ratio_detector'],
    })
    print(f'Frequency metrics and figures: {out_dir}')


if __name__ == '__main__':
    main()
