#!/usr/bin/env python3
"""Render compact paper-style frequency figures from frequency_summary.tsv.

This is intentionally a rendering-only companion to compute_frequency_metrics.
It never rereads images or feature tensors, and it does not modify the source
frequency analysis directory.  Point --out-dir at a new directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


VARIANTS = ('clean', 'low', 'mid', 'high', 'remove_low', 'remove_mid', 'remove_high')
VARIANT_LABELS = ('Clean', 'Low', 'Mid', 'High', '-Low', '-Mid', '-High')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#332288', '#009E73')
MARKERS = ('o', 's', '^', 'D', 'P', 'X', 'v', '<')
DISPLAY_NAMES = {
    'imagenet_dino100e_backbone': 'ImageNet',
    'realuw_dino100e_backbone': 'RealUW',
    'synthetic5_dino100e_backbone': 'Synthetic5',
    'imagenet_dino100e_dfui_backbone': 'ImageNet + DFUI',
    'imagenet_dino100e_ruod_cascade': 'ImageNet -> RUOD',
    'realuw_dino100e_ruod_cascade': 'RealUW -> RUOD',
    'synthetic5_dino100e_ruod_cascade': 'Synthetic5 -> RUOD',
    'imagenet_dino100e_dfui_ruod_cascade': 'ImageNet + DFUI -> RUOD',
}


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-tsv', required=True, help='frequency_summary.tsv from a completed run')
    parser.add_argument('--group', required=True, choices=('pretrained', 'detector'))
    parser.add_argument('--models', required=True, help='Comma-separated model IDs in legend order')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--reference-model', default='', help='Optional muted dashed reference curve')
    parser.add_argument('--out-dir', required=True, help='New output directory; source files are never changed')
    parser.add_argument('--prefix', default='paper_frequency')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory is non-empty: {path}; use a new directory or --overwrite')
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rows(path: Path) -> List[dict]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    required = {'group', 'model', 'layer', 'variant', 'metric', 'mean', 'sem'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Expected frequency summary columns {sorted(required)} in {path}')
    return rows


def record_map(rows: Iterable[dict], metric: str) -> Dict[tuple, dict]:
    result: Dict[tuple, dict] = {}
    for row in rows:
        if row['metric'] == metric:
            result[(row['model'], row['layer'], row['variant'])] = row
    return result


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace('_', ' '))


def plot_metric(
    rows: Sequence[dict], models: Sequence[str], layers: Sequence[str], metric: str,
    ylabel: str, title: str, reference_model: str, stem: Path, dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    lookup = record_map(rows, metric)
    requested = list(models)
    if reference_model and reference_model not in requested:
        requested.append(reference_model)
    missing = [
        (model, layer, variant) for model in requested for layer in layers for variant in VARIANTS
        if (model, layer, variant) not in lookup
    ]
    if missing:
        raise KeyError(f'Missing {metric} summary rows, first missing key: {missing[0]}')

    # A 2x2 grid is compact enough for a two-column paper while preserving
    # layer-specific y ranges, which avoids flattening shallow-layer effects.
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.7), sharex=True)
    x = np.arange(len(VARIANTS))
    palette = {model: COLORS[index % len(COLORS)] for index, model in enumerate(models)}
    marker_map = {model: MARKERS[index % len(MARKERS)] for index, model in enumerate(models)}
    handles = []
    labels = []
    for index, (axis, layer) in enumerate(zip(axes.flat, layers)):
        for model in requested:
            values = np.asarray([float(lookup[(model, layer, variant)]['mean']) for variant in VARIANTS])
            errors = np.asarray([float(lookup[(model, layer, variant)]['sem']) for variant in VARIANTS])
            if model == reference_model:
                line = axis.plot(
                    x, values, color='#3F3F3F', linewidth=1.7, linestyle='--', marker='o',
                    markersize=3.8, label=f'{display_name(model)} (reference)', zorder=2)[0]
            else:
                line = axis.plot(
                    x, values, color=palette[model], linewidth=1.8, marker=marker_map[model],
                    markersize=4.6, label=display_name(model), zorder=3)[0]
            axis.fill_between(x, values - errors, values + errors, color=line.get_color(), alpha=0.10, linewidth=0)
            if index == 0:
                handles.append(line)
                labels.append(line.get_label())
        axis.set_title(layer, fontsize=10, fontweight='semibold', pad=5)
        axis.set_xticks(x, VARIANT_LABELS)
        axis.tick_params(axis='x', labelrotation=32, labelsize=8)
        axis.tick_params(axis='y', labelsize=8)
        axis.grid(axis='y', color='#D8D8D8', linewidth=0.6, zorder=0)
        for spine_name in ('top', 'right'):
            axis.spines[spine_name].set_visible(False)
        if metric == 'log_fg_bg_ratio':
            axis.axhline(0.0, color='#555555', linewidth=0.9, linestyle='--', zorder=1)
        if index % 2 == 0:
            axis.set_ylabel(ylabel, fontsize=9)
        if index >= 2:
            axis.set_xlabel('Frequency variant', fontsize=9)

    figure.suptitle(title, fontsize=12, fontweight='semibold', y=0.985)
    figure.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.945),
                  frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.2)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.82))
    for suffix in ('png', 'pdf'):
        figure.savefig(stem.with_suffix(f'.{suffix}'), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_tsv).expanduser().resolve()
    all_rows = read_rows(summary_path)
    models = parse_csv(args.models)
    layers = parse_csv(args.layers)
    if len(layers) != 4:
        raise ValueError('This paper renderer expects exactly four semantic layers')
    if not models:
        raise ValueError('--models must not be empty')
    output = prepare_output(Path(args.out_dir), args.overwrite)
    reference = args.reference_model.strip()
    # Pretraining panels may intentionally append one detector backbone as a
    # downstream reference. Keep those rows even though their group differs.
    rows = [
        row for row in all_rows
        if row['group'] == args.group or (reference and row['model'] == reference)
    ]
    plot_metric(
        rows, models, layers, 'feature_norm_over_clean', 'Feature RMS / clean feature RMS',
        'Feature response relative to clean input', reference,
        output / f'{args.prefix}_feature_response', args.dpi)
    plot_metric(
        rows, models, layers, 'log_fg_bg_ratio', 'log(FG / BG response)',
        'Foreground-background response contrast', reference,
        output / f'{args.prefix}_fg_bg_response', args.dpi)
    metadata = {
        'source_summary_tsv': str(summary_path), 'group': args.group, 'models': models,
        'reference_model': reference, 'layers': layers, 'variants': list(VARIANTS),
        'variant_labels': list(VARIANT_LABELS),
        'feature_metric': 'Feature RMS(variant) / Feature RMS(clean), matched by model, image, and layer',
        'foreground_background_metric': 'log(mean abs activation in GT foreground / mean abs activation in background)',
        'layout': '2x2 layers with compact legend and short variant labels',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Paper frequency figures: {output}')


if __name__ == '__main__':
    main()
