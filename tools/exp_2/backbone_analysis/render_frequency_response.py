#!/usr/bin/env python3
"""Render paper-ready feature response, AP retention, and FG/BG figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .common import ensure_empty_or_create, write_json


MODEL_COLORS = ('#2f6fbb', '#d29b20', '#d95f4c', '#6f8f55')
BANDS = ('low', 'mid', 'high')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--activation-root', default='')
    parser.add_argument('--detection-metrics', default='')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_tsv(path: Path) -> List[dict]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def model_palette(models: Sequence[str]) -> Mapping[str, str]:
    return {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}


def selected_summary(rows: Sequence[dict], metric: str, kind: str) -> List[dict]:
    return [
        row for row in rows
        if row['metric'] == metric and row['variant_kind'] == kind
    ]


def line_chart(
    rows: Sequence[dict], metric: str, kind: str, output: Path,
    title: str, ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    selected = selected_summary(rows, metric, kind)
    models = sorted({row['model'] for row in selected})
    layers = sorted({row['layer'] for row in selected})
    palette = model_palette(models)
    figure, axes = plt.subplots(
        1, len(layers), figsize=(3.35 * len(layers), 3.8),
        sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers):
        x = np.arange(len(BANDS))
        for model in models:
            values = {
                row['band']: row for row in selected
                if row['model'] == model and row['layer'] == layer
            }
            y = [float(values[band]['median']) for band in BANDS]
            low = [float(values[band]['p05']) for band in BANDS]
            high = [float(values[band]['p95']) for band in BANDS]
            axis.plot(x, y, marker='o', linewidth=1.8, label=model,
                      color=palette[model])
            axis.fill_between(x, low, high, alpha=0.12, color=palette[model])
        axis.set_title(layer)
        axis.grid(axis='y', color='#dddddd', linewidth=0.7)
        axis.set_xticks(x, BANDS)
        axis.set_xlabel('Frequency band')
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)),
                  frameon=False)
    figure.suptitle(title, y=1.06)
    for suffix in ('png', 'pdf'):
        figure.savefig(output.with_suffix(f'.{suffix}'), dpi=240,
                       bbox_inches='tight')
    plt.close(figure)


def heatmaps(rows: Sequence[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    specifications = (
        ('pooled_norm_ratio', 'band_pass', 'Feature norm retention'),
        ('input_normalized_response_gain', 'band_pass', 'Input-normalized response gain'),
        ('pooled_relative_shift', 'band_stop', 'Band-stop feature shift'),
        ('pooled_cosine_distance', 'band_stop', 'Band-stop cosine distance'),
    )
    for metric, kind, title in specifications:
        selected = selected_summary(rows, metric, kind)
        models = sorted({row['model'] for row in selected})
        layers = sorted({row['layer'] for row in selected})
        figure, axes = plt.subplots(
            1, len(models), figsize=(4.2 * len(models), 3.8),
            constrained_layout=True)
        axes = np.atleast_1d(axes)
        matrices = []
        for model in models:
            lookup = {
                (row['layer'], row['band']): float(row['median'])
                for row in selected if row['model'] == model
            }
            matrices.append(np.asarray([
                [lookup[(layer, band)] for band in BANDS] for layer in layers]))
        lower = min(float(matrix.min()) for matrix in matrices)
        upper = max(float(matrix.max()) for matrix in matrices)
        for axis, model, matrix in zip(axes, models, matrices):
            image = axis.imshow(matrix, cmap='cividis', vmin=lower, vmax=upper,
                                aspect='auto')
            axis.set_xticks(range(len(BANDS)), BANDS)
            axis.set_yticks(range(len(layers)), layers)
            axis.set_title(model)
            for row_index in range(len(layers)):
                for column_index in range(len(BANDS)):
                    axis.text(column_index, row_index,
                              f'{matrix[row_index, column_index]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if matrix[row_index, column_index] > (lower + upper) / 2 else '#111111')
        figure.colorbar(image, ax=axes.tolist(), shrink=0.85, label=title)
        figure.suptitle(f'Layer-frequency response: {title}')
        stem = out_dir / f'layer_band_heatmap_{metric}'
        for suffix in ('png', 'pdf'):
            figure.savefig(stem.with_suffix(f'.{suffix}'), dpi=240)
        plt.close(figure)


def activation_summary(root: Path, out_dir: Path) -> List[dict]:
    import matplotlib.pyplot as plt

    combined = []
    for path in sorted(root.glob('*/activation_statistics.tsv')):
        variant = path.parent.name
        for row in read_tsv(path):
            row['variant'] = variant
            combined.append(row)
    if not combined:
        return []
    grouped = {}
    for row in combined:
        key = (row['model'], row['layer'], row['variant'])
        grouped.setdefault(key, []).append(float(row['fg_bg_ratio']))
    summary_rows = []
    for (model, layer, variant), values in grouped.items():
        finite = np.asarray([value for value in values if np.isfinite(value)])
        summary_rows.append({
            'model': model, 'layer': layer, 'variant': variant,
            'median_fg_bg_ratio': float(np.median(finite)),
            'p05': float(np.percentile(finite, 5)),
            'p95': float(np.percentile(finite, 95)),
        })
    write_tsv(out_dir / 'frequency_fg_bg_summary.tsv', summary_rows)
    models = sorted({row['model'] for row in summary_rows})
    layers = sorted({row['layer'] for row in summary_rows})
    variants = ['clean', 'low', 'mid', 'high', 'remove_low', 'remove_mid', 'remove_high']
    palette = model_palette(models)
    figure, axes = plt.subplots(
        1, len(layers), figsize=(3.6 * len(layers), 4.0), sharey=True,
        constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, layer in zip(axes, layers):
        for model in models:
            lookup = {
                row['variant']: float(row['median_fg_bg_ratio'])
                for row in summary_rows if row['model'] == model and row['layer'] == layer
            }
            available = [variant for variant in variants if variant in lookup]
            axis.plot(available, [lookup[value] for value in available], marker='o',
                      color=palette[model], label=model)
        axis.axhline(1.0, color='#555555', linestyle='--', linewidth=1)
        axis.tick_params(axis='x', rotation=45)
        axis.set_title(layer)
        axis.grid(axis='y', color='#dddddd', linewidth=0.7)
    axes[0].set_ylabel('Median foreground/background activation ratio')
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)),
                  frameon=False)
    figure.suptitle('Foreground/background response under frequency changes', y=1.08)
    for suffix in ('png', 'pdf'):
        figure.savefig(out_dir / f'frequency_fg_bg_response.{suffix}', dpi=240,
                       bbox_inches='tight')
    plt.close(figure)
    return summary_rows


def detection_figures(metrics_path: Path, response_rows: Sequence[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = read_tsv(metrics_path)
    models = sorted({row['model'] for row in metrics})
    palette = model_palette(models)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for model in models:
        values = [row for row in metrics if row['model'] == model and row['variant_kind'] == 'band_stop']
        lookup = {row['band']: float(row['bbox_mAP_retention']) for row in values}
        axis.plot(BANDS, [lookup[band] for band in BANDS], marker='o', linewidth=1.8,
                  color=palette[model], label=model)
    axis.axhline(1.0, color='#555555', linestyle='--', linewidth=1)
    axis.set_xlabel('Removed frequency band')
    axis.set_ylabel('BBox mAP retention')
    axis.set_title('Band-stop detection performance retention')
    axis.grid(axis='y', color='#dddddd', linewidth=0.7)
    axis.legend(frameon=False)
    for suffix in ('png', 'pdf'):
        figure.savefig(out_dir / f'band_stop_ap_retention.{suffix}', dpi=240)
    plt.close(figure)

    points = []
    metric_lookup = {
        (row['model'], row['band']): row for row in metrics
        if row['variant_kind'] == 'band_stop'
    }
    for row in response_rows:
        if row['metric'] != 'pooled_relative_shift' or row['variant_kind'] != 'band_stop':
            continue
        key = (row['model'], row['band'])
        if key not in metric_lookup:
            continue
        ap = metric_lookup[key]
        points.append({
            'model': row['model'], 'layer': row['layer'], 'band': row['band'],
            'median_feature_shift': float(row['median']),
            'bbox_mAP_drop': float(ap['clean_bbox_mAP']) - float(ap['bbox_mAP']),
        })
    write_tsv(out_dir / 'feature_shift_ap_drop_points.tsv', points)
    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    markers = {'low': 'o', 'mid': 's', 'high': '^'}
    for point in points:
        axis.scatter(point['median_feature_shift'], point['bbox_mAP_drop'],
                     color=palette[point['model']], marker=markers[point['band']], s=45)
        axis.annotate(f'{point["layer"]}/{point["band"]}',
                      (point['median_feature_shift'], point['bbox_mAP_drop']),
                      xytext=(4, 3), textcoords='offset points', fontsize=7)
    axis.set_xlabel('Median normalized feature shift')
    axis.set_ylabel('BBox mAP drop')
    axis.set_title('Representation shift versus detection degradation')
    axis.grid(True, color='#dddddd', linewidth=0.7)
    for suffix in ('png', 'pdf'):
        figure.savefig(out_dir / f'feature_shift_ap_drop.{suffix}', dpi=240)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    response_dir = Path(args.response_dir).expanduser().resolve()
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    rows = read_tsv(response_dir / 'frequency_response_summary.tsv')
    line_chart(rows, 'pooled_norm_ratio', 'band_pass', out_dir / 'band_feature_norm',
               'Band-pass feature norm retention', 'Feature norm / clean norm')
    line_chart(rows, 'pooled_relative_shift', 'band_pass',
               out_dir / 'bandpass_feature_distance',
               'Band-pass representation distance from clean',
               'Normalized feature distance')
    line_chart(rows, 'input_normalized_shift_gain', 'band_pass',
               out_dir / 'bandpass_input_normalized_feature_distance',
               'Input-normalized band-pass feature distance',
               'Feature shift / input shift')
    line_chart(rows, 'pooled_relative_shift', 'band_stop',
               out_dir / 'bandstop_feature_distance',
               'Band-stop representation distance', 'Normalized feature distance')
    line_chart(rows, 'input_normalized_shift_gain', 'band_stop',
               out_dir / 'bandstop_input_normalized_feature_distance',
               'Input-normalized band-stop feature distance',
               'Feature shift / input shift')
    heatmaps(rows, out_dir)
    fg_rows = []
    if args.activation_root:
        fg_rows = activation_summary(Path(args.activation_root), out_dir)
    if args.detection_metrics:
        detection_figures(Path(args.detection_metrics), rows, out_dir)
    write_json(out_dir / 'frequency_figure_metadata.json', {
        'response_dir': str(response_dir),
        'activation_root': args.activation_root or None,
        'detection_metrics': args.detection_metrics or None,
        'figures': 'PNG and PDF use identical data and shared model colors',
        'fg_bg_rows': len(fg_rows),
    })
    print(f'Frequency response figures: {out_dir}')


if __name__ == '__main__':
    main()
