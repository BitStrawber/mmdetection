#!/usr/bin/env python3
"""Create paper-oriented figures from fixed-GT CAM metric tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-tsv', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--model-order', default='')
    parser.add_argument('--layer-order', default='res2,res3,res4,res5')
    parser.add_argument('--dpi', type=int, default=240)
    return parser.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if not rows:
        raise ValueError(f'No data rows in {path}')
    return rows


def parse_order(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def summarize(rows: List[Dict[str, str]], models: List[str], layers: List[str]):
    metrics = (
        'energy_in_target_box',
        'target_to_background_ratio',
        'pointing_game_hit',
        'top20_iou_with_target',
        'normalized_entropy',
        'peak_distance_over_box_diagonal',
    )
    result = {}
    for metric in metrics:
        matrix = np.full((len(models), len(layers)), np.nan, dtype=np.float64)
        spread = np.full_like(matrix, np.nan)
        count = np.zeros(matrix.shape, dtype=np.int64)
        for model_index, model in enumerate(models):
            for layer_index, layer in enumerate(layers):
                values = np.asarray([
                    float(row[metric]) for row in rows
                    if row['model'] == model and row['layer'] == layer
                ], dtype=np.float64)
                if values.size:
                    matrix[model_index, layer_index] = float(values.mean())
                    spread[model_index, layer_index] = float(
                        values.std(ddof=1) / np.sqrt(values.size)
                        if values.size > 1 else 0.0)
                    count[model_index, layer_index] = values.size
        result[metric] = {'mean': matrix, 'sem': spread, 'count': count}
    return result


def save_grouped_bar(
    output: Path,
    matrix: np.ndarray,
    error: np.ndarray,
    models: List[str],
    layers: List[str],
    title: str,
    ylabel: str,
    dpi: int,
    log_y: bool = False,
) -> None:
    colors = ['#1557a5', '#0096b5', '#3aa76d', '#e1ae17', '#d65a3a', '#7a5aa6']
    x = np.arange(len(layers), dtype=np.float64)
    width = 0.82 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2.0) * width
        plotted = matrix[index]
        if log_y:
            plotted = np.maximum(plotted, 1e-6)
        axis.bar(
            x + offset, plotted, width=width,
            yerr=error[index], capsize=2.0,
            color=colors[index % len(colors)], label=model,
            edgecolor='white', linewidth=0.4)
    axis.set_xticks(x, layers)
    axis.set_xlabel('Backbone stage')
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if log_y:
        axis.set_yscale('log')
    axis.grid(axis='y', color='#d9d9d9', linewidth=0.6, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=min(3, len(models)))
    figure.tight_layout()
    figure.savefig(output.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    figure.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def save_heatmap(
    output: Path,
    matrix: np.ndarray,
    models: List[str],
    layers: List[str],
    title: str,
    dpi: int,
) -> None:
    figure_height = max(3.2, 0.48 * len(models) + 1.8)
    figure, axis = plt.subplots(figsize=(6.4, figure_height))
    image = axis.imshow(matrix, cmap='YlGnBu', aspect='auto')
    axis.set_xticks(np.arange(len(layers)), layers)
    axis.set_yticks(np.arange(len(models)), models)
    axis.set_xlabel('Backbone stage')
    axis.set_title(title)
    finite = matrix[np.isfinite(matrix)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column, row, f'{value:.3f}', ha='center', va='center',
                    color='white' if value > midpoint else 'black', fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.84)
    figure.tight_layout()
    figure.savefig(output.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    figure.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_tsv).expanduser().resolve()
    rows = read_tsv(metrics_path)
    models = parse_order(args.model_order)
    if not models:
        models = list(dict.fromkeys(row['model'] for row in rows))
    layers = parse_order(args.layer_order)
    if not layers:
        layers = list(dict.fromkeys(row['layer'] for row in rows))
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, models, layers)
    specifications = {
        'energy_in_target_box': (
            'CAM energy retained inside the fixed GT box', 'Energy in GT box', False),
        'target_to_background_ratio': (
            'Target-to-background CAM response ratio', 'FG/BG response ratio', True),
        'pointing_game_hit': (
            'Pointing-game accuracy for fixed GT CAM', 'Hit rate', False),
        'top20_iou_with_target': (
            'Top-20% CAM overlap with the fixed GT box', 'Top-20% CAM IoU', False),
        'normalized_entropy': (
            'Spatial entropy of fixed GT CAM', 'Normalized entropy', False),
        'peak_distance_over_box_diagonal': (
            'CAM peak distance from GT center', 'Distance / GT diagonal', False),
    }
    for metric, (title, ylabel, log_y) in specifications.items():
        values = summary[metric]
        save_grouped_bar(
            output / f'{metric}_by_layer',
            values['mean'], values['sem'], models, layers,
            title, ylabel, args.dpi, log_y=log_y)
        save_heatmap(
            output / f'{metric}_heatmap', values['mean'], models, layers,
            title, args.dpi)
    with (output / 'figure_metadata.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'metrics_tsv': str(metrics_path),
            'models': models,
            'layers': layers,
            'error_bars': 'standard error of the mean',
            'figure_metrics': list(specifications),
        }, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'CAM metric figures: {output}')


if __name__ == '__main__':
    main()
