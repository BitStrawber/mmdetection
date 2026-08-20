#!/usr/bin/env python3
"""Render a compact paper-style same-layer CKA heatmap from existing TSV data.

The script is rendering-only: it preserves the original CKA output and writes
new PNG/PDF files to an explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


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
    parser.add_argument('--cka-tsv', required=True, help='same_layer_cka.tsv from a completed CKA run')
    parser.add_argument('--models', default='', help='Optional comma-separated row order')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--out-dir', required=True, help='New output directory; source files are never changed')
    parser.add_argument('--prefix', default='paper_same_layer_cka')
    parser.add_argument('--vmin', type=float, default=0.40)
    parser.add_argument('--vmax', type=float, default=1.00)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory is non-empty: {path}; use a new directory or --overwrite')
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace('_', ' '))


def main() -> None:
    args = parse_args()
    source = Path(args.cka_tsv).expanduser().resolve()
    with source.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    required = {'model', 'reference_model', 'layer', 'linear_cka', 'self_comparison'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Expected same-layer CKA columns {sorted(required)} in {source}')
    if any(row['self_comparison'].strip().lower() == 'true' for row in rows):
        raise ValueError('Paper CKA input must exclude self comparisons; CKA(X, X)=1 is not experimental evidence')
    layers = parse_csv(args.layers)
    observed = list(dict.fromkeys(row['model'] for row in rows))
    models = parse_csv(args.models) or observed
    if set(models) != set(observed):
        raise ValueError(f'--models must contain exactly the TSV models: observed={observed}')
    references = set(row['reference_model'] for row in rows)
    if len(references) != 1:
        raise ValueError(f'Expected one reference model, got {sorted(references)}')
    reference = next(iter(references))
    lookup: Dict[tuple, float] = {(row['model'], row['layer']): float(row['linear_cka']) for row in rows}
    missing = [(model, layer) for model in models for layer in layers if (model, layer) not in lookup]
    if missing:
        raise KeyError(f'Missing CKA value, first missing key: {missing[0]}')
    matrix = np.asarray([[lookup[(model, layer)] for layer in layers] for model in models], dtype=np.float64)
    output = prepare_output(Path(args.out_dir), args.overwrite)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.45, 0.64 * len(models) + 2.15))
    image = axis.imshow(matrix, vmin=args.vmin, vmax=args.vmax, cmap='YlGnBu', aspect='auto')
    axis.set_xticks(range(len(layers)), [item.upper() for item in layers])
    axis.set_yticks(range(len(models)), [display_name(model) for model in models])
    axis.tick_params(axis='both', labelsize=9)
    axis.set_xlabel(f'Same semantic layer of {display_name(reference)}', fontsize=10)
    axis.set_ylabel('Comparison pretraining', fontsize=10)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            color = 'white' if value < (args.vmin + args.vmax) / 2 else '#111111'
            axis.text(column_index, row_index, f'{value:.3f}', ha='center', va='center', fontsize=9, color=color)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.052, pad=0.045)
    colorbar.set_label('Linear CKA', fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    axis.set_title('Same-layer representation similarity', fontsize=12, fontweight='semibold', pad=8)
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(output / f'{args.prefix}.{suffix}', dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    metadata = {
        'source_cka_tsv': str(source), 'reference_model': reference, 'models': models, 'layers': layers,
        'color_range': [args.vmin, args.vmax],
        'self_comparisons': 'rejected; linear CKA(X, X)=1 by definition',
        'layout': 'compact same-layer heatmap with abbreviated model names',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Paper CKA heatmap: {output}')


if __name__ == '__main__':
    main()
