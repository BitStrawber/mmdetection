#!/usr/bin/env python3
"""Jointly project saved backbone embeddings with PCA and t-SNE."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np

from .common import ensure_empty_or_create, parse_csv, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--layer', required=True)
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--pca-components', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    models = parse_csv(args.models)
    rows = read_jsonl(args.manifest)
    root = Path(args.feature_root).expanduser().resolve()
    matrices = []
    identities = []
    for model in models:
        path = root / 'features' / model / args.variant / f'{args.layer}.npy'
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.ndim != 2 or value.shape[0] != len(rows):
            raise ValueError(f'Invalid embedding matrix: {path}, shape={value.shape}')
        matrices.append(normalize(value, norm='l2'))
        identities.extend((model, index) for index in range(len(rows)))
    combined = np.concatenate(matrices, axis=0)
    components = min(args.pca_components, combined.shape[0] - 1, combined.shape[1])
    reduced = PCA(n_components=components, random_state=args.seed).fit_transform(combined)
    if args.perplexity >= len(reduced):
        raise ValueError('t-SNE perplexity must be smaller than the sample count')
    tsne_options = dict(
        n_components=2,
        perplexity=args.perplexity,
        init='pca',
        learning_rate='auto',
        random_state=args.seed,
    )
    if 'max_iter' in inspect.signature(TSNE).parameters:
        tsne_options['max_iter'] = args.iterations
    else:
        tsne_options['n_iter'] = args.iterations
    coordinates = TSNE(**tsne_options).fit_transform(reduced)

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    output_rows = []
    for coordinate, (model, index) in zip(coordinates, identities):
        row = rows[index]
        class_ids = [int(value) for value in row.get('class_ids', [])]
        output_rows.append({
            'model': model,
            'sample_index': index,
            'image_id': int(row['image_id']),
            'x': float(coordinate[0]),
            'y': float(coordinate[1]),
            'class_ids': json.dumps(class_ids),
            'primary_class_id': class_ids[0] if class_ids else -1,
        })
    with (out_dir / 'tsne_coordinates.tsv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(output_rows)
    np.save(out_dir / 'tsne_coordinates.npy', coordinates.astype(np.float32),
            allow_pickle=False)
    write_json(out_dir / 'tsne_metadata.json', {
        'models': models,
        'layer': args.layer,
        'variant': args.variant,
        'seed': args.seed,
        'perplexity': args.perplexity,
        'pca_components': components,
        'iterations': args.iterations,
        'joint_projection': True,
    })
    print(f't-SNE coordinates: {out_dir}')


if __name__ == '__main__':
    main()
