#!/usr/bin/env python3
"""Compare ImageNet, Synthetic5 and RealUW features pairwise by backbone."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import ensure_empty_or_create, read_json, write_json
from .compute_cka import centered_gram, load_features, load_ids


PAIRS = (('ImageNet', 'Synthetic5'), ('ImageNet', 'RealUW'),
         ('Synthetic5', 'RealUW'))
GROUPS = {
    'resnet50': ('res2', 'res3', 'res4', 'res5'),
    'vit_small': ('block1', 'block4', 'block8', 'block12'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models-config', required=True,
                        help='JSON mapping backbone -> source -> extracted model ID')
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    feature_root = Path(args.feature_root).expanduser().resolve()
    mapping = read_json(args.models_config)
    if set(mapping) != set(GROUPS):
        raise ValueError(f'Model mapping must contain {sorted(GROUPS)}')

    rows = []
    for backbone, layers in GROUPS.items():
        models = mapping[backbone]
        if set(models) != {'ImageNet', 'Synthetic5', 'RealUW'}:
            raise ValueError(f'{backbone}: expected ImageNet, Synthetic5 and RealUW')
        ids = {source: load_ids(feature_root, model_id, args.variant)
               for source, model_id in models.items()}
        first = ids['ImageNet']
        for source, values in ids.items():
            if not np.array_equal(first, values):
                raise ValueError(f'{backbone}/{source}: image IDs or order differ')
        grams = {}
        for source, model_id in models.items():
            root = feature_root / 'features' / model_id / args.variant
            features = load_features(root, ','.join(layers), len(first))
            grams[source] = {
                layer: centered_gram(features[layer], args.eps)
                for layer in layers
            }
        for layer in layers:
            for left, right in PAIRS:
                ka, na = grams[left][layer]
                kb, nb = grams[right][layer]
                rows.append({
                    'backbone': backbone, 'layer': layer,
                    'source_a': left, 'source_b': right,
                    'linear_cka': float(np.sum(ka * kb) / (na * nb)),
                    'samples': len(first),
                })

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    fields = ('backbone', 'layer', 'source_a', 'source_b', 'linear_cka', 'samples')
    with (out_dir / 'pretraining_cka.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / 'pretraining_cka_metadata.json', {
        'feature_root': str(feature_root), 'models': mapping,
        'variant': args.variant, 'layers': GROUPS,
        'method': 'linear CKA on centered pooled features',
        'sample_ids_verified': True,
    })
    print(f'CKA results: {out_dir / "pretraining_cka.tsv"}')


if __name__ == '__main__':
    main()
