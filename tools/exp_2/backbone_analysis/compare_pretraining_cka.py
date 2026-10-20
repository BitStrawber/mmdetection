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
MODEL_IDS = {
    'resnet50': {'ImageNet': 'imagenet_r50',
                 'Synthetic5': 'synthetic5_r50', 'RealUW': 'realuw_r50'},
    'vit_small': {'ImageNet': 'imagenet_vits',
                  'Synthetic5': 'synthetic5_vits', 'RealUW': 'realuw_vits'},
}


def resolve_mapping(config: dict) -> dict:
    if set(config) == set(GROUPS):
        return config
    if set(config) == {'models'} and isinstance(config['models'], list):
        ids = [str(model['id']) for model in config['models']]
        expected = {model_id for group in MODEL_IDS.values()
                    for model_id in group.values()}
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError('Six-model config must contain exactly the expected model IDs')
        return MODEL_IDS
    raise ValueError(f'Model mapping must contain {sorted(GROUPS)}')


def validate_provenance(feature_root: Path, mapping: dict, variant: str) -> dict:
    records = {}
    all_ids = None
    manifest_hash = None
    pooling = None
    for backbone, models in mapping.items():
        config_hash = None
        missing_hash = None
        for source, model_id in models.items():
            record = read_json(
                feature_root / 'metadata' / model_id / 'extraction_provenance.json')
            report = record['model_load_report']
            ids = load_ids(feature_root, model_id, variant)
            if all_ids is None:
                all_ids = ids
            elif not np.array_equal(all_ids, ids):
                raise ValueError(f'{model_id}: RUOD image IDs or order differ')
            if manifest_hash is None:
                manifest_hash = record['manifest_sha256']
                pooling = record['pooling']
            elif (manifest_hash != record['manifest_sha256'] or
                  pooling != record['pooling']):
                raise ValueError(f'{model_id}: manifest or pooling method differs')
            if config_hash is None:
                config_hash = record['config_sha256']
            elif config_hash != record['config_sha256']:
                raise ValueError(f'{backbone}/{source}: model config differs')
            if report.get('checkpoint_epoch') != 100:
                raise ValueError(f'{model_id}: not a 100e checkpoint')
            if report.get('required_match_ratio') != 1.0:
                raise ValueError(f'{model_id}: incomplete required backbone weights')
            allowed = report.get('allowed_missing_keys', [])
            if backbone == 'resnet50' and allowed:
                raise ValueError(f'{model_id}: unexpected missing R50 weights')
            if backbone == 'vit_small':
                if len(allowed) != 24 or report.get('initialization_seed') is None:
                    raise ValueError(f'{model_id}: unexpected ViT-S initialization')
                digest = report.get('allowed_missing_sha256')
                if not digest:
                    raise ValueError(f'{model_id}: missing initialization digest')
                if missing_hash is None:
                    missing_hash = digest
                elif missing_hash != digest:
                    raise ValueError(
                        f'{backbone}/{source}: relative-position initialization differs')
            records[model_id] = record
    return {'manifest_sha256': manifest_hash, 'pooling': pooling,
            'image_count': len(all_ids), 'model_records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models-config', required=True,
                        help='Pairwise mapping JSON or six-model extraction config')
    parser.add_argument('--variant', default='clean')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    feature_root = Path(args.feature_root).expanduser().resolve()
    mapping = resolve_mapping(read_json(args.models_config))
    provenance = validate_provenance(feature_root, mapping, args.variant)

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
                if not np.isfinite(rows[-1]['linear_cka']) or not (
                        -1e-8 <= rows[-1]['linear_cka'] <= 1 + 1e-8):
                    raise ValueError(f'Invalid CKA for {backbone}/{layer}/{left}-{right}')

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
        'provenance': provenance,
    })
    print(f'CKA results: {out_dir / "pretraining_cka.tsv"}')


if __name__ == '__main__':
    main()
