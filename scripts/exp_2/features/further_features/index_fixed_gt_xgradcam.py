#!/usr/bin/env python3
"""Rebuild and validate a shared fixed-GT XGradCAM index after parallel workers."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cam_common import atomic_write_json, atomic_write_jsonl, parse_csv, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cam-root', required=True)
    parser.add_argument('--models', default='')
    parser.add_argument('--layers', default='res2,res3,res4,res5')
    parser.add_argument('--require-complete', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cam_root).expanduser().resolve()
    rows = []
    for path in sorted((root / 'raw_cam').glob('*/image_*/ann_*/instance.json')):
        value = read_json(path)
        value['instance_metadata_path'] = str(path)
        rows.append(value)
    if not rows:
        raise RuntimeError(f'No fixed-GT CAM metadata found below {root}')
    models = parse_csv(args.models)
    if not models:
        models = list(dict.fromkeys(str(row['model']) for row in rows))
    layers = parse_csv(args.layers)
    by_instance = defaultdict(set)
    counts = Counter()
    missing_files = []
    for row in rows:
        model = str(row['model'])
        counts[model] += 1
        by_instance[(int(row['image_id']), int(row['annotation_id']))].add(model)
        for layer in layers:
            path_value = row.get('layers', {}).get(layer)
            if not path_value or not Path(path_value).is_file():
                missing_files.append({
                    'model': model,
                    'image_id': row['image_id'],
                    'annotation_id': row['annotation_id'],
                    'layer': layer,
                    'path': path_value,
                })
    model_set = set(models)
    incomplete = [
        {
            'image_id': key[0],
            'annotation_id': key[1],
            'present_models': sorted(present),
            'missing_models': sorted(model_set - present),
        }
        for key, present in sorted(by_instance.items())
        if model_set - present
    ]
    atomic_write_jsonl(root / 'raw_cam_index.jsonl', rows)
    summary = {
        'models': models,
        'layers': layers,
        'metadata_rows': len(rows),
        'model_instance_counts': dict(counts),
        'unique_instances': len(by_instance),
        'complete_instances': len(by_instance) - len(incomplete),
        'incomplete_instances': len(incomplete),
        'missing_layer_files': len(missing_files),
        'incomplete_examples': incomplete[:100],
        'missing_file_examples': missing_files[:100],
    }
    atomic_write_json(root / 'raw_cam_index_summary.json', summary)
    print(f'Indexed {len(rows)} model-instance records below {root}')
    print(f'Complete instances: {summary["complete_instances"]}/{len(by_instance)}')
    if missing_files:
        raise RuntimeError(f'{len(missing_files)} indexed layer files are missing')
    if args.require_complete and incomplete:
        raise RuntimeError(
            f'{len(incomplete)} GT instances are incomplete across selected models')


if __name__ == '__main__':
    main()
