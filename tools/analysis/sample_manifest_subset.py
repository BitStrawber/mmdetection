#!/usr/bin/env python3
"""Create a deterministic, re-indexed subset of an existing sample manifest."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tools.exp_2.backbone_analysis.common import (
    ensure_empty_or_create,
    read_jsonl,
    validate_sample_order,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--samples', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError('--samples must be positive')
    source = Path(args.manifest).expanduser().resolve()
    rows = read_jsonl(source)
    validate_sample_order(rows)
    if args.samples > len(rows):
        raise ValueError(
            f'Requested {args.samples} CAM samples from only {len(rows)} parent samples')
    output = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    selected = random.Random(args.seed).sample(rows, args.samples)
    selected.sort(key=lambda row: int(row['sample_index']))
    reindexed = []
    for subset_index, source_row in enumerate(selected):
        row = dict(source_row)
        row['parent_sample_index'] = int(source_row['sample_index'])
        row['sample_index'] = subset_index
        reindexed.append(row)
    write_jsonl(output / 'manifest.jsonl', reindexed)
    write_json(output / 'selection.json', {
        'parent_manifest': str(source),
        'parent_samples': len(rows),
        'samples': args.samples,
        'seed': args.seed,
        'selected_parent_sample_indices': [
            int(row['parent_sample_index']) for row in reindexed],
        'selected_image_ids': [int(row['image_id']) for row in reindexed],
    })
    print(f'Selected {len(reindexed)}/{len(rows)} parent samples into {output}')


if __name__ == '__main__':
    main()
