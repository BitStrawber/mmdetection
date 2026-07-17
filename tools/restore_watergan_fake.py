#!/usr/bin/env python3
"""Restore WaterGAN fake_*.png outputs to ImageNet-style class folders.

The WaterGAN test code writes flat files named like:
    fake_<epoch>_<item_in_batch>_<batch_index>.png

The prepared WaterGAN dataset records source order in watergan_air_manifest.jsonl.
This script maps each generated fake image back to that manifest order and writes:
    out_dir/<synset>/<original_stem>.png
"""

import argparse
import json
import re
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FAKE_RE = re.compile(r'^fake_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description='Restore WaterGAN fake outputs.')
    parser.add_argument('--manifest', required=True,
                        help='watergan_air_manifest.jsonl from prepare_watergan_imagenet_ruod_dataset.py')
    parser.add_argument('--results-dir', required=True,
                        help='WaterGAN flat results directory containing fake_*.png')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size used during WaterGAN generation. 0 infers from max item index + 1.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def collect_fake_outputs(results_dir: Path, batch_size: int):
    parsed = []
    bad_names = []
    max_item = -1
    for path in results_dir.glob('fake_*'):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = FAKE_RE.match(path.name)
        if not match:
            bad_names.append(path.name)
            continue
        epoch = int(match.group(1))
        item = int(match.group(2))
        batch = int(match.group(3))
        max_item = max(max_item, item)
        parsed.append((epoch, item, batch, path))

    effective_batch = batch_size if batch_size > 0 else max_item + 1
    if effective_batch <= 0:
        effective_batch = 1

    outputs = {}
    duplicate_indices = []
    for epoch, item, batch, path in parsed:
        index = batch * effective_batch + item
        if index in outputs:
            duplicate_indices.append(index)
            continue
        outputs[index] = path

    return outputs, effective_batch, bad_names, duplicate_indices


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest)
    outputs, effective_batch, bad_names, duplicate_indices = collect_fake_outputs(results_dir, args.batch_size)

    written = 0
    skipped = 0
    missing = []
    for idx, record in enumerate(tqdm(records, desc='restore WaterGAN fake', unit='image')):
        generated = outputs.get(idx)
        if generated is None:
            missing.append(idx)
            continue
        synset = record.get('synset') or 'unknown'
        original_name = Path(record.get('original_name') or f'{idx:08d}.png')
        dst_dir = out_dir / synset
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'{original_name.stem}{generated.suffix.lower()}'
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(generated, dst)
        written += 1

    summary = {
        'manifest': str(manifest),
        'results_dir': str(results_dir),
        'out_dir': str(out_dir),
        'records': len(records),
        'outputs_found': len(outputs),
        'effective_batch_size': effective_batch,
        'written': written,
        'skipped_existing': skipped,
        'missing': len(missing),
        'missing_samples': missing[:20],
        'bad_names': len(bad_names),
        'bad_name_samples': bad_names[:20],
        'duplicate_indices': len(duplicate_indices),
        'duplicate_index_samples': duplicate_indices[:20],
    }
    summary_path = out_dir / 'restore_watergan_fake_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()