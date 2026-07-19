#!/usr/bin/env python3
"""Rebalance unfinished WaterGAN inference while preserving completed batches."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


FAKE_RE = re.compile(r'^fake_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.I)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-shard-root', required=True)
    parser.add_argument('--base-results-root', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--base-shards', type=int, required=True)
    parser.add_argument('--resume-shards', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--reset', action='store_true')
    return parser.parse_args()


def load_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def complete_prefix(results_dir, batch_size):
    batches = {}
    for path in results_dir.glob('fake_*'):
        match = FAKE_RE.match(path.name)
        if not match or int(match.group(1)) != 0:
            continue
        item = int(match.group(2))
        batch = int(match.group(3))
        batches.setdefault(batch, set()).add(item)
    required = set(range(batch_size))
    complete_batches = 0
    while batches.get(complete_batches) == required:
        complete_batches += 1
    return complete_batches * batch_size


def distribute(total, shards, batch_size):
    if total % batch_size:
        raise RuntimeError(
            'Pending count {} is not divisible by {}'.format(total, batch_size)
        )
    batches, extra = divmod(total // batch_size, shards)
    return [
        (batches + (1 if index < extra else 0)) * batch_size
        for index in range(shards)
    ]


def link(source, destination):
    destination.symlink_to(source.resolve())


def main():
    args = parse_args()
    base_root = Path(args.base_shard_root).resolve()
    results_root = Path(args.base_results_root).resolve()
    out_root = Path(args.out_root).resolve()
    if args.reset and out_root.exists():
        shutil.rmtree(str(out_root))
    temporary = out_root.with_name(
        '.{}.tmp.{}'.format(out_root.name, os.getpid())
    )
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)
    completed_root = temporary / 'completed_manifests'
    completed_root.mkdir()

    pending = []
    base_plan = []
    water_files = None
    for index in range(args.base_shards):
        name = 'shard{}of{}'.format(index, args.base_shards)
        shard_root = base_root / name
        manifest = shard_root / 'watergan_air_manifest.jsonl'
        results = results_root / name
        records = load_records(manifest)
        completed = min(complete_prefix(results, args.batch_size), len(records))
        completed -= completed % args.batch_size
        completed_manifest = completed_root / (name + '.jsonl')
        with completed_manifest.open('w', encoding='utf-8') as handle:
            for record in records[:completed]:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        pending.extend(records[completed:])
        base_plan.append({
            'name': name,
            'records': len(records),
            'completed': completed,
            'pending': len(records) - completed,
            'results_dir': str(results),
            'completed_manifest': str(out_root / 'completed_manifests' / (name + '.jsonl')),
        })
        if water_files is None:
            water_files = sorted(
                item for item in (shard_root / 'water_images').iterdir()
                if item.is_file()
            )
    if not water_files:
        raise RuntimeError('No water images found in base shards')

    sizes = distribute(len(pending), args.resume_shards, args.batch_size)
    offset = 0
    resume_plan = []
    for shard_index, count in enumerate(sizes):
        name = 'shard{}of{}'.format(shard_index, args.resume_shards)
        shard = temporary / name
        for directory in ('air_images', 'air_depth', 'water_images'):
            (shard / directory).mkdir(parents=True, exist_ok=True)
        for water_index, source in enumerate(water_files):
            destination = shard / 'water_images' / (
                '{:08d}'.format(water_index) + source.suffix.lower()
            )
            link(source, destination)
        selected = pending[offset:offset + count]
        with (shard / 'watergan_air_manifest.jsonl').open(
            'w', encoding='utf-8'
        ) as handle:
            for local_index, original in enumerate(selected):
                record = dict(original)
                air_source = Path(original['air_image'])
                depth_source = Path(original['air_depth'])
                stem = '{:08d}'.format(local_index)
                air_destination = shard / 'air_images' / (
                    stem + air_source.suffix.lower()
                )
                depth_destination = shard / 'air_depth' / (
                    stem + depth_source.suffix.lower()
                )
                link(air_source, air_destination)
                link(depth_source, depth_destination)
                record['resume_source_index'] = record.get('index')
                record['index'] = local_index
                record['air_image'] = str(out_root / name / 'air_images' / air_destination.name)
                record['air_depth'] = str(out_root / name / 'air_depth' / depth_destination.name)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        resume_plan.append({
            'name': name,
            'start': offset,
            'end': offset + count,
            'count': count,
        })
        offset += count

    plan = {
        'base_shard_root': str(base_root),
        'base_results_root': str(results_root),
        'batch_size': args.batch_size,
        'base_shards': args.base_shards,
        'resume_shards': args.resume_shards,
        'completed_total': sum(item['completed'] for item in base_plan),
        'pending_total': len(pending),
        'water_count': len(water_files),
        'base': base_plan,
        'resume': resume_plan,
    }
    (temporary / 'resume_plan.json').write_text(
        json.dumps(plan, indent=2), encoding='utf-8'
    )
    if out_root.exists():
        shutil.rmtree(str(out_root))
    temporary.rename(out_root)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
