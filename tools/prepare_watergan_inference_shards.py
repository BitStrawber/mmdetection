#!/usr/bin/env python3
"""Create zero-copy, batch-aligned WaterGAN inference shards."""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--reset', action='store_true')
    return parser.parse_args()


def list_files(path):
    return sorted(item for item in path.iterdir() if item.is_file())


def shard_sizes(total, num_shards, batch_size):
    if total % batch_size:
        raise RuntimeError(
            'Manifest size {} is not divisible by batch size {}'.format(
                total, batch_size
            )
        )
    total_batches = total // batch_size
    base, extra = divmod(total_batches, num_shards)
    return [
        (base + (1 if index < extra else 0)) * batch_size
        for index in range(num_shards)
    ]


def valid_existing(path, expected, data_root, shard_index, num_shards,
                   batch_size, start):
    summary = path / 'shard_summary.json'
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    expected_summary = {
        'source_data_root': str(data_root),
        'shard_index': shard_index,
        'num_shards': num_shards,
        'batch_size': batch_size,
        'start': start,
        'end': start + expected,
        'count': expected,
    }
    if any(payload.get(key) != value
           for key, value in expected_summary.items()):
        return False
    for name in ('air_images', 'air_depth', 'water_images'):
        if len(list_files(path / name)) != expected:
            return False
    manifest = path / 'watergan_air_manifest.jsonl'
    if not manifest.is_file():
        return False
    with manifest.open('r', encoding='utf-8') as handle:
        return sum(1 for _ in handle) == expected


def link_file(source, destination):
    destination.symlink_to(source.resolve())


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    manifest_path = data_root / 'watergan_air_manifest.jsonl'
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    inputs = {
        name: list_files(data_root / name)
        for name in ('air_images', 'air_depth', 'water_images')
    }
    total = len(records)
    for name, files in inputs.items():
        if len(files) < total:
            raise RuntimeError(
                '{} has {} files, but manifest has {}'.format(name, len(files), total)
            )

    sizes = shard_sizes(total, args.num_shards, args.batch_size)
    out_root.mkdir(parents=True, exist_ok=True)
    offset = 0
    for shard_index, count in enumerate(sizes):
        shard = out_root / 'shard{}of{}'.format(shard_index, args.num_shards)
        if args.reset and shard.exists():
            shutil.rmtree(str(shard))
        if valid_existing(
            shard, count, data_root, shard_index, args.num_shards,
            args.batch_size, offset
        ):
            print('reuse {}: start={}, count={}'.format(shard, offset, count))
            offset += count
            continue

        temporary = out_root / '.shard{}of{}.tmp.{}'.format(
            shard_index, args.num_shards, os.getpid()
        )
        if temporary.exists():
            shutil.rmtree(str(temporary))
        temporary.mkdir(parents=True)
        for name in inputs:
            (temporary / name).mkdir()

        shard_records = []
        for local_index in range(count):
            global_index = offset + local_index
            stem = '{:08d}'.format(local_index)
            destinations = {}
            for name, files in inputs.items():
                source = files[global_index]
                destination = temporary / name / (stem + source.suffix.lower())
                link_file(source, destination)
                destinations[name] = destination
            record = dict(records[global_index])
            record['global_index'] = record.get('index', global_index)
            record['index'] = local_index
            record['air_image'] = str(
                shard / 'air_images' / destinations['air_images'].name
            )
            record['air_depth'] = str(
                shard / 'air_depth' / destinations['air_depth'].name
            )
            shard_records.append(record)

        with (temporary / 'watergan_air_manifest.jsonl').open(
            'w', encoding='utf-8'
        ) as handle:
            for record in shard_records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        summary = {
            'source_data_root': str(data_root),
            'shard_index': shard_index,
            'num_shards': args.num_shards,
            'batch_size': args.batch_size,
            'start': offset,
            'end': offset + count,
            'count': count,
        }
        (temporary / 'shard_summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8'
        )
        if shard.exists():
            shutil.rmtree(str(shard))
        temporary.rename(shard)
        print('created {}: start={}, count={}'.format(shard, offset, count))
        offset += count

    print('total={}, sizes={}'.format(total, sizes))


if __name__ == '__main__':
    main()
