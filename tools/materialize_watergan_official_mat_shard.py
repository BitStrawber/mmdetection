#!/usr/bin/env python3
"""Materialize one WaterGAN inference shard with official single-key MAT depth."""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-shard', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--reset', action='store_true')
    return parser.parse_args()


def list_files(path, suffixes=None):
    files = sorted(item for item in path.iterdir() if item.is_file())
    if suffixes:
        files = [item for item in files if item.suffix.lower() in suffixes]
    return files


def valid_mat(path):
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        values = loadmat(str(path))
        keys = [key for key in values if not key.startswith('__')]
        return (
            keys == ['depth']
            and values['depth'].dtype == np.float32
            and values['depth'].shape == (480, 640)
            and np.isfinite(values['depth']).all()
        )
    except (OSError, TypeError, ValueError):
        return False


def convert_one(task):
    source, destination = task
    if valid_mat(destination):
        return 'reused'
    with Image.open(str(source)) as image:
        image.load()
        if image.size != (640, 480):
            raise RuntimeError(
                'unexpected depth size {} for {}'.format(image.size, source)
            )
        depth = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    if not np.isfinite(depth).all():
        raise RuntimeError('non-finite depth values: {}'.format(source))
    temporary = destination.with_name(
        '.{}.{}.tmp'.format(destination.name, os.getpid())
    )
    savemat(str(temporary), {'depth': depth}, appendmat=False, do_compression=False)
    os.replace(str(temporary), str(destination))
    return 'written'


def link_file(source, destination):
    destination.symlink_to(source.resolve())


def main():
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit('--workers must be positive')
    if args.limit < 0:
        raise SystemExit('--limit must be non-negative')

    source = Path(args.source_shard).resolve()
    out_dir = Path(args.out_dir).resolve()
    source_manifest = source / 'watergan_air_manifest.jsonl'
    records = [
        json.loads(line)
        for line in source_manifest.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    air_images = list_files(
        source / 'air_images', {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    )
    depth_images = list_files(source / 'air_depth', {'.png'})
    total = len(records)
    if len(air_images) != total or len(depth_images) != total:
        raise RuntimeError(
            'source shard mismatch: manifest={}, air={}, depth={}'.format(
                total, len(air_images), len(depth_images)
            )
        )
    count = min(args.limit, total) if args.limit else total
    records = records[:count]
    air_images = air_images[:count]
    depth_images = depth_images[:count]

    if args.reset and out_dir.exists():
        shutil.rmtree(str(out_dir))
    if out_dir.exists():
        summary_path = out_dir / 'official_mat_summary.json'
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            existing_air = list_files(out_dir / 'air_images')
            existing_depth = list_files(out_dir / 'air_depth', {'.mat'})
            if (
                summary.get('source_shard') == str(source)
                and summary.get('count') == count
                and len(existing_air) == count
                and len(existing_depth) == count
                and all(valid_mat(path) for path in existing_depth)
            ):
                print('reuse official MAT shard: {} ({})'.format(out_dir, count))
                return
        raise RuntimeError(
            'incomplete or incompatible output exists; use --reset: {}'.format(
                out_dir
            )
        )

    temporary = out_dir.with_name('.{}.tmp.{}'.format(out_dir.name, os.getpid()))
    if temporary.exists():
        shutil.rmtree(str(temporary))
    (temporary / 'air_images').mkdir(parents=True)
    (temporary / 'air_depth').mkdir()

    for index, image_path in enumerate(air_images):
        link_file(
            image_path,
            temporary / 'air_images' / ('{:08d}'.format(index) + image_path.suffix.lower()),
        )

    tasks = [
        (depth_path, temporary / 'air_depth' / '{:08d}.mat'.format(index))
        for index, depth_path in enumerate(depth_images)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        statuses = list(executor.map(convert_one, tasks))

    with (temporary / 'watergan_air_manifest.jsonl').open(
        'w', encoding='utf-8'
    ) as handle:
        for index, record in enumerate(records):
            item = dict(record)
            item['index'] = index
            item['air_image'] = str(out_dir / 'air_images' / '{:08d}{}'.format(
                index, air_images[index].suffix.lower()
            ))
            item['air_depth'] = str(out_dir / 'air_depth' / '{:08d}.mat'.format(index))
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')

    summary = {
        'source_shard': str(source),
        'out_dir': str(out_dir),
        'count': count,
        'workers': args.workers,
        'mat_layout': 'official',
        'mat_keys': ['depth'],
        'depth_shape': [480, 640],
        'written': statuses.count('written'),
        'reused': statuses.count('reused'),
    }
    (temporary / 'official_mat_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
