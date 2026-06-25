#!/usr/bin/env python3
"""Prepare ImageNet/RUOD data for the original WaterGAN TensorFlow code.

WaterGAN expects three flat datasets:

    air_images/*.png   clean in-air RGB images, usually 640x480
    air_depth/*.mat    matching depth maps
    water_images/*.png real underwater RGB images

This tool uses the per-method sampled ImageNet source tree and MegaDepth PNGs
to build those folders. It also keeps a manifest so generated samples can later
be traced back to ImageNet synsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from scipy.io import savemat

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare WaterGAN ImageNet/RUOD smoke/full data.')
    parser.add_argument('--air-source', required=True,
                        help='ImageNet sampled source, e.g. synthetic_imagenet/watergan/source/train.')
    parser.add_argument('--depth-source', required=True,
                        help='MegaDepth PNG directory mirroring --air-source.')
    parser.add_argument('--water-source', required=True,
                        help='RUOD/real underwater image directory.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--air-limit', type=int, default=1000)
    parser.add_argument('--water-limit', type=int, default=1000)
    parser.add_argument('--air-width', type=int, default=640)
    parser.add_argument('--air-height', type=int, default=480)
    parser.add_argument('--water-width', type=int, default=1360)
    parser.add_argument('--water-height', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    print(f'scanning images: {root}', flush=True)
    images = []
    for path in tqdm(root.rglob('*'), desc=f'scan {root.name}', unit='entry'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    images.sort()
    print(f'found images under {root}: {len(images)}', flush=True)
    return images


def prepare_rgb(src: Path, dst: Path, size: tuple[int, int]) -> None:
    with Image.open(src) as image:
        image = image.convert('RGB')
        image = ImageOps.fit(image, size, method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
        image.save(dst)


def prepare_depth(src: Path, dst: Path, size: tuple[int, int]) -> None:
    with Image.open(src) as depth:
        depth = depth.convert('L')
        depth = ImageOps.fit(depth, size, method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    arr = np.asarray(depth).astype(np.float32) / 255.0
    # Keep several common keys so old code variants can load the file.
    savemat(dst, {
        'depth': arr,
        'dph': arr,
        'D': arr,
        'data': arr,
    })


def main() -> None:
    args = parse_args()
    air_source = Path(args.air_source)
    depth_source = Path(args.depth_source)
    water_source = Path(args.water_source)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{out_dir} exists; pass --overwrite to replace files in-place.')
    (out_dir / 'air_images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'air_depth').mkdir(parents=True, exist_ok=True)
    (out_dir / 'water_images').mkdir(parents=True, exist_ok=True)

    for child in ('air_images', 'air_depth', 'water_images'):
        for old in (out_dir / child).iterdir():
            if old.is_file() or old.is_symlink():
                old.unlink()

    air_images = list_images(air_source)
    water_images = list_images(water_source)
    if args.air_limit > 0:
        air_images = air_images[:args.air_limit]
    if args.water_limit > 0:
        water_images = water_images[:args.water_limit]

    records = []
    missing_depth = []
    air_size = (args.air_width, args.air_height)
    water_size = (args.water_width, args.water_height)

    for index, image_path in enumerate(tqdm(air_images, desc='prepare WaterGAN air/depth', unit='image')):
        rel = image_path.relative_to(air_source)
        depth_path = depth_source / rel.with_suffix('.png')
        if not depth_path.exists():
            missing_depth.append(str(rel).replace('\\', '/'))
            continue
        stem = f'{index:08d}'
        air_dst = out_dir / 'air_images' / f'{stem}.png'
        depth_dst = out_dir / 'air_depth' / f'{stem}.mat'
        prepare_rgb(image_path, air_dst, air_size)
        prepare_depth(depth_path, depth_dst, air_size)
        records.append({
            'index': index,
            'source': str(image_path),
            'depth': str(depth_path),
            'relative': str(rel).replace('\\', '/'),
            'synset': rel.parts[0] if len(rel.parts) > 1 else 'unknown',
            'original_name': image_path.name,
            'air_image': str(air_dst),
            'air_depth': str(depth_dst),
        })

    for index, image_path in enumerate(tqdm(water_images, desc='prepare WaterGAN water', unit='image')):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        prepare_rgb(image_path, water_dst, water_size)

    manifest = out_dir / 'watergan_air_manifest.jsonl'
    with manifest.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = {
        'air_source': str(air_source),
        'depth_source': str(depth_source),
        'water_source': str(water_source),
        'out_dir': str(out_dir),
        'air_limit': args.air_limit,
        'water_limit': args.water_limit,
        'prepared_air': len(records),
        'prepared_water': len(water_images),
        'missing_depth': len(missing_depth),
        'missing_depth_samples': missing_depth[:20],
        'air_size': list(air_size),
        'water_size': list(water_size),
        'manifest': str(manifest),
    }
    summary_path = out_dir / 'prepare_watergan_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
