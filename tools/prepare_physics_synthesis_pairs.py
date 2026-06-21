#!/usr/bin/env python3
"""Prepare MegaDepth image/depth pairs for SyreaNet or WaterGAN.

MegaDepth output is expected to mirror ``--image-dir`` and use ``.png`` depth
files. The generated names flatten the original relative path so that legacy
projects which only scan one directory can consume the pairs safely.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('syreanet', 'watergan'), required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--depth-dir', required=True,
                        help='MegaDepth root that mirrors --image-dir.')
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many pairs; 0 means all.')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def output_name(relative_path):
    stem = str(relative_path.with_suffix('')).replace('/', '__').replace('\\', '__')
    return stem + '.png'


def depth_path(depth_root, relative_path):
    return (depth_root / relative_path).with_suffix('.png')


def resize_pair(image, depth, width, height):
    if not width or not height:
        return image, depth
    target = (width, height)
    resampling = getattr(Image, 'Resampling', Image)
    return (image.resize(target, resampling.LANCZOS),
            depth.resize(target, resampling.NEAREST))


def main():
    args = parse_args()
    image_root = Path(args.image_dir).resolve()
    depth_root = Path(args.depth_dir).resolve()
    out_root = Path(args.out_root).resolve()
    if not image_root.is_dir() or not depth_root.is_dir():
        raise FileNotFoundError('Both --image-dir and --depth-dir must exist.')
    if args.mode == 'watergan' and (not args.width or not args.height):
        args.width, args.height = 640, 480

    if args.mode == 'syreanet':
        image_out, depth_out = out_root / 'images', out_root / 'depth'
    else:
        image_out, depth_out = out_root / 'air_images', out_root / 'air_depth'
    image_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in image_root.rglob('*')
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[:args.limit]

    manifest_path = out_root / 'pair_manifest.jsonl'
    written = skipped = missing_depth = 0
    with manifest_path.open('w', encoding='utf-8') as manifest:
        for image_path in tqdm(images, desc=f'prepare {args.mode}', unit='image'):
            relative = image_path.relative_to(image_root)
            source_depth = depth_path(depth_root, relative)
            if not source_depth.is_file():
                missing_depth += 1
                continue
            name = output_name(relative)
            image_target = image_out / name
            depth_target = depth_out / (name if args.mode == 'syreanet' else Path(name).with_suffix('.mat'))
            if image_target.exists() and depth_target.exists() and not args.overwrite:
                skipped += 1
                record = {'source_image': str(relative), 'image': name,
                          'depth': depth_target.name, 'status': 'existing'}
                manifest.write(json.dumps(record) + '\n')
                continue

            with Image.open(image_path) as image, Image.open(source_depth) as depth:
                image = image.convert('RGB')
                depth = depth.convert('L')
                image, depth = resize_pair(image, depth, args.width, args.height)
                image.save(image_target, format='PNG')

                if args.mode == 'syreanet':
                    depth.save(depth_target, format='PNG')
                else:
                    try:
                        from scipy.io import savemat
                    except ImportError as error:
                        raise RuntimeError('WaterGAN preparation requires scipy.') from error
                    # WaterGAN's read_depth() accesses the ``depth`` MAT variable.
                    savemat(depth_target, {'depth': np.asarray(depth, dtype=np.float32) / 255.0})

            written += 1
            record = {'source_image': str(relative), 'image': name,
                      'depth': depth_target.name, 'status': 'written'}
            manifest.write(json.dumps(record) + '\n')

    summary = {
        'mode': args.mode, 'image_dir': str(image_root), 'depth_dir': str(depth_root),
        'out_root': str(out_root), 'size': [args.width, args.height],
        'total_considered': len(images), 'written': written, 'skipped_existing': skipped,
        'missing_depth': missing_depth,
    }
    (out_root / 'pair_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'pair manifest: {manifest_path}')


if __name__ == '__main__':
    main()
