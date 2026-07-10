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


import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy.io import savemat

BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC

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
    parser.add_argument('--air-limit', type=int, default=1000,
                        help='Maximum number of air/source images after optional per-class sampling; 0 keeps all selected images.')
    parser.add_argument('--water-limit', type=int, default=1000,
                        help='Maximum number of RUOD/water images before optional repeating; 0 keeps all water images.')
    parser.add_argument('--air-per-class', type=int, default=0,
                        help='Randomly select up to this many source images per synset/class before --air-limit. 0 disables class-balanced sampling.')
    parser.add_argument('--water-repeat-to', type=int, default=0,
                        help='Repeat RUOD/water images in deterministic order until this count. 0 disables repeating.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed used for class-balanced source sampling and final source order shuffle.')
    parser.add_argument('--air-width', type=int, default=640)
    parser.add_argument('--air-height', type=int, default=480)
    parser.add_argument('--water-width', type=int, default=1360)
    parser.add_argument('--water-height', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    print(f'scanning images: {root}', flush=True)
    images = []
    for path in tqdm(root.rglob('*'), desc=f'scan {root.name}', unit='entry'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    images.sort()
    print(f'found images under {root}: {len(images)}', flush=True)
    return images


def select_air_images(images: List[Path], root: Path, per_class: int, seed: int) -> List[Path]:
    if per_class <= 0:
        return images

    by_class = {}  # type: Dict[str, List[Path]]
    for path in images:
        rel = path.relative_to(root)
        class_name = rel.parts[0] if len(rel.parts) > 1 else '__root__'
        by_class.setdefault(class_name, []).append(path)

    rng = random.Random(seed)
    selected = []  # type: List[Path]
    for class_name in sorted(by_class):
        class_images = sorted(by_class[class_name])
        if len(class_images) > per_class:
            class_images = rng.sample(class_images, per_class)
            class_images.sort()
        selected.extend(class_images)

    # Shuffle after balanced selection so WaterGAN's prefix-based train loop
    # still sees all synsets if a smaller train_size is used.
    rng.shuffle(selected)
    print(
        f'class-balanced air sampling: classes={len(by_class)}, '
        f'per_class={per_class}, selected={len(selected)}',
        flush=True,
    )
    return selected


def repeat_images(images: List[Path], target_count: int) -> List[Path]:
    if target_count <= 0 or not images or len(images) >= target_count:
        return images
    repeated = [images[index % len(images)] for index in range(target_count)]
    print(
        f'repeated water images: original={len(images)}, target={len(repeated)}',
        flush=True,
    )
    return repeated

def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(str(src), str(dst))
        return
    except OSError:
        pass
    try:
        os.link(str(src), str(dst))
        return
    except OSError:
        pass
    # Last-resort fallback for filesystems that disallow links.
    with src.open('rb') as fsrc, dst.open('wb') as fdst:
        while True:
            chunk = fsrc.read(1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)


def prepare_rgb(src: Path, dst: Path, size: Tuple[int, int]) -> None:
    with Image.open(src) as image:
        image = image.convert('RGB')
        image = ImageOps.fit(image, size, method=BICUBIC, centering=(0.5, 0.5))
        image.save(dst)


def prepare_depth(src: Path, dst: Path, size: Tuple[int, int]) -> None:
    with Image.open(src) as depth:
        depth = depth.convert('L')
        depth = ImageOps.fit(depth, size, method=BICUBIC, centering=(0.5, 0.5))
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
    air_images = select_air_images(air_images, air_source, args.air_per_class, args.seed)
    selected_air_before_limit = len(air_images)
    if args.air_limit > 0:
        air_images = air_images[:args.air_limit]
    if args.water_limit > 0:
        water_images = water_images[:args.water_limit]
    selected_water_before_repeat = len(water_images)
    water_images = repeat_images(water_images, args.water_repeat_to)

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

    linked_water = 0
    for index, image_path in enumerate(tqdm(water_images, desc='prepare WaterGAN water', unit='image')):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        if args.water_repeat_to > 0 and selected_water_before_repeat > 0 and index >= selected_water_before_repeat:
            base_dst = out_dir / 'water_images' / f'{index % selected_water_before_repeat:08d}.png'
            link_or_copy(base_dst, water_dst)
            linked_water += 1
        else:
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
        'air_per_class': args.air_per_class,
        'water_repeat_to': args.water_repeat_to,
        'seed': args.seed,
        'selected_air_before_limit': selected_air_before_limit,
        'selected_water_before_repeat': selected_water_before_repeat,
        'prepared_air': len(records),
        'prepared_water': len(water_images),
        'linked_or_reused_water': linked_water,
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
