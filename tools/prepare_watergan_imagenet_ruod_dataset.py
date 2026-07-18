#!/usr/bin/env python3
"""Prepare ImageNet/RUOD data for the original WaterGAN TensorFlow code.

WaterGAN expects three flat datasets:

    air_images/*.png   clean in-air RGB images, usually 640x480
    air_depth/*.mat    matching depth maps in the original code path, or
    air_depth/*.png    matching depth maps after patch_watergan_tf15_compat.sh
    water_images/*.png real underwater RGB images

This tool uses the per-method sampled ImageNet source tree and MegaDepth PNGs
to build those folders. It also keeps a manifest so generated samples can later
be traced back to ImageNet synsets.
"""


import argparse
import concurrent.futures
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps

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
    parser.add_argument('--depth-format', choices=('mat', 'png'), default='mat',
                        help='Output depth storage. mat matches the original WaterGAN code; png is much smaller and requires patch_watergan_tf15_compat.sh.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel image preparation workers. Use 1 for sequential preparation.')
    parser.add_argument('--verify-existing', action='store_true',
                        help='Decode existing outputs before skipping them in resume mode.')
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument('--overwrite', action='store_true',
                             help='Remove existing prepared files and rebuild from the beginning.')
    output_mode.add_argument('--resume', action='store_true',
                             help='Keep valid existing outputs and prepare only missing files.')
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
    tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp')
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        os.symlink(str(src), str(tmp))
        os.replace(str(tmp), str(dst))
        return
    except OSError:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    try:
        os.link(str(src), str(tmp))
        os.replace(str(tmp), str(dst))
        return
    except OSError:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    # Last-resort fallback for filesystems that disallow links.
    with src.open('rb') as fsrc, tmp.open('wb') as fdst:
        while True:
            chunk = fsrc.read(1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)
    os.replace(str(tmp), str(dst))


def prepare_rgb(src: Path, dst: Path, size: Tuple[int, int]) -> None:
    tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp.png')
    with Image.open(src) as image:
        image = image.convert('RGB')
        image = ImageOps.fit(image, size, method=BICUBIC, centering=(0.5, 0.5))
        image.save(tmp, format='PNG')
    os.replace(str(tmp), str(dst))


def prepare_depth(src: Path, dst: Path, size: Tuple[int, int], depth_format: str) -> None:
    with Image.open(src) as depth:
        depth = depth.convert('L')
        depth = ImageOps.fit(depth, size, method=BICUBIC, centering=(0.5, 0.5))
    if depth_format == 'png':
        tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp.png')
        depth.save(tmp, format='PNG')
        os.replace(str(tmp), str(dst))
        return

    from scipy.io import savemat

    tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp')
    arr = np.asarray(depth).astype(np.float32) / 255.0
    # Keep several common keys so old code variants can load the file.
    savemat(str(tmp), {
        'depth': arr,
        'dph': arr,
        'D': arr,
        'data': arr,
    }, appendmat=False)
    os.replace(str(tmp), str(dst))


def image_output_ready(
    path: Path,
    size: Tuple[int, int],
    mode: str,
    verify: bool,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not verify:
        return True
    try:
        with Image.open(path) as image:
            if image.size != size or image.mode != mode:
                return False
            image.load()
    except (OSError, TypeError, ValueError):
        return False
    return True


def depth_output_ready(
    path: Path,
    size: Tuple[int, int],
    depth_format: str,
    verify: bool,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not verify:
        return True
    if depth_format == 'png':
        return image_output_ready(path, size, 'L', True)

    try:
        from scipy.io import loadmat

        values = loadmat(str(path))
        for key in ('depth', 'dph', 'D', 'data'):
            if key in values and values[key].shape == (size[1], size[0]):
                return True
    except (OSError, TypeError, ValueError):
        return False
    return False


def prepare_air_task(task):
    (
        image_path,
        depth_path,
        air_dst,
        depth_dst,
        air_size,
        depth_format,
        prepare_air,
        prepare_air_depth,
    ) = task
    if prepare_air:
        prepare_rgb(Path(image_path), Path(air_dst), air_size)
    if prepare_air_depth:
        prepare_depth(Path(depth_path), Path(depth_dst), air_size, depth_format)
    return int(prepare_air), int(prepare_air_depth)


def prepare_water_task(task):
    image_path, water_dst, water_size = task
    prepare_rgb(Path(image_path), Path(water_dst), water_size)
    return 1


def run_parallel(tasks, worker, workers: int, description: str):
    if not tasks:
        return []
    if workers <= 1:
        return [
            worker(task)
            for task in tqdm(tasks, desc=description, unit='image')
        ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(worker, tasks, chunksize=8)
        return list(tqdm(
            results,
            total=len(tasks),
            desc=description,
            unit='image',
        ))


def write_json_atomic(path: Path, value) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    os.replace(str(tmp), str(path))


def prepare_config(args, air_source: Path, depth_source: Path, water_source: Path):
    return {
        'air_source': str(air_source.resolve()),
        'depth_source': str(depth_source.resolve()),
        'water_source': str(water_source.resolve()),
        'air_limit': args.air_limit,
        'water_limit': args.water_limit,
        'air_per_class': args.air_per_class,
        'water_repeat_to': args.water_repeat_to,
        'seed': args.seed,
        'air_size': [args.air_width, args.air_height],
        'water_size': [args.water_width, args.water_height],
        'depth_format': args.depth_format,
    }


def find_config_mismatches(previous: Dict, current: Dict) -> List[str]:
    mismatches = []
    for key in sorted(current):
        if previous.get(key) != current[key]:
            mismatches.append(
                f'{key}: previous={previous.get(key)!r}, current={current[key]!r}'
            )
    return mismatches


def main() -> None:
    args = parse_args()
    air_source = Path(args.air_source)
    depth_source = Path(args.depth_source)
    water_source = Path(args.water_source)
    out_dir = Path(args.out_dir)

    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    if out_dir.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f'{out_dir} exists; pass --resume to continue or --overwrite to rebuild.'
        )
    (out_dir / 'air_images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'air_depth').mkdir(parents=True, exist_ok=True)
    (out_dir / 'water_images').mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for child in ('air_images', 'air_depth', 'water_images'):
            for old in (out_dir / child).iterdir():
                if old.is_file() or old.is_symlink():
                    old.unlink()

    config_path = out_dir / 'prepare_watergan_config.json'
    current_config = prepare_config(args, air_source, depth_source, water_source)
    if args.resume and config_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding='utf-8'))
        mismatches = find_config_mismatches(previous_config, current_config)
        if mismatches:
            raise RuntimeError(
                'Resume configuration does not match the existing dataset:\n  '
                + '\n  '.join(mismatches)
            )
    elif args.resume:
        print(
            f'warning: resume metadata not found: {config_path}; '
            'existing files will be matched by deterministic index',
            flush=True,
        )
    write_json_atomic(config_path, current_config)

    other_depth_suffix = '.mat' if args.depth_format == 'png' else '.png'
    stale_depth = next((out_dir / 'air_depth').glob(f'*{other_depth_suffix}'), None)
    if args.resume and stale_depth is not None:
        raise RuntimeError(
            f'Resume found depth files in the wrong format, for example {stale_depth}. '
            f'Remove *{other_depth_suffix} from air_depth or use --overwrite.'
        )

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
    air_tasks = []
    reused_air = 0
    reused_depth = 0
    air_size = (args.air_width, args.air_height)
    water_size = (args.water_width, args.water_height)

    for index, image_path in enumerate(air_images):
        rel = image_path.relative_to(air_source)
        depth_path = depth_source / rel.with_suffix('.png')
        if not depth_path.exists():
            missing_depth.append(str(rel).replace('\\', '/'))
            continue
        stem = f'{index:08d}'
        air_dst = out_dir / 'air_images' / f'{stem}.png'
        depth_dst = out_dir / 'air_depth' / f'{stem}.{args.depth_format}'
        air_ready = args.resume and image_output_ready(
            air_dst, air_size, 'RGB', args.verify_existing
        )
        depth_ready = args.resume and depth_output_ready(
            depth_dst, air_size, args.depth_format, args.verify_existing
        )
        reused_air += int(air_ready)
        reused_depth += int(depth_ready)
        if not air_ready or not depth_ready:
            air_tasks.append((
                str(image_path),
                str(depth_path),
                str(air_dst),
                str(depth_dst),
                air_size,
                args.depth_format,
                not air_ready,
                not depth_ready,
            ))
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

    print(
        f'air/depth resume plan: selected={len(records)}, '
        f'reused_air={reused_air}, reused_depth={reused_depth}, '
        f'tasks={len(air_tasks)}, workers={args.workers}',
        flush=True,
    )
    air_results = run_parallel(
        air_tasks,
        prepare_air_task,
        args.workers,
        'prepare WaterGAN air/depth',
    )
    written_air = sum(result[0] for result in air_results)
    written_depth = sum(result[1] for result in air_results)

    linked_water = 0
    reused_water = 0
    water_tasks = []
    base_water_count = min(selected_water_before_repeat, len(water_images))
    for index, image_path in enumerate(water_images[:base_water_count]):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        water_ready = args.resume and image_output_ready(
            water_dst, water_size, 'RGB', args.verify_existing
        )
        if water_ready:
            reused_water += 1
        else:
            water_tasks.append((str(image_path), str(water_dst), water_size))

    print(
        f'water resume plan: selected={len(water_images)}, '
        f'reused={reused_water}, tasks={len(water_tasks)}, '
        f'workers={args.workers}',
        flush=True,
    )
    run_parallel(
        water_tasks,
        prepare_water_task,
        args.workers,
        'prepare WaterGAN water',
    )

    for index in tqdm(
        range(base_water_count, len(water_images)),
        desc='link repeated WaterGAN water',
        unit='image',
    ):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        if args.resume and (water_dst.is_file() or water_dst.is_symlink()):
            reused_water += 1
            continue
        base_dst = out_dir / 'water_images' / f'{index % base_water_count:08d}.png'
        link_or_copy(base_dst, water_dst)
        linked_water += 1

    manifest = out_dir / 'watergan_air_manifest.jsonl'
    manifest_tmp = manifest.with_name(f'.{manifest.name}.{os.getpid()}.tmp')
    with manifest_tmp.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    os.replace(str(manifest_tmp), str(manifest))

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
        'workers': args.workers,
        'resume': args.resume,
        'verify_existing': args.verify_existing,
        'selected_air_before_limit': selected_air_before_limit,
        'selected_water_before_repeat': selected_water_before_repeat,
        'prepared_air': len(records),
        'prepared_water': len(water_images),
        'reused_air': reused_air,
        'reused_depth': reused_depth,
        'written_air': written_air,
        'written_depth': written_depth,
        'reused_water': reused_water,
        'written_water': len(water_tasks),
        'linked_water': linked_water,
        'missing_depth': len(missing_depth),
        'missing_depth_samples': missing_depth[:20],
        'air_size': list(air_size),
        'water_size': list(water_size),
        'depth_format': args.depth_format,
        'manifest': str(manifest),
    }
    summary_path = out_dir / 'prepare_watergan_summary.json'
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
