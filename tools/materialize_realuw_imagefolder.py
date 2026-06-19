#!/usr/bin/env python3
"""Copy an existing RealUW ImageFolder dataset into a real-file directory.

This script is intended for the RealUW SSL dataset produced by
``tools/build_realuw_ssl_dataset.py``.  That builder normally creates symlinks
under:

    REALUW_SSL/imagefolder/train/realuw
    REALUW_SSL/imagefolder/val/realuw

For long self-supervised pretraining, reading many symlinks from HDD can add
I/O jitter.  This tool resolves every source image and copies it into a
dedicated REALUW directory, usually on SSD:

    REALUW/imagefolder/train/realuw
    REALUW/imagefolder/val/realuw

The copied output keeps the same ImageFolder layout, so existing DINO/MAE
scripts can use it by setting REALUW_SSL_ROOT to the new output root.

If all RealUW images should be used for self-supervised pretraining, pass
``--merge-to-train``.  Then all requested input splits are copied into:

    REALUW/imagefolder/train/realuw
"""

import argparse
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-root',
        default='/media/HDD1/XCX/exp_2/REALUW_SSL',
        help='Existing RealUW SSL root that contains imagefolder/train/realuw.')
    parser.add_argument(
        '--out-root',
        default='/media/SSD1/XCX/exp_2/REALUW',
        help='Output root for the materialized real-file RealUW dataset.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='ImageFolder splits to copy. Default: train val.')
    parser.add_argument(
        '--class-name',
        default='realuw',
        help='ImageFolder class directory name. Default: realuw.')
    parser.add_argument(
        '--merge-to-train',
        action='store_true',
        help='Copy all requested source splits into output train/realuw.')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Parallel copy workers. Use 4-8 for HDD sources.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing destination files.')
    parser.add_argument(
        '--clean-out-root',
        action='store_true',
        help='Delete out-root before copying. Use carefully.')
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Copy at most N images per split for smoke tests. 0 means all.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only scan and report planned work.')
    parser.add_argument(
        '--copy-meta',
        action='store_true',
        help='Copy src-root/meta and src-root/annotations if present.')
    parser.add_argument(
        '--check-images',
        action='store_true',
        help='Verify copied images after materialization and write bad image reports.')
    parser.add_argument(
        '--check-workers',
        type=int,
        default=16,
        help='Parallel image verification workers. Default: 16.')
    return parser.parse_args()


def iter_images(split_dir):
    for path in split_dir.iterdir():
        if path.is_file() or path.is_symlink():
            if path.suffix.lower() in IMG_SUFFIXES:
                yield path


def same_size(src, dst):
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def copy_one(item):
    src, dst, overwrite = item
    result = {
        'src': str(src),
        'dst': str(dst),
        'status': 'unknown',
        'error': '',
    }
    try:
        resolved = src.resolve(strict=True)
        if not resolved.is_file():
            result['status'] = 'missing_source'
            result['error'] = 'resolved source is not a file'
            return result

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not overwrite and same_size(resolved, dst):
                result['status'] = 'skipped_exists'
                return result
            if not overwrite:
                result['status'] = 'conflict'
                result['error'] = 'destination exists with different size'
                return result

        tmp = dst.with_suffix(dst.suffix + '.tmp')
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(str(resolved), str(tmp))
        os.replace(str(tmp), str(dst))
        result['status'] = 'copied'
        return result
    except Exception as exc:  # noqa: BLE001
        result['status'] = 'error'
        result['error'] = '{}: {}'.format(type(exc).__name__, exc)
        return result


def check_image_file(path):
    path = Path(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert('RGB').load()
        return None
    except Exception as exc:  # noqa: BLE001
        return '{}\t{}: {}'.format(path, type(exc).__name__, exc)


def copy_tree_if_present(src, dst, overwrite=False):
    if not src.exists():
        return False
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst, symlinks=False)
    return True


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if value < 1024.0 or unit == 'TiB':
            return '{:.2f} {}'.format(value, unit)
        value /= 1024.0


def collect_imagefolder_stats(out_root, splits):
    stats = {
        'total_images': 0,
        'total_bytes': 0,
        'total_size': '0.00 B',
        'splits': {},
    }
    imagefolder = out_root / 'imagefolder'
    for split in splits:
        split_dir = imagefolder / split
        split_stats = {
            'images': 0,
            'bytes': 0,
            'size': '0.00 B',
            'categories': {},
        }
        if split_dir.exists():
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                paths = list(iter_images(class_dir))
                class_bytes = 0
                for path in paths:
                    try:
                        class_bytes += path.stat().st_size
                    except OSError:
                        pass
                split_stats['categories'][class_dir.name] = {
                    'images': len(paths),
                    'bytes': class_bytes,
                    'size': format_bytes(class_bytes),
                }
                split_stats['images'] += len(paths)
                split_stats['bytes'] += class_bytes
        split_stats['size'] = format_bytes(split_stats['bytes'])
        stats['splits'][split] = split_stats
        stats['total_images'] += split_stats['images']
        stats['total_bytes'] += split_stats['bytes']
    stats['total_size'] = format_bytes(stats['total_bytes'])
    return stats


def scan_bad_images(out_root, splits, class_name, workers):
    image_paths = []
    for split in splits:
        split_dir = out_root / 'imagefolder' / split / class_name
        if split_dir.exists():
            image_paths.extend(sorted(iter_images(split_dir)))

    report_dir = out_root / 'quality_check'
    report_dir.mkdir(parents=True, exist_ok=True)
    bad_txt = report_dir / 'bad_images.txt'
    good_txt = report_dir / 'good_images.txt'
    summary_json = report_dir / 'bad_images_summary.json'

    bad = []
    good = []
    error_types = Counter()
    started = time.time()
    print('')
    print('Image quality check')
    print('=' * 80)
    print('images:', len(image_paths))
    print('workers:', workers)

    if image_paths:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_path = {
                executor.submit(check_image_file, str(p)): p
                for p in image_paths
            }
            for future in tqdm(
                    as_completed(future_to_path),
                    total=len(future_to_path),
                    desc='check RealUW images',
                    unit='img'):
                path = future_to_path[future]
                result = future.result()
                if result:
                    bad.append(result)
                    error_type = result.split('\t', 1)[1].split(':', 1)[0]
                    error_types[error_type] += 1
                    if len(bad) % 100 == 0:
                        write_lines(bad_txt, bad)
                else:
                    good.append(str(path))

    write_lines(bad_txt, bad)
    if len(good) == len(image_paths) - len(bad):
        write_lines(good_txt, good)
    summary = {
        'root': str(out_root),
        'splits': splits,
        'class_name': class_name,
        'total_images': len(image_paths),
        'good_images': len(image_paths) - len(bad),
        'bad_images': len(bad),
        'bad_error_types': dict(error_types),
        'workers': workers,
        'elapsed_sec': round(time.time() - started, 2),
        'bad_list': str(bad_txt),
        'good_list': str(good_txt),
    }
    with open(summary_json, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print('bad_images:', len(bad))
    print('good_images:', len(image_paths) - len(bad))
    if error_types:
        print('bad_error_types:', dict(error_types))
    print('bad_list:', bad_txt)
    print('good_list:', good_txt)
    print('summary:', summary_json)
    return summary


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    src_imagefolder = src_root / 'imagefolder'
    out_imagefolder = out_root / 'imagefolder'

    if src_root.resolve(strict=False) == out_root.resolve(strict=False):
        raise ValueError('src-root and out-root must be different directories.')

    if args.clean_out_root and out_root.exists():
        shutil.rmtree(out_root)

    start = time.time()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    summary = {
        'src_root': str(src_root),
        'out_root': str(out_root),
        'splits': {},
        'workers': args.workers,
        'overwrite': args.overwrite,
        'merge_to_train': args.merge_to_train,
        'dry_run': args.dry_run,
        'check_images': args.check_images,
        'check_workers': args.check_workers,
        'limit_per_split': args.limit,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    all_jobs = []
    planned_destinations = {}
    for split in args.splits:
        src_split = src_imagefolder / split / args.class_name
        out_split = 'train' if args.merge_to_train else split
        dst_split = out_imagefolder / out_split / args.class_name
        split_summary = {
            'src_dir': str(src_split),
            'dst_dir': str(dst_split),
            'output_split': out_split,
            'source_images': 0,
            'planned_images': 0,
            'output_files': 0,
            'output_symlinks': 0,
            'copied': 0,
            'skipped_exists': 0,
            'conflict': 0,
            'missing_source': 0,
            'error': 0,
        }
        summary['splits'][split] = split_summary

        if not src_split.exists():
            split_summary['error'] = 1
            split_summary['error_message'] = 'source split does not exist'
            continue

        print('scan source split:', src_split, flush=True)
        paths = list(iter_images(src_split))
        print('  found images:', len(paths), flush=True)
        split_summary['source_images'] = len(paths)
        if args.limit > 0:
            paths = paths[:args.limit]
        split_summary['planned_images'] = len(paths)

        for src in paths:
            dst = dst_split / src.name
            if dst in planned_destinations and planned_destinations[dst] != src:
                dst = dst_split / '{}__{}'.format(split, src.name)
            planned_destinations[dst] = src
            all_jobs.append((split, out_split, src, dst))
            manifest.append({
                'source_split': split,
                'output_split': out_split,
                'class_name': args.class_name,
                'source': str(src),
                'resolved_source': str(src.resolve(strict=False)),
                'file_name': str(Path('imagefolder') / out_split / args.class_name / dst.name),
            })

    print('src_root:', src_root, flush=True)
    print('out_root:', out_root, flush=True)
    print('planned images:', len(all_jobs), flush=True)
    print('workers:', args.workers, flush=True)
    print('dry_run:', args.dry_run, flush=True)

    if not args.dry_run and all_jobs:
        jobs = [(src, dst, args.overwrite) for _, _, src, dst in all_jobs]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_split = {
                executor.submit(copy_one, job): split
                for (split, _, _, _), job in zip(all_jobs, jobs)
            }
            for future in tqdm(
                    as_completed(future_to_split),
                    total=len(future_to_split),
                    desc='copy RealUW images',
                    unit='img'):
                split = future_to_split[future]
                result = future.result()
                status = result['status']
                split_stats = summary['splits'][split]
                if status in split_stats:
                    split_stats[status] += 1
                else:
                    split_stats['error'] += 1

    if args.copy_meta and not args.dry_run:
        copied_meta = []
        for name in ['meta', 'annotations']:
            if copy_tree_if_present(src_root / name, out_root / name, args.overwrite):
                copied_meta.append(name)
        summary['copied_metadata_dirs'] = copied_meta

    if not args.dry_run:
        manifest_path = out_root / 'materialized_manifest.jsonl'
        with open(manifest_path, 'w', encoding='utf-8') as file:
            for row in manifest:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
        summary['manifest'] = str(manifest_path)

        output_splits = ['train'] if args.merge_to_train else args.splits
        for split in output_splits:
            dst_split = out_imagefolder / split / args.class_name
            if dst_split.exists():
                output_paths = sorted(iter_images(dst_split))
                target_stats = None
                if split in summary['splits']:
                    target_stats = summary['splits'][split]
                elif args.merge_to_train:
                    target_stats = summary['splits'].setdefault('merged_train', {})
                if target_stats is not None:
                    target_stats['output_files'] = len(output_paths)
                    target_stats['output_symlinks'] = sum(
                        1 for p in output_paths if p.is_symlink())
                lines = [
                    str(Path('imagefolder') / split / args.class_name / p.name)
                    for p in output_paths
                ]
                write_lines(out_root / 'meta' / '{}.txt'.format(split), lines)

    if args.check_images and not args.dry_run:
        check_splits = ['train'] if args.merge_to_train else args.splits
        summary['imagefolder_stats'] = collect_imagefolder_stats(
            out_root, check_splits)
        summary['image_quality_check'] = scan_bad_images(
            out_root, check_splits, args.class_name, args.check_workers)
    elif not args.dry_run:
        stat_splits = ['train'] if args.merge_to_train else args.splits
        summary['imagefolder_stats'] = collect_imagefolder_stats(
            out_root, stat_splits)

    summary['elapsed_sec'] = round(time.time() - start, 2)
    summary['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_path = out_root / 'materialize_summary.json'
    if not args.dry_run:
        with open(summary_path, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        summary['summary_path'] = str(summary_path)

    print('')
    print('Summary')
    print('=' * 80)
    total_source = total_planned = total_copied = total_skipped = total_errors = 0
    for split, stats in summary['splits'].items():
        total_source += stats.get('source_images', 0)
        total_planned += stats.get('planned_images', 0)
        total_copied += stats.get('copied', 0)
        total_skipped += stats.get('skipped_exists', 0)
        total_errors += (
            stats.get('conflict', 0)
            + stats.get('missing_source', 0)
            + stats.get('error', 0)
        )
        print(
            '{:<8s} source={} planned={} copied={} skipped={} conflicts={} '
            'missing={} errors={} output={} symlinks={}'.format(
                split,
                stats.get('source_images', 0),
                stats.get('planned_images', 0),
                stats.get('copied', 0),
                stats.get('skipped_exists', 0),
                stats.get('conflict', 0),
                stats.get('missing_source', 0),
                stats.get('error', 0),
                stats.get('output_files', 0),
                stats.get('output_symlinks', 0),
            )
        )
    print('-' * 80)
    print(
        'TOTAL source={} planned={} copied={} skipped={} errors={}'.format(
            total_source, total_planned, total_copied, total_skipped, total_errors
        )
    )
    if 'imagefolder_stats' in summary:
        stats = summary['imagefolder_stats']
        print('')
        print('Dataset Size / Category Counts')
        print('=' * 80)
        print('total_images:', stats['total_images'])
        print('total_size:', stats['total_size'])
        for split, split_stats in stats['splits'].items():
            print('{}: images={} size={}'.format(
                split, split_stats['images'], split_stats['size']))
            for category, cat_stats in split_stats['categories'].items():
                print('  {}: images={} size={}'.format(
                    category, cat_stats['images'], cat_stats['size']))
    if not args.dry_run:
        print('summary:', summary_path)


if __name__ == '__main__':
    main()
