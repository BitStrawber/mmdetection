#!/usr/bin/env python3
"""Merge selected underwater datasets into one unlabeled RealUW image pool.

The input COCO files are only used to locate the already-selected images. This
script discards bbox/category labels and writes an SSL-friendly dataset:

    out_root/images/train/*.jpg
    out_root/images/val/*.jpg
    out_root/imagefolder/train/realuw/*.jpg
    out_root/imagefolder/val/realuw/*.jpg
    out_root/annotations/train.txt
    out_root/annotations/val.txt
    out_root/annotations/manifest.jsonl
    out_root/annotations/summary.json

By default files are symlinked instead of copied.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


EXP2_BBOX20_PRESET = [
    (
        'coralscop_train',
        '/media/HDD1/XCX/exp_2/CoralSCOP/annotations/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/CoralSCOP',
    ),
    (
        'coralscop_test',
        '/media/HDD1/XCX/exp_2/CoralSCOP/annotations/instances_test_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/CoralSCOP',
    ),
    (
        'maris_train',
        '/media/HDD1/XCX/exp_2/MARIS/annotations/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/MARIS/train',
    ),
    (
        'maris_val',
        '/media/HDD1/XCX/exp_2/MARIS/annotations/instances_val_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/MARIS/val',
    ),
    (
        'muot3m_train',
        '/media/HDD1/XCX/exp_2/MUOT_3M/annotations/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/MUOT_3M',
    ),
    (
        'muot3m_test',
        '/media/HDD1/XCX/exp_2/MUOT_3M/annotations/instances_test_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/MUOT_3M',
    ),
    (
        'uvot400_train',
        '/media/HDD1/XCX/exp_2/UVOT400/train/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/UVOT400/train',
    ),
    (
        'uvot400_test',
        '/media/HDD1/XCX/exp_2/UVOT400/test/instances_test_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/UVOT400/test',
    ),
    (
        'duo_train',
        '/media/HDD1/XCX/exp_2/DUO/annotations/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/DUO/images',
    ),
    (
        'duo_test',
        '/media/HDD1/XCX/exp_2/DUO/annotations/instances_test_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/DUO/images',
    ),
    (
        'fathomnet_all',
        '/media/HDD1/XCX/exp_2/FathomNet/fathomnet_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/FathomNet',
    ),
    (
        'usis16k_train',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/annotations/instances_train_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/train',
    ),
    (
        'usis16k_val',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/annotations/instances_val_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/val',
    ),
    (
        'usis16k_test',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/annotations/instances_test_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/USIS16K/USIS16K/test',
    ),
    (
        'uot100_all',
        '/media/HDD1/XCX/exp_2/UOT100/annotations/instances_all_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/UOT100',
    ),
    (
        'uwcot220_all',
        '/media/HDD1/XCX/exp_2/UW-COT220/annotations/instances_all_bbox20pct.json',
        '/media/HDD1/XCX/exp_2/UW-COT220/UW-COT220/UW-COT220',
    ),
    (
        'webuot1m_train',
        '/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/annotations/instances_train_webuot_bbox20pct.json',
        '/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/train_frames',
    ),
    (
        'webuot1m_test',
        '/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/annotations/instances_test_webuot_bbox20pct.json',
        '/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/test_frames',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preset',
        choices=['exp2_bbox20pct'],
        default='exp2_bbox20pct')
    parser.add_argument(
        '--dataset',
        nargs=3,
        action='append',
        metavar=('NAME', 'ANN', 'IMG_ROOT'),
        help='Custom dataset triplet. Can be used multiple times.')
    parser.add_argument(
        '--out-root',
        default='/media/HDD1/XCX/exp_2/REALUW_SSL',
        help='Output RealUW SSL dataset root.')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.02,
        help='Stable validation split ratio.')
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy image files instead of creating symlinks.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove existing output root before writing.')
    parser.add_argument(
        '--manifest-only',
        action='store_true',
        help='Only write train/val manifest files, do not link/copy images.')
    parser.add_argument(
        '--write-imagefolder',
        action='store_true',
        help='Also write ImageFolder-style links under imagefolder/train/realuw.')
    return parser.parse_args()


def stable_split(key, val_ratio):
    if val_ratio <= 0:
        return 'train'
    if val_ratio >= 1:
        return 'val'
    value = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)
    return 'val' if value / 0xffffffff < val_ratio else 'train'


def safe_stem(text):
    keep = []
    for char in str(text):
        if char.isalnum() or char in {'-', '_', '.'}:
            keep.append(char)
        else:
            keep.append('_')
    return ''.join(keep).strip('_')[:180] or 'image'


def load_coco(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get('images'), list):
        raise ValueError(f'{path} is not a COCO-like json with images.')
    return data


def build_basename_index(root):
    index = defaultdict(list)
    root = Path(root)
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMG_SUFFIXES:
            index[path.name].append(path)
    return index


def find_image(root, file_name, basename_cache):
    root = Path(root)
    rel = Path(file_name)
    candidates = [
        root / rel,
        root / rel.name,
    ]
    for path in candidates:
        if path.exists():
            return path, 'exact'

    cache_key = str(root)
    if cache_key not in basename_cache:
        basename_cache[cache_key] = build_basename_index(root)
    hits = basename_cache[cache_key].get(rel.name, [])
    if hits:
        return hits[0], 'basename'
    return None, 'missing'


def link_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return 'exists'
    if copy:
        shutil.copy2(src, dst)
        return 'copied'
    try:
        os.symlink(src, dst)
        return 'linked'
    except OSError:
        shutil.copy2(src, dst)
        return 'copied_fallback'


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


def main():
    args = parse_args()
    datasets = args.dataset if args.dataset else EXP2_BBOX20_PRESET
    out_root = Path(args.out_root)
    ann_out = out_root / 'annotations'
    meta_out = out_root / 'meta'
    image_out = out_root / 'images'
    imagefolder_out = out_root / 'imagefolder'

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    ann_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        (image_out / 'train').mkdir(parents=True, exist_ok=True)
        (image_out / 'val').mkdir(parents=True, exist_ok=True)
        if args.write_imagefolder:
            (imagefolder_out / 'train' / 'realuw').mkdir(parents=True, exist_ok=True)
            (imagefolder_out / 'val' / 'realuw').mkdir(parents=True, exist_ok=True)

    basename_cache = {}
    split_lines = {'train': [], 'val': []}
    manifest_path = ann_out / 'manifest.jsonl'
    summary = {
        'out_root': str(out_root),
        'mode': 'copy' if args.copy else 'symlink',
        'manifest_only': args.manifest_only,
        'write_imagefolder': args.write_imagefolder,
        'val_ratio': args.val_ratio,
        'datasets': [],
        'splits': {'train': 0, 'val': 0},
        'missing_images': 0,
        'basename_hits': 0,
        'exact_hits': 0,
        'linked': 0,
        'copied': 0,
        'exists': 0,
        'imagefolder_linked': 0,
        'imagefolder_copied': 0,
        'imagefolder_exists': 0,
    }

    with open(manifest_path, 'w', encoding='utf-8') as manifest:
        for dataset_name, ann_path, img_root in datasets:
            ann_path = Path(ann_path)
            if not ann_path.exists():
                print(f'Warning: skip missing annotation: {ann_path}')
                summary['datasets'].append({
                    'dataset': dataset_name,
                    'ann': str(ann_path),
                    'img_root': img_root,
                    'status': 'missing_annotation',
                })
                continue

            coco = load_coco(ann_path)
            dataset_summary = {
                'dataset': dataset_name,
                'ann': str(ann_path),
                'img_root': img_root,
                'input_images': len(coco.get('images', [])),
                'written_images': 0,
                'missing_images': 0,
                'exact_hits': 0,
                'basename_hits': 0,
            }

            for image in tqdm(coco.get('images', []), desc=dataset_name, unit='img'):
                image_id = image.get('id')
                file_name = image.get('file_name', '')
                src, hit_type = find_image(img_root, file_name, basename_cache)
                if src is None:
                    dataset_summary['missing_images'] += 1
                    summary['missing_images'] += 1
                    continue
                if hit_type == 'exact':
                    dataset_summary['exact_hits'] += 1
                    summary['exact_hits'] += 1
                elif hit_type == 'basename':
                    dataset_summary['basename_hits'] += 1
                    summary['basename_hits'] += 1

                split = stable_split(f'{dataset_name}:{image_id}:{file_name}', args.val_ratio)
                suffix = src.suffix.lower()
                dst_name = f'{safe_stem(dataset_name)}__{safe_stem(image_id)}__{safe_stem(Path(file_name).stem)}{suffix}'
                rel_dst = Path('images') / split / dst_name
                dst = out_root / rel_dst

                status = 'manifest_only'
                if not args.manifest_only:
                    status = link_or_copy(src, dst, copy=args.copy)
                    if status in {'linked'}:
                        summary['linked'] += 1
                    elif status in {'copied', 'copied_fallback'}:
                        summary['copied'] += 1
                    elif status == 'exists':
                        summary['exists'] += 1

                    if args.write_imagefolder:
                        imagefolder_dst = imagefolder_out / split / 'realuw' / dst_name
                        imagefolder_status = link_or_copy(
                            src, imagefolder_dst, copy=args.copy)
                        if imagefolder_status == 'linked':
                            summary['imagefolder_linked'] += 1
                        elif imagefolder_status in {'copied', 'copied_fallback'}:
                            summary['imagefolder_copied'] += 1
                        elif imagefolder_status == 'exists':
                            summary['imagefolder_exists'] += 1

                row = {
                    'dataset': dataset_name,
                    'split': split,
                    'image_id': image_id,
                    'original_file_name': file_name,
                    'source_path': str(src),
                    'relative_path': str(rel_dst),
                    'width': image.get('width'),
                    'height': image.get('height'),
                    'status': status,
                    'hit_type': hit_type,
                    'annotation_file': str(ann_path),
                }
                manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                split_lines[split].append(str(rel_dst))
                dataset_summary['written_images'] += 1
                summary['splits'][split] += 1

            summary['datasets'].append(dataset_summary)
            print(
                f'{dataset_name}: {dataset_summary["written_images"]} images, '
                f'missing={dataset_summary["missing_images"]}, '
                f'exact={dataset_summary["exact_hits"]}, '
                f'basename={dataset_summary["basename_hits"]}')

    write_lines(ann_out / 'train.txt', split_lines['train'])
    write_lines(ann_out / 'val.txt', split_lines['val'])
    write_lines(ann_out / 'all.txt', split_lines['train'] + split_lines['val'])
    write_lines(meta_out / 'train.txt', split_lines['train'])
    write_lines(meta_out / 'val.txt', split_lines['val'])
    write_lines(meta_out / 'all.txt', split_lines['train'] + split_lines['val'])

    with open(ann_out / 'summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print('\nRealUW SSL dataset')
    print('=' * 80)
    print('out_root:', out_root)
    print('train:', summary['splits']['train'])
    print('val:', summary['splits']['val'])
    print('total:', summary['splits']['train'] + summary['splits']['val'])
    print('missing_images:', summary['missing_images'])
    print('manifest:', manifest_path)
    print('train_txt:', ann_out / 'train.txt')
    print('val_txt:', ann_out / 'val.txt')
    print('mmpretrain_train_txt:', meta_out / 'train.txt')
    print('mmpretrain_val_txt:', meta_out / 'val.txt')
    if args.write_imagefolder:
        print('imagefolder_train:', imagefolder_out / 'train')
        print('imagefolder_val:', imagefolder_out / 'val')


if __name__ == '__main__':
    main()
