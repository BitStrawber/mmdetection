#!/usr/bin/env python3
"""Prepare CUT unaligned A/B directories from sampled ImageNet and RUOD.

The source selection stage stores ImageNet samples as:

    synthetic_imagenet/cut/source/{train,val}/<synset>/<image>

CUT expects flat directories:

    dataroot/trainA
    dataroot/trainB
    dataroot/testA
    dataroot/testB

This tool flattens those inputs and writes manifests so generated fake_B files
can be restored to ImageNet-style <synset>/<image> folders later.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare CUT ImageNet/RUOD dataset.')
    parser.add_argument('--train-a-source', required=True)
    parser.add_argument('--train-b-source', required=True)
    parser.add_argument('--test-a-source', required=True)
    parser.add_argument('--test-b-source', default='',
                        help='Defaults to --train-b-source.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--train-a-limit', type=int, default=1000)
    parser.add_argument('--train-b-limit', type=int, default=1000)
    parser.add_argument('--test-a-limit', type=int, default=100)
    parser.add_argument('--test-b-limit', type=int, default=100)
    parser.add_argument('--link-mode', choices=('symlink', 'hardlink', 'copy'), default='symlink')
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


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'{path} exists; pass --overwrite to replace it.')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'symlink':
        os.symlink(src, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def flatten_split(
    *,
    src_root: Path,
    dst_dir: Path,
    manifest_path: Path,
    limit: int,
    link_mode: str,
    keep_synset: bool,
    label: str,
) -> dict:
    images = list_images(src_root)
    if limit > 0:
        images = images[:limit]

    records = []
    for index, src in enumerate(tqdm(images, desc=f'prepare {label}', unit='image')):
        rel = src.relative_to(src_root)
        synset = rel.parts[0] if keep_synset and len(rel.parts) > 1 else 'unknown'
        flat_name = f'{index:08d}{src.suffix.lower()}'
        dst = dst_dir / flat_name
        link_or_copy(src, dst, link_mode)
        records.append({
            'index': index,
            'flat_name': flat_name,
            'flat_stem': Path(flat_name).stem,
            'source': str(src),
            'relative': str(rel).replace('\\', '/'),
            'synset': synset,
            'original_name': src.name,
            'destination': str(dst),
        })

    with manifest_path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return {
        'source': str(src_root),
        'destination': str(dst_dir),
        'manifest': str(manifest_path),
        'limit': limit,
        'count': len(records),
    }


def main() -> None:
    args = parse_args()
    train_a_source = Path(args.train_a_source)
    train_b_source = Path(args.train_b_source)
    test_a_source = Path(args.test_a_source)
    test_b_source = Path(args.test_b_source) if args.test_b_source else train_b_source
    out_dir = Path(args.out_dir)

    for path, label in (
        (train_a_source, 'train-a-source'),
        (train_b_source, 'train-b-source'),
        (test_a_source, 'test-a-source'),
        (test_b_source, 'test-b-source'),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f'{label} not found: {path}')

    reset_dir(out_dir, args.overwrite)
    manifest_dir = out_dir / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'out_dir': str(out_dir),
        'link_mode': args.link_mode,
        'splits': {},
    }
    summary['splits']['trainA'] = flatten_split(
        src_root=train_a_source,
        dst_dir=out_dir / 'trainA',
        manifest_path=manifest_dir / 'trainA_manifest.jsonl',
        limit=args.train_a_limit,
        link_mode=args.link_mode,
        keep_synset=True,
        label='trainA',
    )
    summary['splits']['trainB'] = flatten_split(
        src_root=train_b_source,
        dst_dir=out_dir / 'trainB',
        manifest_path=manifest_dir / 'trainB_manifest.jsonl',
        limit=args.train_b_limit,
        link_mode=args.link_mode,
        keep_synset=False,
        label='trainB',
    )
    summary['splits']['testA'] = flatten_split(
        src_root=test_a_source,
        dst_dir=out_dir / 'testA',
        manifest_path=manifest_dir / 'testA_manifest.jsonl',
        limit=args.test_a_limit,
        link_mode=args.link_mode,
        keep_synset=True,
        label='testA',
    )
    summary['splits']['testB'] = flatten_split(
        src_root=test_b_source,
        dst_dir=out_dir / 'testB',
        manifest_path=manifest_dir / 'testB_manifest.jsonl',
        limit=args.test_b_limit,
        link_mode=args.link_mode,
        keep_synset=False,
        label='testB',
    )

    summary_path = manifest_dir / 'prepare_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
