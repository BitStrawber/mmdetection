#!/usr/bin/env python3
"""Create reproducible per-method ImageNet source trees for synthesis.

For every requested method, ``train/<synset>`` receives an independently
sampled set of images. Validation images are allocated without overlap across
methods, which is possible for standard ImageNet-1K validation (50 images per
synset): five methods times ten images consumes all 50 validation images.

The script creates only source links/copies. It deliberately does not create
or populate generated-result directories: each synthesis method creates its
own ``generated/`` tree later, leaving source selection immutable and
auditable.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select ImageNet train/val source images for synthesis methods.')
    parser.add_argument('--train-root', required=True,
                        help='ImageNet train root with 1000 synset directories.')
    parser.add_argument('--val-root', required=True,
                        help='Prepared ImageNet val root with 1000 synset directories.')
    parser.add_argument('--out-root', required=True,
                        help='Synthetic dataset root; only source/ and manifests/ are created.')
    parser.add_argument('--methods', nargs='+', required=True,
                        help='Method directory names, e.g. uwnr watergan syreanet cut method5.')
    parser.add_argument('--train-per-class', type=int, default=200)
    parser.add_argument('--val-per-class', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260621)
    parser.add_argument('--link-mode', choices=('symlink', 'hardlink', 'copy'), default='symlink')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace only existing source/ and manifests/ under --out-root.')
    return parser.parse_args()


def images_in(directory):
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def stable_rng(seed, *parts):
    payload = ':'.join([str(seed), *map(str, parts)]).encode('utf-8')
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16], 16))


def validate_roots(train_root, val_root):
    train_classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    val_classes = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    if not train_classes:
        raise RuntimeError(f'No class directories under train root: {train_root}')
    if train_classes != val_classes:
        missing_val = sorted(set(train_classes) - set(val_classes))
        missing_train = sorted(set(val_classes) - set(train_classes))
        raise RuntimeError(
            'Train/val synset directories differ. '
            f'Missing in val={missing_val[:5]}, missing in train={missing_train[:5]}')
    return train_classes


def link_or_copy(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return False
        raise FileExistsError(f'Refuse to replace existing destination: {destination}')
    if mode == 'symlink':
        os.symlink(source, destination)
    elif mode == 'hardlink':
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def reset_selection(out_root, methods):
    manifest_root = out_root / 'manifests'
    if manifest_root.exists():
        shutil.rmtree(manifest_root)
    for method in methods:
        path = out_root / method / 'source'
        if path.exists():
            shutil.rmtree(path)


def main():
    args = parse_args()
    train_root = Path(args.train_root).resolve()
    val_root = Path(args.val_root).resolve()
    out_root = Path(args.out_root).resolve()
    methods = list(dict.fromkeys(args.methods))

    if len(methods) != len(args.methods):
        raise ValueError('Method names must be unique.')
    if args.train_per_class <= 0 or args.val_per_class <= 0:
        raise ValueError('Per-class sample counts must be positive.')
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError('Both --train-root and --val-root must exist.')

    manifest_root = out_root / 'manifests'
    source_exists = any((out_root / method / 'source').exists() for method in methods)
    if source_exists or manifest_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'{out_root} already contains method source/ or manifests/. '
                'Use --overwrite to replace only selections.')
        reset_selection(out_root, methods)
    manifest_root.mkdir(parents=True, exist_ok=True)

    classes = validate_roots(train_root, val_root)
    required_val = len(methods) * args.val_per_class
    counts = {method: Counter() for method in methods}
    train_overlap = Counter()

    global_manifest = (manifest_root / 'selection.jsonl').open('w', encoding='utf-8')
    try:
        for synset in tqdm(classes, desc='select ImageNet classes', unit='class'):
            train_images = images_in(train_root / synset)
            val_images = images_in(val_root / synset)
            if len(train_images) < args.train_per_class:
                raise RuntimeError(
                    f'{synset}: only {len(train_images)} train images; need {args.train_per_class}.')
            if len(val_images) < required_val:
                raise RuntimeError(
                    f'{synset}: only {len(val_images)} val images; need {required_val} for '
                    f'{len(methods)} methods x {args.val_per_class}.')

            # Train selections are independent by method. Some source overlap is
            # unavoidable for five 200-image draws in classes with fewer than
            # 1,000 source images; generated outputs remain method-specific.
            selected_train = {}
            for method in methods:
                rng = stable_rng(args.seed, 'train', method, synset)
                selected_train[method] = rng.sample(train_images, args.train_per_class)
                for source in selected_train[method]:
                    destination = out_root / method / 'source' / 'train' / synset / source.name
                    link_or_copy(source, destination, args.link_mode)
                    counts[method]['train'] += 1
                    global_manifest.write(json.dumps({
                        'method': method, 'split': 'train', 'synset': synset,
                        'source': str(source.relative_to(train_root)),
                        'selected': str(destination.relative_to(out_root)),
                    }) + '\n')
            occurrences = Counter(source.name for group in selected_train.values() for source in group)
            train_overlap[synset] = sum(count - 1 for count in occurrences.values() if count > 1)

            # Standard ImageNet val has 50 images per class. One shuffled list
            # gives five disjoint 10-image allocations and therefore 50k total.
            val_rng = stable_rng(args.seed, 'val', synset)
            val_rng.shuffle(val_images)
            for index, method in enumerate(methods):
                start = index * args.val_per_class
                for source in val_images[start:start + args.val_per_class]:
                    destination = out_root / method / 'source' / 'val' / synset / source.name
                    link_or_copy(source, destination, args.link_mode)
                    counts[method]['val'] += 1
                    global_manifest.write(json.dumps({
                        'method': method, 'split': 'val', 'synset': synset,
                        'source': str(source.relative_to(val_root)),
                        'selected': str(destination.relative_to(out_root)),
                    }) + '\n')
    finally:
        global_manifest.close()

    summary = {
        'train_root': str(train_root), 'val_root': str(val_root), 'out_root': str(out_root),
        'methods': methods, 'class_count': len(classes), 'train_per_class': args.train_per_class,
        'val_per_class': args.val_per_class, 'seed': args.seed, 'link_mode': args.link_mode,
        'per_method': {method: dict(counts[method]) for method in methods},
        'total_train': sum(counts[method]['train'] for method in methods),
        'total_val': sum(counts[method]['val'] for method in methods),
        'train_source_overlap_count': sum(train_overlap.values()),
        'train_source_overlap_by_class': dict(train_overlap),
        'selection_manifest': str(manifest_root / 'selection.jsonl'),
    }
    summary_path = manifest_root / 'selection_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
