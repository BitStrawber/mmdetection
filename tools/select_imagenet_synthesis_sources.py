#!/usr/bin/env python3
"""Create reproducible per-method ImageNet source trees for synthesis.

For every requested method, ``train/<synset>`` receives a class-balanced
selection. By default each method receives 250k training images: 250 images per
ImageNet-1K class. Methods use rotated windows inside each class, so different
methods cover different source images as much as possible while allowing overlap
when a class does not have enough unique images.

Validation images are still allocated without overlap across methods by default,
which is possible for standard ImageNet-1K validation (50 images per synset):
five methods times ten images consumes all 50 validation images.

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
    parser.add_argument('--train-per-method', type=int, default=250000,
                        help='Total train images selected for every method. '
                             'Ignored when --train-per-class is set.')
    parser.add_argument('--train-per-class', type=int, default=None,
                        help='Fixed train images per class for every method. '
                             'Overrides --train-per-method when set.')
    parser.add_argument('--val-per-class', type=int, default=10,
                        help='Validation images per class for every method. Set 0 to skip val.')
    parser.add_argument('--train-selection',
                        choices=('rotating-class-balanced', 'independent-random'),
                        default='rotating-class-balanced',
                        help='rotating-class-balanced maximizes cross-method coverage inside each class; '
                             'independent-random reproduces the old independent sampling style.')
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


def rotated_take(items, start, count):
    if count <= 0:
        return []
    if not items:
        return []
    size = len(items)
    if count > size:
        raise RuntimeError(f'Cannot take {count} unique items from a class with only {size} images.')
    start %= size
    if start + count <= size:
        return items[start:start + count]
    return items[start:] + items[:start + count - size]


def build_train_quotas(classes, train_per_class, train_per_method, seed):
    if train_per_class is not None:
        if train_per_class <= 0:
            raise ValueError('--train-per-class must be positive when set.')
        return {synset: train_per_class for synset in classes}, train_per_class * len(classes)

    if train_per_method <= 0:
        raise ValueError('--train-per-method must be positive.')
    if train_per_method < len(classes):
        raise ValueError(
            f'--train-per-method={train_per_method} is smaller than class count={len(classes)}.')

    base = train_per_method // len(classes)
    remainder = train_per_method % len(classes)
    class_order = list(classes)
    stable_rng(seed, 'train-quota').shuffle(class_order)
    extra_classes = set(class_order[:remainder])
    quotas = {synset: base + (1 if synset in extra_classes else 0) for synset in classes}
    return quotas, train_per_method


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
    if args.val_per_class < 0:
        raise ValueError('--val-per-class must be >= 0.')
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
    train_quotas, train_per_method = build_train_quotas(
        classes, args.train_per_class, args.train_per_method, args.seed)
    required_val = len(methods) * args.val_per_class
    counts = {method: Counter() for method in methods}
    train_overlap = Counter()
    train_unique_coverage = Counter()

    global_manifest = (manifest_root / 'selection.jsonl').open('w', encoding='utf-8')
    try:
        for synset in tqdm(classes, desc='select ImageNet classes', unit='class'):
            train_images = images_in(train_root / synset)
            val_images = images_in(val_root / synset)
            train_quota = train_quotas[synset]
            if len(train_images) < train_quota:
                raise RuntimeError(
                    f'{synset}: only {len(train_images)} train images; need {train_quota}.')
            if args.val_per_class > 0 and len(val_images) < required_val:
                raise RuntimeError(
                    f'{synset}: only {len(val_images)} val images; need {required_val} for '
                    f'{len(methods)} methods x {args.val_per_class}.')

            train_order = list(train_images)
            stable_rng(args.seed, 'train-order', synset).shuffle(train_order)

            # In rotating mode, method k starts at k * quota in the same class
            # order. Five 250-image windows therefore cover about 1,250 unique
            # images per class before overlap is necessary. Independent-random
            # is kept for reproducing earlier source selections.
            selected_train = {}
            for method_index, method in enumerate(methods):
                if args.train_selection == 'independent-random':
                    rng = stable_rng(args.seed, 'train', method, synset)
                    selected_train[method] = rng.sample(train_images, train_quota)
                else:
                    selected_train[method] = rotated_take(
                        train_order, method_index * train_quota, train_quota)
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
            train_unique_coverage[synset] = len(occurrences)

            # Standard ImageNet val has 50 images per class. One shuffled list
            # gives five disjoint 10-image allocations and therefore 50k total.
            if args.val_per_class > 0:
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
        'methods': methods, 'class_count': len(classes),
        'train_per_method': train_per_method,
        'train_per_class': args.train_per_class,
        'train_quota_min': min(train_quotas.values()),
        'train_quota_max': max(train_quotas.values()),
        'val_per_class': args.val_per_class,
        'seed': args.seed, 'link_mode': args.link_mode,
        'train_selection': args.train_selection,
        'per_method': {method: dict(counts[method]) for method in methods},
        'total_train': sum(counts[method]['train'] for method in methods),
        'total_val': sum(counts[method]['val'] for method in methods),
        'train_source_overlap_count': sum(train_overlap.values()),
        'train_source_overlap_by_class': dict(train_overlap),
        'train_unique_source_count': sum(train_unique_coverage.values()),
        'train_unique_source_by_class': dict(train_unique_coverage),
        'selection_manifest': str(manifest_root / 'selection.jsonl'),
    }
    summary_path = manifest_root / 'selection_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
