#!/usr/bin/env python3
"""Check RealUW image files for decode/format errors in parallel."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
        '--root',
        default='/media/SSD1/XCX/exp_2/REALUW/imagefolder/train/realuw',
        help='Image directory to check.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory for bad/good lists and summary. '
        'Default: <REALUW_ROOT>/quality_check.')
    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='Parallel worker processes. Default: 32.')
    parser.add_argument(
        '--chunksize',
        type=int,
        default=64,
        help='ProcessPool map chunksize. Default: 64.')
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10000,
        help='Print and flush intermediate files every N checked images.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan images under root.')
    parser.add_argument(
        '--no-good-list',
        action='store_true',
        help='Do not write good_images.txt.')
    parser.add_argument(
        '--fail-on-bad',
        action='store_true',
        help='Exit with code 2 if any bad image is found.')
    return parser.parse_args()


def iter_images(root, recursive=False):
    iterator = root.rglob('*') if recursive else root.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMG_SUFFIXES:
            yield path


def default_out_dir(root):
    parts = root.parts
    if len(parts) >= 3 and parts[-3:] == ('imagefolder', 'train', 'realuw'):
        return root.parents[2] / 'quality_check'
    return root / 'quality_check'


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


def check_image(path):
    path = Path(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert('RGB').load()
        return str(path), ''
    except Exception as exc:  # noqa: BLE001
        return str(path), '{}: {}'.format(type(exc).__name__, exc)


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_txt = out_dir / 'bad_images.txt'
    good_txt = out_dir / 'good_images.txt'
    summary_json = out_dir / 'bad_images_summary.json'

    paths = list(iter_images(root, args.recursive))
    bad = []
    good = []
    error_types = Counter()
    start = time.time()

    print('root:', root, flush=True)
    print('out_dir:', out_dir, flush=True)
    print('total_images:', len(paths), flush=True)
    print('workers:', args.workers, flush=True)
    print('chunksize:', args.chunksize, flush=True)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(check_image, paths, chunksize=max(1, args.chunksize))
        for idx, (path, error) in enumerate(
                tqdm(results, total=len(paths), desc='check RealUW images', unit='img'),
                1):
            if error:
                bad.append('{}\t{}'.format(path, error))
                error_types[error.split(':', 1)[0]] += 1
            elif not args.no_good_list:
                good.append(path)

            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(
                    '[progress] checked={}/{} bad={} good={}'.format(
                        idx, len(paths), len(bad),
                        idx - len(bad) if args.no_good_list else len(good)),
                    flush=True)
                write_lines(bad_txt, bad)
                if not args.no_good_list:
                    write_lines(good_txt, good)

    write_lines(bad_txt, bad)
    if not args.no_good_list:
        write_lines(good_txt, good)

    summary = {
        'root': str(root),
        'out_dir': str(out_dir),
        'total_images': len(paths),
        'good_images': len(paths) - len(bad),
        'bad_images': len(bad),
        'bad_error_types': dict(error_types),
        'workers': args.workers,
        'chunksize': args.chunksize,
        'recursive': args.recursive,
        'elapsed_sec': round(time.time() - start, 2),
        'bad_list': str(bad_txt),
        'good_list': '' if args.no_good_list else str(good_txt),
    }
    with open(summary_json, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_bad and bad:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
