#!/usr/bin/env python3
"""Create a minimal COCO image manifest from a recursive ImageFolder tree.

This is for image synthesis, not object detection. The JSON intentionally has
no annotations; it lets COCO-oriented conversion tools enumerate ImageNet
images while preserving each relative file name.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    out_path = Path(args.out).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')

    files = sorted(path for path in image_dir.rglob('*')
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        files = files[:args.limit]
    images, failures = [], []
    for image_id, path in enumerate(tqdm(files, desc='build image manifest', unit='image'), 1):
        try:
            with Image.open(path) as image:
                width, height = image.size
            images.append({'id': image_id, 'file_name': path.relative_to(image_dir).as_posix(),
                           'width': width, 'height': height})
        except Exception as error:
            failures.append({'file_name': str(path), 'error': repr(error)})

    data = {'images': images, 'annotations': [], 'categories': [],
            'meta': {'source_image_dir': str(image_dir), 'failed_images': failures}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data), encoding='utf-8')
    print(f'images: {len(images)}')
    print(f'failed: {len(failures)}')
    print(f'output: {out_path}')


if __name__ == '__main__':
    main()
