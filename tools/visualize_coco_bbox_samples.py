#!/usr/bin/env python3
"""Visualize random COCO bbox samples from large-object filtered datasets.

The script samples up to N images from each COCO annotation file, draws all
annotations for each sampled image, and writes visualizations into one
subdirectory per dataset. Boxes whose bbox area is at least ``--threshold`` of
the image area are drawn in red; other annotations kept in the same filtered
COCO file are drawn in yellow.
"""

import argparse
import json
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preset',
        choices=['exp2_bbox20pct'],
        default=None,
        help='Use built-in exp_2 large-object filtered dataset paths.')
    parser.add_argument(
        '--dataset',
        nargs=3,
        action='append',
        metavar=('NAME', 'ANN_JSON', 'IMAGE_ROOT'),
        help='Dataset triplet. Can be specified multiple times.')
    parser.add_argument(
        '--out-dir',
        default='/media/HDD1/XCX/exp_2/bbox20pct_visual_check',
        help='Output directory containing one subfolder per dataset.')
    parser.add_argument('--num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
        help='BBox/image area ratio used for red highlight.')
    parser.add_argument(
        '--recursive-search',
        action='store_true',
        default=True,
        help='Fallback to recursive basename search when direct paths fail.')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on missing annotation files instead of skipping them.')
    return parser.parse_args()


def safe_name(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_') or 'item'


def load_font(size=18):
    for path in (
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def bbox_ratio(ann, image):
    bbox = ann.get('bbox') or []
    if len(bbox) != 4:
        return 0.0
    img_w = float(image.get('width') or 0)
    img_h = float(image.get('height') or 0)
    if img_w <= 0 or img_h <= 0:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])) / (img_w * img_h)


def build_basename_index(root):
    index = {}
    root = Path(root)
    if not root.is_dir():
        return index
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in IMG_SUFFIXES:
            continue
        index.setdefault(path.name, []).append(path)
    return index


def resolve_image_path(image, root, basename_index=None):
    root = Path(root)
    file_name = str(image.get('file_name') or '')
    candidates = [
        root / file_name,
        root / Path(file_name).name,
    ]
    for path in candidates:
        if path.is_file():
            return path, 'direct'

    if basename_index is not None:
        matches = basename_index.get(Path(file_name).name, [])
        if len(matches) == 1:
            return matches[0], 'basename'
        if len(matches) > 1:
            return matches[0], f'basename_ambiguous:{len(matches)}'
    return None, 'missing'


def draw_label(draw, xy, text, font):
    x, y = xy
    if hasattr(draw, 'textbbox'):
        box = draw.textbbox((x, y), text, font=font)
        w = box[2] - box[0]
        h = box[3] - box[1]
    else:
        w, h = draw.textsize(text, font=font)
    draw.rectangle([x, y, x + w + 6, y + h + 4], fill=(0, 0, 0))
    draw.text((x + 3, y + 2), text, fill=(255, 255, 255), font=font)


def draw_annotations(image_path, image_info, anns, categories, threshold, out_path):
    with Image.open(image_path) as img:
        canvas = img.convert('RGB')

    draw = ImageDraw.Draw(canvas)
    font = load_font()
    width, height = canvas.size

    for ann in anns:
        bbox = ann.get('bbox') or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + max(0.0, w)))
        y2 = max(0.0, min(float(height), y + max(0.0, h)))
        if x2 <= x1 or y2 <= y1:
            continue

        ratio = bbox_ratio(ann, image_info)
        color = (255, 36, 36) if ratio >= threshold else (255, 210, 0)
        line_width = 5 if ratio >= threshold else 3
        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color)
        cat_name = categories.get(ann.get('category_id'), str(ann.get('category_id')))
        label = f'{cat_name} {ratio:.3f}'
        draw_label(draw, (int(x1), max(0, int(y1) - 24)), label, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def visualize_dataset(name, ann_path, image_root, out_dir, num, seed, threshold,
                      recursive_search, strict):
    ann_path = Path(ann_path)
    image_root = Path(image_root)
    dataset_out = Path(out_dir) / safe_name(name)

    if not ann_path.is_file():
        message = f'[skip] {name}: annotation file not found: {ann_path}'
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {
            'name': name,
            'status': 'missing_annotation',
            'annotation': str(ann_path),
        }

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = list(coco.get('images', []))
    anns_by_img = {img.get('id'): [] for img in images}
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)

    if not images:
        print(f'[skip] {name}: no images in {ann_path}')
        return {
            'name': name,
            'status': 'empty',
            'annotation': str(ann_path),
            'images': 0,
            'annotations': len(coco.get('annotations', [])),
        }

    rng = random.Random(seed)
    sampled = rng.sample(images, min(num, len(images)))
    categories = {
        cat.get('id'): cat.get('name', str(cat.get('id')))
        for cat in coco.get('categories', [])
    }
    basename_index = build_basename_index(image_root) if recursive_search else None

    manifest = []
    missing = 0
    for idx, image in enumerate(tqdm(sampled, desc=name, unit='img'), start=1):
        src_path, resolve_mode = resolve_image_path(image, image_root, basename_index)
        if src_path is None:
            missing += 1
            manifest.append({
                'image_id': image.get('id'),
                'file_name': image.get('file_name'),
                'status': 'missing_image',
            })
            continue

        anns = anns_by_img.get(image.get('id'), [])
        max_ratio = max((bbox_ratio(ann, image) for ann in anns), default=0.0)
        out_name = (
            f'{idx:03d}_id{image.get("id")}_'
            f'max{max_ratio:.3f}_{safe_name(Path(str(image.get("file_name"))).stem)}.jpg')
        out_path = dataset_out / out_name
        draw_annotations(src_path, image, anns, categories, threshold, out_path)
        manifest.append({
            'image_id': image.get('id'),
            'file_name': image.get('file_name'),
            'source_path': str(src_path),
            'output_path': str(out_path),
            'resolve_mode': resolve_mode,
            'annotations': len(anns),
            'max_ratio': max_ratio,
        })

    dataset_out.mkdir(parents=True, exist_ok=True)
    with open(dataset_out / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(
        f'{name}: sampled={len(sampled)}, visualized={len(sampled) - missing}, '
        f'missing_images={missing}, output={dataset_out}')
    return {
        'name': name,
        'status': 'ok',
        'annotation': str(ann_path),
        'image_root': str(image_root),
        'images': len(images),
        'annotations': len(coco.get('annotations', [])),
        'sampled': len(sampled),
        'visualized': len(sampled) - missing,
        'missing_images': missing,
        'output': str(dataset_out),
    }


def main():
    args = parse_args()

    datasets = []
    if args.preset == 'exp2_bbox20pct':
        datasets.extend(EXP2_BBOX20_PRESET)
    if args.dataset:
        datasets.extend(args.dataset)
    if not datasets:
        raise ValueError('Use --preset exp2_bbox20pct or at least one --dataset triplet.')

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stats = []
    for name, ann_path, image_root in datasets:
        stats.append(visualize_dataset(
            name=name,
            ann_path=ann_path,
            image_root=image_root,
            out_dir=args.out_dir,
            num=args.num,
            seed=args.seed,
            threshold=args.threshold,
            recursive_search=args.recursive_search,
            strict=args.strict))

    with open(Path(args.out_dir) / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('\nSummary')
    print('=' * 96)
    for item in stats:
        print(
            f'{item.get("name", ""):18} '
            f'{item.get("status", ""):20} '
            f'images={item.get("images", "NA")} '
            f'sampled={item.get("sampled", "NA")} '
            f'visualized={item.get("visualized", "NA")} '
            f'missing={item.get("missing_images", "NA")} '
            f'out={item.get("output", "")}')


if __name__ == '__main__':
    main()
