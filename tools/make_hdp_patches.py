"""Build heavily-degraded patch sets for HDP/RFTM pretraining.

This script estimates an underwater transmission map with a lightweight UDCP
approximation and crops patches whose mean transmission is below a threshold.
It is intended for the first stage of:

    Learning Heavily-Degraded Prior for Underwater Object Detection

Example:
    python tools/make_hdp_patches.py \
        --img-dir /media/HDD0/XCX/exp_2/RUOD/coco/train \
        --out-dir /media/HDD0/XCX/exp_2/HDP/ruod_hd_t06 \
        --threshold 0.6

    python tools/make_hdp_patches.py \
        --img-dir /media/HDD0/XCX/exp_2/RUOD/coco/easy \
        --out-dir /media/HDD0/XCX/exp_2/HDP/easy_hd_t06 \
        --threshold 0.6
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True, help='Image root.')
    parser.add_argument(
        '--ann',
        default=None,
        help='Optional COCO annotation. If set, only images in the json are used.')
    parser.add_argument('--out-dir', required=True, help='Patch output dir.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Transmission threshold. Patches with mean t < threshold are HD.')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument(
        '--min-hd-ratio',
        type=float,
        default=0.35,
        help='Minimum ratio of pixels whose t is below threshold in a patch.')
    parser.add_argument('--max-per-image', type=int, default=8)
    parser.add_argument('--omega', type=float, default=0.95)
    parser.add_argument('--dark-kernel', type=int, default=15)
    parser.add_argument('--min-size', type=int, default=128)
    parser.add_argument('--save-debug-map', action='store_true')
    return parser.parse_args()


def collect_images(img_dir, ann=None):
    img_dir = Path(img_dir)
    if ann:
        with open(ann, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        files = []
        for item in coco['images']:
            path = img_dir / item['file_name']
            if not path.exists():
                path = img_dir / os.path.basename(item['file_name'])
            files.append(path)
        return [p for p in files if p.suffix.lower() in IMG_SUFFIXES]

    files = []
    for suffix in IMG_SUFFIXES:
        files.extend(img_dir.rglob(f'*{suffix}'))
        files.extend(img_dir.rglob(f'*{suffix.upper()}'))
    return sorted(set(files))


def dark_channel_udcp(img_rgb, kernel_size):
    """Underwater dark channel using green/blue channels.

    The original UDCP excludes the red channel because red attenuates quickly
    underwater. This approximation is sufficient for ranking HD/LD regions.
    """
    gb_min = np.min(img_rgb[:, :, 1:3], axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(gb_min, kernel)


def estimate_transmission_udcp(img_bgr, omega=0.95, kernel_size=15):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    dark = dark_channel_udcp(img_rgb, kernel_size)

    flat_dark = dark.reshape(-1)
    flat_img = img_rgb.reshape(-1, 3)
    top_k = max(1, int(flat_dark.size * 0.001))
    top_idx = np.argpartition(flat_dark, -top_k)[-top_k:]
    brightest = top_idx[np.argmax(flat_img[top_idx].sum(axis=1))]
    atmospheric = np.maximum(flat_img[brightest], 1e-3)

    normalized = img_rgb / atmospheric.reshape(1, 1, 3)
    dark_norm = dark_channel_udcp(np.clip(normalized, 0.0, 1.0), kernel_size)
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.0, 1.0)


def patch_windows(height, width, patch_size, stride):
    if height < patch_size or width < patch_size:
        return
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    for y in ys:
        for x in xs:
            yield x, y, x + patch_size, y + patch_size


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    patch_dir = out_dir / 'patches'
    debug_dir = out_dir / 'debug_t'
    patch_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_map:
        debug_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.img_dir, args.ann)
    metadata = {
        'source_img_dir': args.img_dir,
        'ann': args.ann,
        'threshold': args.threshold,
        'patch_size': args.patch_size,
        'stride': args.stride,
        'min_hd_ratio': args.min_hd_ratio,
        'max_per_image': args.max_per_image,
        'patches': [],
    }

    saved = 0
    skipped = 0
    for img_path in tqdm(images, desc='making HD patches'):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_size:
            skipped += 1
            continue

        t_map = estimate_transmission_udcp(
            img, omega=args.omega, kernel_size=args.dark_kernel)

        if args.save_debug_map:
            t_vis = (t_map * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f'{img_path.stem}_t.png'), t_vis)

        candidates = []
        for x1, y1, x2, y2 in patch_windows(h, w, args.patch_size, args.stride):
            t_patch = t_map[y1:y2, x1:x2]
            hd_ratio = float(np.mean(t_patch < args.threshold))
            mean_t = float(np.mean(t_patch))
            if mean_t < args.threshold and hd_ratio >= args.min_hd_ratio:
                candidates.append((mean_t, -hd_ratio, x1, y1, x2, y2))

        candidates.sort(key=lambda item: (item[0], item[1]))
        for rank, (mean_t, neg_hd_ratio, x1, y1, x2, y2) in enumerate(
                candidates[:args.max_per_image]):
            patch = img[y1:y2, x1:x2]
            name = f'{img_path.stem}_hd{rank:02d}_t{mean_t:.3f}_{x1}_{y1}.jpg'
            cv2.imwrite(str(patch_dir / name), patch)
            metadata['patches'].append({
                'file_name': f'patches/{name}',
                'source': str(img_path),
                'bbox_xyxy': [x1, y1, x2, y2],
                'mean_t': mean_t,
                'hd_ratio': -neg_hd_ratio,
            })
            saved += 1

    metadata['num_source_images'] = len(images)
    metadata['num_patches'] = saved
    metadata['num_skipped_images'] = skipped
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'Done. source images={len(images)}, patches={saved}, skipped={skipped}')
    print(f'Patch dir: {patch_dir}')


if __name__ == '__main__':
    main()
