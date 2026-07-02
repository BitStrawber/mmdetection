#!/usr/bin/env python3
"""Generate Depth Anything V2 pseudo-depth PNGs for an image tree.

The output mirrors ``--image-dir`` and always saves one-channel PNG depth maps
with the same spatial size as the original source image. This makes the maps
drop-in replacements for the MegaDepth maps used by SyreaNet, UWNR, and
WaterGAN preparation scripts.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,
             'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128,
             'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256,
             'out_channels': [256, 512, 1024, 1024]},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch-generate normalized relative depth maps with Depth Anything V2.')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--depthanything-dir', required=True,
                        help='Checkout of Depth-Anything-V2.')
    parser.add_argument('--checkpoint', required=True,
                        help='Depth Anything V2 checkpoint .pth.')
    parser.add_argument('--encoder', default='vitb', choices=sorted(MODEL_CONFIGS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Depth Anything V2 inference input size.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many images; 0 means all.')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Split the sorted image list into this many shards.')
    parser.add_argument('--shard-index', type=int, default=0,
                        help='Process only this shard index, 0-based.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--invert', action='store_true',
                        help='Invert normalized depth before saving.')
    parser.add_argument('--preserve-size-check', action='store_true', default=True,
                        help='Verify every saved depth map has the source image size.')
    return parser.parse_args()


def list_images(root):
    print(f'scanning images: {root}', flush=True)
    images = []
    for path in tqdm(root.rglob('*'), desc=f'scan {root.name}', unit='entry'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    images.sort()
    print(f'found images under {root}: {len(images)}', flush=True)
    return images


def load_model(depthanything_dir, checkpoint_path, encoder, device):
    depthanything_dir = str(Path(depthanything_dir).resolve())
    if depthanything_dir not in sys.path:
        sys.path.insert(0, depthanything_dir)

    try:
        module = importlib.import_module('depth_anything_v2.dpt')
    except ImportError as error:
        raise RuntimeError(
            f'Cannot import Depth Anything V2 code from {depthanything_dir}.') from error

    model = module.DepthAnythingV2(**MODEL_CONFIGS[encoder])
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported Depth Anything V2 checkpoint format.')
    state = {}
    for key, value in checkpoint.items():
        key = str(key)
        state[key[7:] if key.startswith('module.') else key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'Depth Anything V2 checkpoint does not match the selected encoder. '
            f'encoder={encoder}, missing={len(missing)}, unexpected={len(unexpected)}.')
    return model.to(device).eval()


def normalize_to_uint8(depth, invert=False):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError('Depth Anything V2 produced no finite values.')
    values = depth[finite]
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        out = np.zeros(depth.shape, dtype=np.uint8)
    else:
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        if invert:
            normalized = 1.0 - normalized
        out = np.round(normalized * 255.0).astype(np.uint8)
    return out


def load_pil_rgb(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        return image.copy()


def image_size(path):
    with Image.open(path) as image:
        return image.size


def infer_depth(model, image_path, input_size, target_size):
    image = load_pil_rgb(image_path)
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = model.infer_image(bgr, input_size)
    target_w, target_h = target_size
    if depth.shape[:2] != (target_h, target_w):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return depth


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    depthanything_dir = Path(args.depthanything_dir).resolve()

    for path, label in ((image_dir, 'image directory'), (checkpoint, 'checkpoint'),
                        (depthanything_dir, 'Depth Anything V2 directory')):
        if not path.exists():
            raise FileNotFoundError(f'{label} does not exist: {path}')
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        raise RuntimeError('CUDA device was requested but CUDA is unavailable.')

    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    images = list_images(image_dir)
    if args.limit:
        images = images[:args.limit]
    if args.num_shards <= 0:
        raise ValueError('--num-shards must be positive.')
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('--shard-index must satisfy 0 <= shard-index < num-shards.')
    if args.num_shards > 1:
        images = images[args.shard_index::args.num_shards]
    if not images:
        raise RuntimeError(f'No images found under {image_dir}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(depthanything_dir, checkpoint, args.encoder, device)

    written = skipped = failed = size_mismatch = 0
    failures = []
    for image_path in tqdm(images, desc='DepthAnythingV2', unit='image'):
        relative = image_path.relative_to(image_dir)
        out_path = (out_dir / relative).with_suffix('.png')
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            src_size = image_size(image_path)
            depth = infer_depth(model, image_path, args.input_size, src_size)
            depth_u8 = normalize_to_uint8(depth, invert=args.invert)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(depth_u8, mode='L').save(out_path)
            if args.preserve_size_check:
                out_size = image_size(out_path)
                if src_size != out_size:
                    size_mismatch += 1
                    raise RuntimeError(
                        f'depth size mismatch: source={src_size}, depth={out_size}')
            written += 1
        except Exception as error:
            failed += 1
            failures.append({'image': str(image_path), 'error': repr(error)})

    summary = {
        'image_dir': str(image_dir),
        'out_dir': str(out_dir),
        'depthanything_dir': str(depthanything_dir),
        'checkpoint': str(checkpoint),
        'encoder': args.encoder,
        'device': str(device),
        'input_size': args.input_size,
        'total_images': len(images),
        'written': written,
        'skipped_existing': skipped,
        'failed': failed,
        'size_mismatch': size_mismatch,
        'limit': args.limit,
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'invert': args.invert,
        'depth_semantics': (
            'Depth Anything V2 relative depth normalized per image; '
            'larger predicted values are saved brighter before optional invert. '
            'Use smoke tests to choose invert/non-invert for physical synthesis.'
        ),
        'preserve_size': 'saved depth PNG size is verified against source image size',
        'failures': failures[:200],
        'failure_count_total': len(failures),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / (
        f'depthanything_v2_{args.encoder}_'
        f'shard{args.shard_index}of{args.num_shards}_summary.json'
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'summary: {summary_path}')
    if failed:
        raise SystemExit(f'Failed to generate {failed} depth maps.')


if __name__ == '__main__':
    main()
