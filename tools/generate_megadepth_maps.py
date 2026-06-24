#!/usr/bin/env python3
"""Generate MegaDepth pseudo-depth PNGs for an image directory.

The script intentionally keeps the official MegaDepth model external. Clone
https://github.com/zhengqili/MegaDepth and download its
``best_generalization_net_G.pth`` checkpoint before running this tool.

The output mirrors the input directory hierarchy, but depth files use the
``.png`` suffix. Values represent normalized relative distance (farther pixels
are brighter); they are suitable as a common depth source for UWNR, WaterGAN,
and SyreaNet preprocessing.
"""
import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch-generate normalized relative depth maps with MegaDepth.')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--megadepth-dir', required=True,
                        help='Checkout of zhengqili/MegaDepth.')
    parser.add_argument('--checkpoint', required=True,
                        help='Official best_generalization_net_G.pth checkpoint.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-height', type=int, default=384)
    parser.add_argument('--input-width', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many images; 0 means all.')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Split the sorted image list into this many shards.')
    parser.add_argument('--shard-index', type=int, default=0,
                        help='Process only this shard index, 0-based.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def list_images(root):
    return sorted(path for path in root.rglob('*')
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def checkpoint_state(checkpoint_path):
    # ``weights_only=False`` is required by older official MegaDepth checkpoints.
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported MegaDepth checkpoint format.')
    state = {}
    for key, value in checkpoint.items():
        key = str(key)
        state[key[7:] if key.startswith('module.') else key] = value
    return state


def load_model(megadepth_dir, checkpoint_path, device):
    megadepth_dir = str(Path(megadepth_dir).resolve())
    if megadepth_dir not in sys.path:
        sys.path.insert(0, megadepth_dir)

    try:
        module = importlib.import_module('pytorch_DIW_scratch')
    except ImportError as error:
        raise RuntimeError(
            f'Cannot import official MegaDepth code from {megadepth_dir}.') from error

    # The official module exports one global Sequential instance. Deep-copy it
    # so this process never mutates a module-level shared network.
    model = copy.deepcopy(module.pytorch_DIW_scratch)
    state = checkpoint_state(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'MegaDepth checkpoint does not match the official architecture. '
            f'Missing={len(missing)}, unexpected={len(unexpected)}.')
    return model.to(device).eval()


def depth_to_uint8(depth):
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError('MegaDepth produced no finite values.')
    values = depth[finite]
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def infer_depth(model, image_path, device, input_height, input_width):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        original_size = image.size
        resampling = getattr(Image, 'Resampling', Image)
        array = np.asarray(image.resize((input_width, input_height), resampling.BILINEAR),
                           dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        log_depth = model(tensor)
        depth = torch.exp(log_depth)
        depth = F.interpolate(depth, size=(original_size[1], original_size[0]),
                              mode='bilinear', align_corners=False)
    return depth.squeeze().cpu().numpy()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    megadepth_dir = Path(args.megadepth_dir).resolve()

    for path, label in ((image_dir, 'image directory'), (checkpoint, 'checkpoint'),
                        (megadepth_dir, 'MegaDepth directory')):
        if not path.exists():
            raise FileNotFoundError(f'{label} does not exist: {path}')
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        raise RuntimeError('CUDA device was requested but CUDA is unavailable.')

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
    model = load_model(megadepth_dir, checkpoint, device)

    written = skipped = failed = 0
    failures = []
    for image_path in tqdm(images, desc='MegaDepth', unit='image'):
        relative = image_path.relative_to(image_dir)
        out_path = (out_dir / relative).with_suffix('.png')
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            depth = infer_depth(model, image_path, device,
                                args.input_height, args.input_width)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(depth_to_uint8(depth), mode='L').save(out_path)
            written += 1
        except Exception as error:  # Keep a long batch job progressing.
            failed += 1
            failures.append({'image': str(image_path), 'error': repr(error)})

    summary = {
        'image_dir': str(image_dir), 'out_dir': str(out_dir),
        'megadepth_dir': str(megadepth_dir), 'checkpoint': str(checkpoint),
        'device': str(device), 'total_images': len(images), 'written': written,
        'skipped_existing': skipped, 'failed': failed,
        'limit': args.limit, 'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'depth_semantics': 'normalized relative distance; farther pixels are brighter',
        'failures': failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'megadepth_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
