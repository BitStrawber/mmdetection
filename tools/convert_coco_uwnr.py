"""Convert clean COCO images to underwater images using UWNR and MegaDepth maps.

Usage:
    python tools/convert_coco_uwnr.py \
        --ann /path/to/instances_train50k.json \
        --img-dir /path/to/train2017 \
        --output-dir /path/to/coco_uwnr \
        --uwnr-dir /path/to/UWNR \
        --uwnr-model /path/to/uwnr_epoch200.pth \
        --depth-dir /path/to/megadepth_maps

``--depth-dir`` must mirror the image directory hierarchy. A source image
``train/n01440764/foo.JPEG`` therefore uses
``train/n01440764/foo.png`` under the depth directory. Generate those maps
with ``tools/generate_megadepth_maps.py``.
"""
import argparse
import os
import sys
import json
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


def load_uwnr_generator(model_path, uwnr_dir, device):
    sys.path.insert(0, uwnr_dir)
    from model.FSU2 import Generator
    netG = Generator()
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt['G1'] if 'G1' in ckpt else ckpt
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace('module.', '', 1)] = v
    netG.load_state_dict(new_state)
    netG.to(device)
    netG.eval()
    return netG


def _compute_a_map(img_rgb):
    from myutils.dcp import MutiScaleLuminanceEstimation
    return MutiScaleLuminanceEstimation(img_rgb)


def resolve_depth_path(depth_dir, file_name):
    relative = os.path.normpath(file_name)
    stem, _ = os.path.splitext(relative)
    candidates = [
        os.path.join(depth_dir, stem + '.png'),
        os.path.join(depth_dir, relative),
        os.path.join(depth_dir, os.path.basename(stem) + '.png'),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def process_single_image(img_path, depth_path, netG, device, size):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h_orig, w_orig = img.shape[:2]

    img_resized = cv2.resize(img, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    A_map = _compute_a_map(img_rgb)
    A_map_tensor = transforms.ToTensor()(np.float32(A_map) / 255.0)

    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        return None
    depth = cv2.resize(depth, (size, size)).astype(np.float32) / 255.0

    depth_tensor = torch.from_numpy(depth).unsqueeze(0)
    img_tensor = transforms.ToTensor()(img_rgb)

    x = torch.cat([img_tensor, depth_tensor, A_map_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = netG(x)

    output = output.squeeze(0).cpu()
    output = (output + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    output_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    output_np = cv2.resize(output_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return output_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--uwnr-dir', required=True)
    parser.add_argument('--uwnr-model', required=True)
    parser.add_argument('--depth-dir', required=True,
                        help='MegaDepth PNG root mirroring --img-dir.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.ann, 'r') as f:
        coco = json.load(f)
    images = coco['images']
    print(f'Images to process: {len(images)}')

    ann_out_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(ann_out_dir, exist_ok=True)
    ann_out = os.path.join(ann_out_dir, os.path.basename(args.ann))
    if not os.path.exists(ann_out):
        shutil.copy2(args.ann, ann_out)
        print(f'Copied annotation to {ann_out}')

    img_out_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)

    print(f'Loading UWNR model from {args.uwnr_model} ...')
    netG = load_uwnr_generator(args.uwnr_model, args.uwnr_dir, device)

    skipped = missing_depth = 0
    for i, img_info in enumerate(tqdm(images, desc='UWNR converting')):
        filename = img_info['file_name']
        src_path = os.path.join(args.img_dir, filename)
        dst_path = os.path.join(img_out_dir, filename)

        if os.path.exists(dst_path):
            continue

        depth_path = resolve_depth_path(args.depth_dir, filename)
        if depth_path is None:
            print(f'Warning: missing MegaDepth map for {filename}')
            missing_depth += 1
            continue

        result = process_single_image(
            src_path, depth_path, netG, device, args.size
        )

        if result is None:
            print(f'Warning: failed to read {src_path}')
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f'Done. Processed: {len(images) - skipped - missing_depth}, '
          f'failed_images: {skipped}, missing_depth: {missing_depth}')
    print(f'Output: {img_out_dir}')


if __name__ == '__main__':
    main()
