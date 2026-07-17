#!/usr/bin/env python
"""Extract ResNet-50 feature maps for random ImageNet and RUOD samples.

This script is designed for a small qualitative comparison:

1. Randomly sample 10 ImageNet images and extract features with torchvision
   ImageNet-supervised ResNet-50.
2. Randomly sample 10 RUOD images and extract features with the backbone of a
   supervised Cascade R-CNN checkpoint trained with a ResNet-50 backbone.

The outputs include raw feature tensors, pooled feature vectors, copied source
images, simple feature heatmaps, and a manifest for traceability.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract random10 ImageNet/RUOD ResNet50 feature maps.')
    parser.add_argument(
        '--imagenet-root',
        default='/media/SSD1/XCX/exp_2/IMAGENET1K/imagefolder/train',
        help='ImageNet ImageFolder train root.')
    parser.add_argument(
        '--ruod-root',
        default='/media/HDD0/XCX/exp_2/RUOD',
        help='RUOD root or image directory. The script recursively finds images.')
    parser.add_argument(
        '--cascade-config',
        default='',
        help='MMDetection Cascade R-CNN config for the supervised ResNet50 model.')
    parser.add_argument(
        '--cascade-checkpoint',
        default='',
        help='MMDetection Cascade R-CNN checkpoint trained with ResNet50 backbone.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/exp_2/feature_maps/random10_imagenet_ruod',
        help='Output directory.')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--imagenet-layers',
        default='layer1,layer2,layer3,layer4',
        help='Comma-separated torchvision ResNet layers to save.')
    parser.add_argument(
        '--ruod-max-side',
        type=int,
        default=1333,
        help='Resize RUOD images so the max side is at most this value before feature extraction.')
    parser.add_argument(
        '--torchvision-weights',
        default='DEFAULT',
        help='Torchvision ResNet50 weights enum name. Use DEFAULT for current ImageNet weights.')
    return parser.parse_args()


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f'Image root does not exist: {root}')
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        return [root]
    images = [
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return sorted(images)


def sample_images(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if len(paths) < n:
        raise RuntimeError(f'Need at least {n} images, got {len(paths)}')
    rng = random.Random(seed)
    return rng.sample(list(paths), n)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def save_heatmap(feature: torch.Tensor, out_path: Path) -> None:
    """Save a simple mean-channel feature heatmap as PNG."""
    fmap = feature.detach().float().cpu()
    if fmap.ndim == 4:
        fmap = fmap[0]
    if fmap.ndim != 3:
        raise ValueError(f'Expected CHW feature, got shape {tuple(fmap.shape)}')
    heat = fmap.abs().mean(dim=0)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    heat_img = (heat.numpy() * 255).astype('uint8')
    Image.fromarray(heat_img).save(out_path)


def pooled_vector(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        return F.adaptive_avg_pool2d(feature, 1).flatten(1).detach().cpu()
    if feature.ndim == 3:
        return F.adaptive_avg_pool2d(feature.unsqueeze(0), 1).flatten(1).detach().cpu()
    raise ValueError(f'Unsupported feature shape: {tuple(feature.shape)}')


def build_torchvision_resnet50(weights_name: str, layers: Iterable[str], device: torch.device):
    if weights_name == 'DEFAULT':
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = getattr(models.ResNet50_Weights, weights_name)
    model = models.resnet50(weights=weights).to(device).eval()
    return_nodes = {layer: layer for layer in layers}
    extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device).eval()
    preprocess = weights.transforms()
    return extractor, preprocess


def extract_torchvision_features(
    samples: Sequence[Path],
    out_dir: Path,
    device: torch.device,
    layers: Sequence[str],
    weights_name: str,
) -> List[dict]:
    extractor, preprocess = build_torchvision_resnet50(weights_name, layers, device)
    records = []
    with torch.no_grad():
        for index, path in enumerate(samples, start=1):
            stem = f'{index:06d}_{path.stem}'
            img = load_rgb(path)
            x = preprocess(img).unsqueeze(0).to(device)
            feats = extractor(x)

            sample_dir = out_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, sample_dir / f'input{path.suffix.lower()}')

            save_obj = {}
            pooled_obj = {}
            for name, feat in feats.items():
                cpu_feat = feat.detach().cpu()
                save_obj[name] = cpu_feat
                pooled_obj[name] = pooled_vector(feat)
                save_heatmap(feat, sample_dir / f'{name}_heatmap.png')

            torch.save(save_obj, sample_dir / 'features.pt')
            torch.save(pooled_obj, sample_dir / 'pooled_features.pt')
            records.append({
                'subset': 'imagenet',
                'index': index,
                'source_path': str(path),
                'output_dir': str(sample_dir),
                'layers': ','.join(feats.keys()),
            })
    return records


def build_mmdet_model(config: str, checkpoint: str, device: str):
    if not config or not checkpoint:
        raise ValueError('--cascade-config and --cascade-checkpoint are required for RUOD feature extraction')
    from mmdet.apis import init_detector
    model = init_detector(config, checkpoint, device=device)
    model.eval()
    return model


def resize_keep_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / side
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.BILINEAR)


def mmdet_preprocess(
    img: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Apply the detector's configured channel conversion and normalization."""
    tensor = transforms.ToTensor()(img) * 255.0
    tensor = tensor[[2, 1, 0], :, :].to(device)  # RGB -> loader-style BGR
    processed = model.data_preprocessor(
        {'inputs': [tensor], 'data_samples': None},
        training=False,
    )
    return processed['inputs']


def extract_mmdet_backbone_features(
    samples: Sequence[Path],
    out_dir: Path,
    config: str,
    checkpoint: str,
    device: torch.device,
    max_side: int,
) -> List[dict]:
    model = build_mmdet_model(config, checkpoint, str(device))
    backbone = model.backbone
    backbone.eval()

    records = []
    with torch.no_grad():
        for index, path in enumerate(samples, start=1):
            stem = f'{index:06d}_{path.stem}'
            img = resize_keep_max_side(load_rgb(path), max_side)
            x = mmdet_preprocess(img, model, device)
            feats = backbone(x)
            if isinstance(feats, torch.Tensor):
                feats = (feats,)

            sample_dir = out_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, sample_dir / f'input{path.suffix.lower()}')

            save_obj = {}
            pooled_obj = {}
            layer_names = []
            for layer_idx, feat in enumerate(feats, start=1):
                name = f'backbone_stage{layer_idx}'
                layer_names.append(name)
                save_obj[name] = feat.detach().cpu()
                pooled_obj[name] = pooled_vector(feat)
                save_heatmap(feat, sample_dir / f'{name}_heatmap.png')

            torch.save(save_obj, sample_dir / 'features.pt')
            torch.save(pooled_obj, sample_dir / 'pooled_features.pt')
            records.append({
                'subset': 'ruod',
                'index': index,
                'source_path': str(path),
                'output_dir': str(sample_dir),
                'layers': ','.join(layer_names),
            })
    return records


def write_manifest(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['subset', 'index', 'source_path', 'output_dir', 'layers']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    imagenet_out = out_dir / 'imagenet_torchvision_resnet50'
    ruod_out = out_dir / 'ruod_supervised_cascade_resnet50'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    layers = [x.strip() for x in args.imagenet_layers.split(',') if x.strip()]

    imagenet_paths = find_images(Path(args.imagenet_root))
    ruod_paths = find_images(Path(args.ruod_root))
    imagenet_samples = sample_images(imagenet_paths, args.num_samples, args.seed)
    ruod_samples = sample_images(ruod_paths, args.num_samples, args.seed + 1000)

    records = []
    records.extend(extract_torchvision_features(
        imagenet_samples, imagenet_out, device, layers, args.torchvision_weights))
    records.extend(extract_mmdet_backbone_features(
        ruod_samples,
        ruod_out,
        args.cascade_config,
        args.cascade_checkpoint,
        device,
        args.ruod_max_side,
    ))
    write_manifest(records, out_dir / 'manifest.tsv')
    print(f'Wrote outputs to: {out_dir}')
    print(f'Wrote manifest: {out_dir / "manifest.tsv"}')


if __name__ == '__main__':
    main()
