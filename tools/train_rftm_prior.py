"""Stage-1 HDP/RFTM feature transference training for J10.

This is the paper-style first stage: it does not train a detector. It freezes
an ImageNet pretrained ResNet-50 shallow feature extractor and only optimizes
RFTM so that heavily-degraded RUOD patch features move toward detector-friendly
easy/DFUI patch features.

The saved checkpoint contains keys matching MMDetection's detector checkpoint
format, e.g. ``backbone.rftm.conv1.weight``. It can be loaded by the S2 config
through ``load_from``.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError:  # pragma: no cover
    from torchvision.models import resnet50
    ResNet50_Weights = None

from mmdet.models.backbones.resnet_rftm import RFTM


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy-patch-dir', required=True, help='HDf patch dir.')
    parser.add_argument('--ruod-patch-dir', required=True, help='HDu patch dir.')
    parser.add_argument('--out', required=True, help='Output checkpoint path.')
    parser.add_argument('--work-dir', default='work_dirs/j10_hdp_s1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--loss',
        choices=['kl', 'mse', 'kl_mse'],
        default='kl_mse',
        help='Feature alignment objective.')
    parser.add_argument('--mse-weight', type=float, default=0.1)
    return parser.parse_args()


def list_images(root):
    root = Path(root)
    if (root / 'patches').is_dir():
        root = root / 'patches'
    files = []
    for suffix in IMG_SUFFIXES:
        files.extend(root.rglob(f'*{suffix}'))
        files.extend(root.rglob(f'*{suffix.upper()}'))
    return sorted(set(files))


class RandomPairPatchDataset(Dataset):
    def __init__(self, easy_dir, ruod_dir, image_size):
        self.easy_files = list_images(easy_dir)
        self.ruod_files = list_images(ruod_dir)
        if not self.easy_files:
            raise RuntimeError(f'No easy/DFUI patches found in {easy_dir}')
        if not self.ruod_files:
            raise RuntimeError(f'No RUOD patches found in {ruod_dir}')

        self.length = max(len(self.easy_files), len(self.ruod_files))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.length

    def _load(self, path):
        with Image.open(path) as img:
            return self.transform(img.convert('RGB'))

    def __getitem__(self, idx):
        ruod_path = self.ruod_files[idx % len(self.ruod_files)]
        easy_path = random.choice(self.easy_files)
        return self._load(ruod_path), self._load(easy_path)


class FrozenResNetStage1(nn.Module):
    """ImageNet ResNet stem + layer1 feature extractor.

    The output is the feature tensor that RFTM transforms in this repo's
    ResNetWithRFTM implementation.
    """

    def __init__(self):
        super().__init__()
        if ResNet50_Weights is not None:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:  # compatible with older torchvision
            backbone = resnet50(pretrained=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.stem(x)


def feature_kl_loss(source, target):
    b = source.shape[0]
    source_flat = source.reshape(b, -1)
    target_flat = target.reshape(b, -1)
    return F.kl_div(
        F.log_softmax(source_flat, dim=1),
        F.softmax(target_flat.detach(), dim=1),
        reduction='batchmean')


def save_rftm_checkpoint(rftm, out_path, meta):
    state_dict = {}
    for key, value in rftm.state_dict().items():
        state_dict[f'backbone.rftm.{key}'] = value.detach().cpu()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state_dict, 'meta': meta}, out_path)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    dataset = RandomPairPatchDataset(
        args.easy_patch_dir, args.ruod_patch_dir, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True)

    extractor = FrozenResNetStage1().to(device)
    rftm = RFTM(in_channels=256).to(device)
    optimizer = torch.optim.AdamW(
        rftm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        'method': 'j10_hdp_rftm_prior',
        'easy_patch_dir': args.easy_patch_dir,
        'ruod_patch_dir': args.ruod_patch_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'loss': args.loss,
        'mse_weight': args.mse_weight,
        'image_size': args.image_size,
    }
    with open(Path(args.work_dir) / 'train_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    best_loss = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        rftm.train()
        running = 0.0
        pbar = tqdm(dataloader, desc=f'epoch {epoch}/{args.epochs}')
        for step, (ruod_img, easy_img) in enumerate(pbar, start=1):
            ruod_img = ruod_img.to(device, non_blocking=True)
            easy_img = easy_img.to(device, non_blocking=True)

            with torch.no_grad():
                ruod_feat = extractor(ruod_img)
                easy_feat = extractor(easy_img)

            transferred = rftm(ruod_feat)
            if args.loss == 'kl':
                loss = feature_kl_loss(transferred, easy_feat)
            elif args.loss == 'mse':
                loss = F.mse_loss(transferred, easy_feat.detach())
            else:
                loss = feature_kl_loss(transferred, easy_feat)
                loss = loss + args.mse_weight * F.mse_loss(
                    transferred, easy_feat.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running += float(loss.item())
            if step % args.log_interval == 0:
                pbar.set_postfix(loss=running / step)

        epoch_loss = running / max(1, len(dataloader))
        print(f'Epoch {epoch}: loss={epoch_loss:.6f}')

        latest_path = Path(args.work_dir) / 'latest_rftm.pth'
        save_rftm_checkpoint(rftm, latest_path, {**meta, 'epoch': epoch})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(args.work_dir) / 'best_rftm.pth'
            save_rftm_checkpoint(
                rftm, best_path, {**meta, 'epoch': epoch, 'best_loss': best_loss})

    save_rftm_checkpoint(rftm, args.out, {**meta, 'best_loss': best_loss})
    print(f'Saved RFTM checkpoint to {args.out}')


if __name__ == '__main__':
    main()
