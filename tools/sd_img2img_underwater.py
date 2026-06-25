#!/usr/bin/env python3
"""Stable Diffusion img2img underwater baseline for sampled ImageNet.

This baseline intentionally uses only two conditioning signals:

1. Image guidance: the ImageNet image is encoded by Stable Diffusion's VAE into
   the latent space and used as the init latent.
2. Text guidance: an underwater prompt is encoded by CLIP and used as the text
   condition.

No depth, reference underwater image, ControlNet, or extra adapter is used here.
The output is restored to ImageNet-style class folders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate underwater-style ImageNet images with Stable Diffusion img2img.')
    parser.add_argument('--source-dir', required=True,
                        help='Sampled ImageNet source directory, e.g. .../uwnr/source/train.')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory restored as <out>/<synset>/<image>.png.')
    parser.add_argument('--model', default='runwayml/stable-diffusion-v1-5',
                        help='Diffusers model id/local directory, or a CompVis .ckpt/.safetensors file with --single-file.')
    parser.add_argument('--single-file', action='store_true',
                        help='Load --model with diffusers from_single_file, useful for original CompVis .ckpt weights.')
    parser.add_argument('--prompt', default=(
        'a realistic underwater photograph of the same scene, blue-green water, '
        'underwater haze, natural color attenuation, low contrast, realistic lighting'
    ))
    parser.add_argument('--negative-prompt', default=(
        'cartoon, painting, illustration, deformed object, extra objects, fish, coral, '
        'diver, text, watermark, blurry, low quality'
    ))
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20,
                        help='Latent diffusion denoising steps. This is the default "latent 20" setting.')
    parser.add_argument('--strength', type=float, default=0.35,
                        help='Img2img noise strength. Lower preserves ImageNet semantics better.')
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many images before sharding; 0 means all.')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-manifest', default='',
                        help='Optional JSONL manifest path. Defaults to <out-dir>/sd_img2img_manifest.jsonl.')
    parser.add_argument('--disable-safety-checker', action='store_true',
                        help='Disable diffusers safety checker for local dataset generation.')
    parser.add_argument('--enable-attention-slicing', action='store_true',
                        help='Reduce memory usage at some speed cost.')
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    print(f'scanning images: {root}', flush=True)
    images = []
    for path in tqdm(root.rglob('*'), desc=f'scan {root.name}', unit='entry'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    images.sort()
    print(f'found images under {root}: {len(images)}', flush=True)
    return images


def prepare_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert('RGB')
    # Use a square canvas so img2img batching is deterministic and labels stay visible.
    image = ImageOps.fit(
        image,
        size,
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return image


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError('--shard-index must be in [0, num_shards)')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.save_manifest) if args.save_manifest else out_dir / 'sd_img2img_manifest.jsonl'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(source_dir)
    if args.limit > 0:
        images = images[:args.limit]
    images = images[args.shard_index::args.num_shards]

    print('Stable Diffusion img2img underwater baseline')
    print('=' * 80)
    print(f'source_dir:       {source_dir}')
    print(f'out_dir:          {out_dir}')
    print(f'model:            {args.model}')
    print(f'prompt:           {args.prompt}')
    print(f'negative_prompt:  {args.negative_prompt}')
    print(f'size:             {args.width}x{args.height}')
    print(f'steps:            {args.steps}')
    print(f'strength:         {args.strength}')
    print(f'guidance_scale:   {args.guidance_scale}')
    print(f'batch_size:       {args.batch_size}')
    print(f'device:           {args.device}')
    print(f'fp16:             {args.fp16}')
    print(f'num_shards:       {args.num_shards}')
    print(f'shard_index:      {args.shard_index}')
    print(f'images:           {len(images)}')
    print('=' * 80)

    if not images:
        raise SystemExit('No images selected.')

    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if args.fp16 and str(args.device).startswith('cuda') else torch.float32
    if args.single_file:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            args.model,
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
        )
    if args.disable_safety_checker:
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    pipe = pipe.to(args.device)
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()

    generator = torch.Generator(device=args.device if str(args.device).startswith('cuda') else 'cpu')
    generator.manual_seed(args.seed + args.shard_index)

    size = (args.width, args.height)
    records = []
    written = 0
    skipped = 0
    failed = []

    progress = tqdm(
        batched(images, args.batch_size),
        total=math.ceil(len(images) / args.batch_size),
        desc='SD img2img underwater',
        unit='batch',
    )
    for batch_paths in progress:
        batch_inputs = []
        valid_paths = []
        for image_path in batch_paths:
            rel = image_path.relative_to(source_dir)
            synset = rel.parts[0] if len(rel.parts) > 1 else 'unknown'
            dst = out_dir / synset / f'{image_path.stem}.png'
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                batch_inputs.append(prepare_image(image_path, size))
                valid_paths.append(image_path)
            except Exception as exc:
                failed.append({'source': str(image_path), 'error': repr(exc)})

        if not valid_paths:
            continue

        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=[args.prompt] * len(valid_paths),
                    negative_prompt=[args.negative_prompt] * len(valid_paths) if args.negative_prompt else None,
                    image=batch_inputs,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps,
                    generator=generator,
                )
        except Exception as exc:
            for image_path in valid_paths:
                failed.append({'source': str(image_path), 'error': repr(exc)})
            continue

        for image_path, generated in zip(valid_paths, result.images):
            rel = image_path.relative_to(source_dir)
            synset = rel.parts[0] if len(rel.parts) > 1 else 'unknown'
            dst = out_dir / synset / f'{image_path.stem}.png'
            dst.parent.mkdir(parents=True, exist_ok=True)
            generated.save(dst)
            written += 1
            records.append({
                'source': str(image_path),
                'relative': str(rel).replace('\\', '/'),
                'synset': synset,
                'output': str(dst),
                'prompt': args.prompt,
                'negative_prompt': args.negative_prompt,
                'steps': args.steps,
                'strength': args.strength,
                'guidance_scale': args.guidance_scale,
                'seed': args.seed,
                'shard_index': args.shard_index,
                'num_shards': args.num_shards,
                'conditioning': ['vae_image_latent', 'text_prompt_embedding'],
            })
        progress.set_postfix(written=written, skipped=skipped, failed=len(failed))

    mode = 'a' if args.num_shards > 1 else 'w'
    with manifest_path.open(mode, encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = {
        'source_dir': str(source_dir),
        'out_dir': str(out_dir),
        'model': args.model,
        'single_file': args.single_file,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'steps': args.steps,
        'strength': args.strength,
        'guidance_scale': args.guidance_scale,
        'batch_size': args.batch_size,
        'selected_images': len(images),
        'written': written,
        'skipped_existing': skipped,
        'failed': len(failed),
        'failures': failed[:20],
        'conditioning': ['vae_image_latent', 'text_prompt_embedding'],
        'manifest': str(manifest_path),
    }
    summary_path = out_dir / f'sd_img2img_summary_shard{args.shard_index}of{args.num_shards}.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
