#!/usr/bin/env python3
"""Generate deterministic clean/low/mid/high frequency image variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .common import (
    ensure_empty_or_create,
    existing_file,
    read_jsonl,
    validate_sample_order,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--bands', default='low:0.0:0.15,mid:0.15:0.40,high:0.40:1.0',
        help='Comma-separated NAME:LOW:HIGH normalized radial bands')
    parser.add_argument(
        '--reconstruction', choices=('mean-preserve', 'clip', 'rescale'),
        default='mean-preserve')
    parser.add_argument(
        '--save-float', action=argparse.BooleanOptionalAction, default=False,
        help='Save pre-quantization float32 arrays next to PNG variants')
    parser.add_argument(
        '--copy-clean', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def parse_bands(value: str) -> List[Tuple[str, float, float]]:
    result = []
    names = set()
    for item in value.split(','):
        fields = item.strip().split(':')
        if len(fields) != 3:
            raise ValueError(f'Invalid band specification: {item}')
        name, low_text, high_text = fields
        low, high = float(low_text), float(high_text)
        if not name or name == 'clean' or name in names:
            raise ValueError(f'Invalid or duplicate band name: {name}')
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f'Band must satisfy 0 <= low < high <= 1: {item}')
        names.add(name)
        result.append((name, low, high))
    if not result:
        raise ValueError('At least one frequency band is required')
    return result


def radial_frequency(height: int, width: int) -> np.ndarray:
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    radius = np.sqrt(xx * xx + yy * yy)
    maximum = float(radius.max())
    return radius / maximum if maximum > 0 else radius


def reconstruct_band(
    image: np.ndarray,
    low: float,
    high: float,
    reconstruction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    radius = radial_frequency(height, width)
    mask = ((radius >= low) & (radius < high)).astype(np.float32)
    if high >= 1.0:
        mask[radius <= high] = (radius[radius <= high] >= low)
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[2]):
        source = image[:, :, channel]
        spectrum = np.fft.fftshift(np.fft.fft2(source))
        reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
        if reconstruction == 'mean-preserve' and low > 0.0:
            reconstructed += float(source.mean())
        elif reconstruction == 'rescale':
            minimum, maximum = float(reconstructed.min()), float(reconstructed.max())
            reconstructed = (
                (reconstructed - minimum) / max(maximum - minimum, 1e-12))
        output[:, :, channel] = reconstructed
    return np.clip(output, 0.0, 1.0).astype(np.float32), mask


def save_png(path: Path, value: np.ndarray, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(quantized, mode='RGB').save(
        path, format='PNG', compress_level=compress_level)


def main() -> None:
    args = parse_args()
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be between 0 and 9')
    rows = read_jsonl(args.manifest)
    validate_sample_order(rows)
    bands = parse_bands(args.bands)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    output_rows = []
    shape_counts: Dict[str, int] = {}

    for position, row in enumerate(rows, start=1):
        source = existing_file(row['image_path'])
        with Image.open(source) as opened:
            rgb = opened.convert('RGB')
            image = np.asarray(rgb, dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        shape_key = f'{height}x{width}'
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        sample_name = f'{int(row["sample_index"]):05d}_{int(row["image_id"])}'
        variants: Dict[str, dict] = {}
        clean_path = out_dir / 'images' / 'clean' / f'{sample_name}.png'
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        if args.copy_clean:
            save_png(clean_path, image, args.png_compress_level)
        else:
            if clean_path.exists() or clean_path.is_symlink():
                clean_path.unlink()
            clean_path.symlink_to(source)
        variants['clean'] = {'image_path': str(clean_path.absolute())}

        for name, low, high in bands:
            filtered, mask = reconstruct_band(image, low, high, args.reconstruction)
            image_path = out_dir / 'images' / name / f'{sample_name}.png'
            save_png(image_path, filtered, args.png_compress_level)
            variant = {
                'image_path': str(image_path.absolute()),
                'low': low,
                'high': high,
                'source_mean': [float(v) for v in image.mean(axis=(0, 1))],
                'output_mean': [float(v) for v in filtered.mean(axis=(0, 1))],
                'output_std': [float(v) for v in filtered.std(axis=(0, 1))],
                'mask_fraction': float(mask.mean()),
            }
            if args.save_float:
                float_path = out_dir / 'arrays' / name / f'{sample_name}.npy'
                float_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(float_path, filtered, allow_pickle=False)
                variant['float_path'] = str(float_path.absolute())
            variants[name] = variant
        output = dict(row)
        output['variants'] = variants
        output_rows.append(output)
        print(f'[{position}/{len(rows)}] {row["file_name"]}', flush=True)

    write_jsonl(out_dir / 'frequency_manifest.jsonl', output_rows)
    write_json(out_dir / 'filter_config.json', {
        'source_manifest': str(existing_file(args.manifest)),
        'bands': [
            {'name': name, 'low': low, 'high': high}
            for name, low, high in bands
        ],
        'frequency_radius': 'fftshift radial distance normalized by corner Nyquist',
        'reconstruction': args.reconstruction,
        'save_float': args.save_float,
        'png_quantization': 'clip [0,1], round to uint8',
        'shape_counts': shape_counts,
    })
    print(f'Frequency variants completed: {out_dir}')
    print(f'Manifest: {out_dir / "frequency_manifest.jsonl"}')


if __name__ == '__main__':
    main()
