#!/usr/bin/env python3
"""Generate reproducible frequency inputs and reconstruction QA artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class Band:
    name: str
    low: float
    high: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--method', choices=('soft-cpp', 'legacy-hard-corner'),
        default='soft-cpp',
        help='soft-cpp is the paper-ready partition-of-unity implementation')
    parser.add_argument(
        '--band-policy', choices=('fixed', 'dataset-energy'), default='fixed',
        help=(
            'fixed uses --bands; dataset-energy derives two cutoffs from the '
            'mean normalized dataset power spectrum'))
    parser.add_argument(
        '--bands', default='',
        help=(
            'Comma-separated NAME:LOW:HIGH bands. soft-cpp values are in '
            'cycles/pixel and the final HIGH may be "max". Defaults to '
            'low:0:1/32,mid:1/32:1/8,high:1/8:max.'))
    parser.add_argument(
        '--energy-quantiles', default='1/3,2/3',
        help='Two cumulative-energy quantiles for dataset-energy policy')
    parser.add_argument(
        '--energy-bins', type=int, default=1024,
        help='Radial cycles/pixel bins used for dataset-energy calibration')
    parser.add_argument(
        '--energy-color-space', choices=('rgb', 'luminance'), default='rgb',
        help='Signal used to estimate dataset frequency energy')
    parser.add_argument(
        '--calibration-manifest', default='',
        help=(
            'Optional independent sample manifest for dataset-energy cutoffs; '
            'defaults to --manifest'))
    parser.add_argument(
        '--transition-ratio', type=float, default=0.25,
        help='Raised-cosine width as a fraction of each interior cutoff')
    parser.add_argument(
        '--resize', default='1333x800',
        help='Keep-ratio MAX_WIDTHxMAX_HEIGHT before filtering, or none')
    parser.add_argument(
        '--pad-fraction', type=float, default=0.05,
        help='Reflect padding fraction applied before FFT')
    parser.add_argument(
        '--model-input-mode', choices=('natural-energy', 'equal-rms'),
        default='natural-energy')
    parser.add_argument(
        '--save-raw', action=argparse.BooleanOptionalAction, default=True,
        help='Save signed, unclipped float32 frequency bands')
    parser.add_argument(
        '--save-band-stop', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--save-visualizations', action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        '--reconstruction', choices=('mean-preserve', 'clip', 'rescale'),
        default='mean-preserve',
        help='Only used by legacy-hard-corner')
    parser.add_argument(
        '--copy-clean', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--reconstruction-tolerance', type=float, default=1e-5)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def parse_number(value: str) -> float:
    text = value.strip().lower()
    if '/' in text:
        numerator, denominator = text.split('/', maxsplit=1)
        result = float(numerator) / float(denominator)
    else:
        result = float(text)
    if not math.isfinite(result):
        raise ValueError(f'Frequency value is not finite: {value}')
    return result


def parse_bands(value: str, method: str) -> List[Band]:
    if not value:
        value = (
            'low:0:1/32,mid:1/32:1/8,high:1/8:max'
            if method == 'soft-cpp'
            else 'low:0:0.15,mid:0.15:0.40,high:0.40:1.0')
    bands = []
    names = set()
    for item in value.split(','):
        fields = item.strip().split(':')
        if len(fields) != 3:
            raise ValueError(f'Invalid band specification: {item}')
        name, low_text, high_text = fields
        low = parse_number(low_text)
        high = None if high_text.strip().lower() == 'max' else parse_number(high_text)
        if not name or name == 'clean' or name.startswith('remove_') or name in names:
            raise ValueError(f'Invalid or duplicate band name: {name}')
        if low < 0 or high is not None and high <= low:
            raise ValueError(f'Band must satisfy 0 <= low < high: {item}')
        names.add(name)
        bands.append(Band(name=name, low=low, high=high))
    if not bands:
        raise ValueError('At least one frequency band is required')
    if method == 'soft-cpp':
        if bands[0].low != 0:
            raise ValueError('soft-cpp bands must start at zero')
        if bands[-1].high is not None:
            raise ValueError('The final soft-cpp band must end at max')
        for left, right in zip(bands[:-1], bands[1:]):
            if left.high is None or not math.isclose(left.high, right.low):
                raise ValueError('soft-cpp bands must be ordered and contiguous')
    else:
        for band in bands:
            if band.high is None or band.high > 1:
                raise ValueError('Legacy normalized-corner bands must end at <= 1')
    return bands


def parse_energy_quantiles(value: str) -> Tuple[float, float]:
    fields = [parse_number(item) for item in value.split(',') if item.strip()]
    if len(fields) != 2 or not 0 < fields[0] < fields[1] < 1:
        raise ValueError(
            '--energy-quantiles must contain two values satisfying 0 < q1 < q2 < 1')
    return fields[0], fields[1]


def parse_resize(value: str) -> Optional[Tuple[int, int]]:
    text = value.strip().lower()
    if text in ('', 'none', 'off', 'original'):
        return None
    fields = text.split('x')
    if len(fields) != 2:
        raise ValueError('--resize must be MAX_WIDTHxMAX_HEIGHT or none')
    width, height = int(fields[0]), int(fields[1])
    if width <= 0 or height <= 0:
        raise ValueError('--resize dimensions must be positive')
    return width, height


def resize_keep_ratio(image: Image.Image, limit: Optional[Tuple[int, int]]) -> Image.Image:
    if limit is None:
        return image.copy()
    max_width, max_height = limit
    width, height = image.size
    scale = min(max_width / float(width), max_height / float(height))
    new_size = (
        max(1, int(width * scale + 0.5)),
        max(1, int(height * scale + 0.5)),
    )
    return image.resize(new_size, RESAMPLE_BILINEAR)


def radial_frequency_cpp(height: int, width: int) -> np.ndarray:
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    return np.sqrt(xx * xx + yy * yy).astype(np.float32)


def radial_frequency_corner(height: int, width: int) -> np.ndarray:
    radius = radial_frequency_cpp(height, width)
    maximum = float(radius.max())
    return radius / maximum if maximum > 0 else radius


def spectrum_signal(image: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == 'rgb':
        return image
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(image * weights[None, None, :], axis=2, keepdims=True)


def quantile_from_histogram(
    bin_edges: np.ndarray, cumulative: np.ndarray, quantile: float,
) -> float:
    index = min(int(np.searchsorted(cumulative, quantile, side='left')),
                len(cumulative) - 1)
    previous = float(cumulative[index - 1]) if index > 0 else 0.0
    current = float(cumulative[index])
    fraction = (
        (quantile - previous) / (current - previous)
        if current > previous else 0.5)
    return float(
        bin_edges[index] + fraction * (bin_edges[index + 1] - bin_edges[index]))


def derive_dataset_energy_bands(
    rows: Sequence[Mapping[str, object]],
    resize_limit: Optional[Tuple[int, int]],
    pad_fraction: float,
    quantiles: Tuple[float, float],
    bins: int,
    color_space: str,
) -> Tuple[List[Band], List[Mapping[str, float]], Mapping[str, object]]:
    if bins < 32:
        raise ValueError('--energy-bins must be at least 32')
    maximum_frequency = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    edges = np.linspace(0.0, maximum_frequency, bins + 1, dtype=np.float64)
    normalized_histograms = []
    for position, row in enumerate(rows, start=1):
        source = existing_file(str(row['image_path']))
        with Image.open(source) as opened:
            resized = resize_keep_ratio(opened.convert('RGB'), resize_limit)
        image = np.asarray(resized, dtype=np.float32) / 255.0
        signal = spectrum_signal(image, color_space)
        padded, _ = reflect_pad(signal, pad_fraction)
        centered = padded - padded.mean(axis=(0, 1), keepdims=True)
        window = (
            np.hanning(centered.shape[0])[:, None] *
            np.hanning(centered.shape[1])[None, :])
        windowed = centered * window[:, :, None]
        spectrum = np.fft.fftshift(
            np.fft.fft2(windowed, axes=(0, 1)), axes=(0, 1))
        power = np.sum(np.square(np.abs(spectrum)), axis=2)
        radius = radial_frequency_cpp(power.shape[0], power.shape[1])
        histogram, _ = np.histogram(
            radius.reshape(-1), bins=edges, weights=power.reshape(-1))
        total = float(histogram.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f'No finite spectral energy for {source}')
        normalized_histograms.append(histogram.astype(np.float64) / total)
        print(
            f'[energy calibration {position}/{len(rows)}] {row["file_name"]}',
            flush=True)
    mean_energy = np.mean(np.stack(normalized_histograms), axis=0)
    mean_energy /= max(float(mean_energy.sum()), 1e-12)
    cumulative = np.cumsum(mean_energy)
    first = quantile_from_histogram(edges, cumulative, quantiles[0])
    second = quantile_from_histogram(edges, cumulative, quantiles[1])
    if not 0 < first < second < maximum_frequency:
        raise RuntimeError(
            f'Invalid dataset-energy cutoffs derived: {first}, {second}')
    bands = [
        Band('low', 0.0, first),
        Band('mid', first, second),
        Band('high', second, None),
    ]
    profile = [
        {
            'bin_index': index,
            'low_cpp': float(edges[index]),
            'high_cpp': float(edges[index + 1]),
            'mean_energy_fraction': float(mean_energy[index]),
            'cumulative_energy_fraction': float(cumulative[index]),
        }
        for index in range(bins)
    ]
    metadata = {
        'samples': len(rows),
        'equal_sample_weight': True,
        'quantiles': list(quantiles),
        'cutoffs_cpp': [first, second],
        'cutoff_wavelengths_pixels': [1.0 / first, 1.0 / second],
        'bins': bins,
        'maximum_frequency_cpp': maximum_frequency,
        'color_space': color_space,
        'dc_removed_per_channel': True,
        'window': '2D Hann',
        'resize_limit': list(resize_limit) if resize_limit else None,
        'pad_fraction': pad_fraction,
        'padding_mode': 'reflect',
        'aggregation': (
            'normalize each image radial power histogram to unit energy, then mean'),
    }
    return bands, profile, metadata


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def raised_cosine_lowpass(
    radius: np.ndarray,
    cutoff: float,
    transition: float,
) -> np.ndarray:
    if transition <= 0:
        return (radius < cutoff).astype(np.float32)
    lower = max(0.0, cutoff - transition / 2.0)
    upper = cutoff + transition / 2.0
    result = np.ones_like(radius, dtype=np.float32)
    result[radius >= upper] = 0.0
    middle = (radius > lower) & (radius < upper)
    phase = (radius[middle] - lower) / max(upper - lower, 1e-12)
    result[middle] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return result


def soft_partition_masks(
    height: int,
    width: int,
    bands: Sequence[Band],
    transition_ratio: float,
) -> Mapping[str, np.ndarray]:
    radius = radial_frequency_cpp(height, width)
    cutoffs = [float(band.high) for band in bands[:-1]]
    lowpasses = [
        raised_cosine_lowpass(radius, cutoff, cutoff * transition_ratio)
        for cutoff in cutoffs
    ]
    masks = []
    if lowpasses:
        masks.append(lowpasses[0])
        masks.extend(
            np.maximum(upper - lower, 0.0)
            for lower, upper in zip(lowpasses[:-1], lowpasses[1:]))
        masks.append(1.0 - lowpasses[-1])
    else:
        masks.append(np.ones_like(radius, dtype=np.float32))
    stack = np.stack(masks).astype(np.float32)
    total = stack.sum(axis=0, keepdims=True)
    stack /= np.maximum(total, 1e-12)
    return {band.name: stack[index] for index, band in enumerate(bands)}


def legacy_masks(
    height: int,
    width: int,
    bands: Sequence[Band],
) -> Mapping[str, np.ndarray]:
    radius = radial_frequency_corner(height, width)
    result = {}
    for band in bands:
        high = float(band.high)
        mask = ((radius >= band.low) & (radius < high)).astype(np.float32)
        if high >= 1:
            mask = ((radius >= band.low) & (radius <= high)).astype(np.float32)
        result[band.name] = mask
    return result


def reflect_pad(image: np.ndarray, fraction: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]
    pad_y = min(max(0, int(round(height * fraction))), max(height - 1, 0))
    pad_x = min(max(0, int(round(width * fraction))), max(width - 1, 0))
    if pad_y == 0 and pad_x == 0:
        return image, (0, 0)
    return np.pad(
        image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='reflect'), (
            pad_y, pad_x)


def crop_padding(value: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    pad_y, pad_x = padding
    y_slice = slice(pad_y, -pad_y if pad_y else None)
    x_slice = slice(pad_x, -pad_x if pad_x else None)
    return value[y_slice, x_slice]


def decompose_soft(
    image: np.ndarray,
    bands: Sequence[Band],
    transition_ratio: float,
    pad_fraction: float,
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    padded, padding = reflect_pad(image, pad_fraction)
    masks = soft_partition_masks(
        padded.shape[0], padded.shape[1], bands, transition_ratio)
    spectrum = np.fft.fftshift(
        np.fft.fft2(padded, axes=(0, 1)), axes=(0, 1))
    outputs = {}
    for band in bands:
        filtered = np.fft.ifft2(
            np.fft.ifftshift(
                spectrum * masks[band.name][:, :, None], axes=(0, 1)),
            axes=(0, 1)).real
        outputs[band.name] = crop_padding(filtered, padding).astype(np.float32)
    return outputs, masks


def decompose_legacy(
    image: np.ndarray,
    bands: Sequence[Band],
    reconstruction: str,
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    masks = legacy_masks(image.shape[0], image.shape[1], bands)
    spectrum = np.fft.fftshift(
        np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))
    outputs = {}
    for band in bands:
        reconstructed = np.fft.ifft2(
            np.fft.ifftshift(
                spectrum * masks[band.name][:, :, None], axes=(0, 1)),
            axes=(0, 1)).real.astype(np.float32)
        if reconstruction == 'mean-preserve' and band.low > 0:
            reconstructed += image.mean(axis=(0, 1), keepdims=True)
        elif reconstruction == 'rescale':
            minimum = reconstructed.min(axis=(0, 1), keepdims=True)
            maximum = reconstructed.max(axis=(0, 1), keepdims=True)
            reconstructed = (reconstructed - minimum) / np.maximum(
                maximum - minimum, 1e-12)
        outputs[band.name] = np.clip(reconstructed, 0, 1).astype(np.float32)
    return outputs, masks


def centered_rms(value: np.ndarray) -> float:
    centered = value - value.mean(axis=(0, 1), keepdims=True)
    return float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))


def model_band_input(
    raw: np.ndarray,
    clean: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, float, Mapping[str, float]]:
    clean_mean = clean.mean(axis=(0, 1), keepdims=True)
    centered = raw - raw.mean(axis=(0, 1), keepdims=True)
    scale = 1.0
    if mode == 'equal-rms':
        scale = centered_rms(clean) / max(centered_rms(raw), 1e-12)
    unbounded = clean_mean + scale * centered
    statistics = {
        'clip_low_fraction': float((unbounded < 0).mean()),
        'clip_high_fraction': float((unbounded > 1).mean()),
    }
    return np.clip(unbounded, 0, 1).astype(np.float32), scale, statistics


def model_band_stop_input(
    raw: np.ndarray,
    clean: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, float]]:
    centered = raw - raw.mean(axis=(0, 1), keepdims=True)
    unbounded = clean - centered
    statistics = {
        'clip_low_fraction': float((unbounded < 0).mean()),
        'clip_high_fraction': float((unbounded > 1).mean()),
    }
    return np.clip(unbounded, 0, 1).astype(np.float32), statistics


def magnitude_visualization(raw: np.ndarray) -> np.ndarray:
    centered = raw - raw.mean(axis=(0, 1), keepdims=True)
    magnitude = np.abs(centered)
    lower, upper = np.percentile(magnitude, (1, 99))
    return np.clip((magnitude - lower) / max(upper - lower, 1e-12), 0, 1)


def save_png(path: Path, value: np.ndarray, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(quantized, mode='RGB').save(
        path, format='PNG', compress_level=compress_level)


def vector(value: np.ndarray) -> List[float]:
    return [float(item) for item in value]


def write_qa(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_rows(path, rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be between 0 and 9')
    if not 0 <= args.transition_ratio < 1:
        raise ValueError('--transition-ratio must satisfy 0 <= value < 1')
    if not 0 <= args.pad_fraction < 0.5:
        raise ValueError('--pad-fraction must satisfy 0 <= value < 0.5')
    rows = read_jsonl(args.manifest)
    validate_sample_order(rows)
    resize_limit = parse_resize(args.resize)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    energy_calibration = None
    if args.band_policy == 'dataset-energy':
        if args.method != 'soft-cpp':
            raise ValueError('dataset-energy policy requires --method soft-cpp')
        if args.bands:
            raise ValueError('--bands must be empty with dataset-energy policy')
        calibration_path = existing_file(
            args.calibration_manifest or args.manifest)
        calibration_rows = read_jsonl(calibration_path)
        validate_sample_order(calibration_rows)
        bands, energy_profile, energy_calibration = derive_dataset_energy_bands(
            calibration_rows,
            resize_limit,
            args.pad_fraction,
            parse_energy_quantiles(args.energy_quantiles),
            args.energy_bins,
            args.energy_color_space,
        )
        energy_calibration = {
            **energy_calibration,
            'manifest': str(calibration_path),
        }
        write_rows(out_dir / 'dataset_energy_profile.tsv', energy_profile)
        write_json(
            out_dir / 'dataset_energy_calibration.json', energy_calibration)
    else:
        bands = parse_bands(args.bands, args.method)
    output_rows = []
    qa_rows = []
    shape_counts: Dict[str, int] = {}
    maximum_reconstruction_error = 0.0

    for position, row in enumerate(rows, start=1):
        source = existing_file(row['image_path'])
        with Image.open(source) as opened:
            rgb = resize_keep_ratio(opened.convert('RGB'), resize_limit)
        image = np.asarray(rgb, dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        shape_key = f'{height}x{width}'
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        sample_name = f'{int(row["sample_index"]):05d}_{int(row["image_id"])}'
        variants: Dict[str, dict] = {}

        clean_path = out_dir / 'images' / 'clean' / f'{sample_name}.png'
        if args.copy_clean or resize_limit is not None:
            save_png(clean_path, image, args.png_compress_level)
        else:
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            if clean_path.exists() or clean_path.is_symlink():
                clean_path.unlink()
            clean_path.symlink_to(source)
        variants['clean'] = {
            'image_path': str(clean_path.absolute()),
            'representation': (
                'resized-clean-model-input'
                if resize_limit is not None else 'clean-model-input'),
            'height': height,
            'width': width,
        }

        if args.method == 'soft-cpp':
            raw_bands, masks = decompose_soft(
                image, bands, args.transition_ratio, args.pad_fraction)
            reconstruction = sum(raw_bands.values())
            reconstruction_error = np.abs(reconstruction - image)
            reconstruction_max = float(reconstruction_error.max())
            reconstruction_mae = float(reconstruction_error.mean())
            maximum_reconstruction_error = max(
                maximum_reconstruction_error, reconstruction_max)
            if reconstruction_max > args.reconstruction_tolerance:
                raise RuntimeError(
                    f'{source}: frequency reconstruction max error '
                    f'{reconstruction_max:.8g} exceeds '
                    f'{args.reconstruction_tolerance:.8g}')
        else:
            raw_bands, masks = decompose_legacy(
                image, bands, args.reconstruction)
            reconstruction_max = float('nan')
            reconstruction_mae = float('nan')

        source_energy = float(np.mean(np.square(image, dtype=np.float64)))
        source_centered_rms = centered_rms(image)
        for band in bands:
            raw = raw_bands[band.name]
            if args.method == 'soft-cpp':
                model_input, scale, clip_stats = model_band_input(
                    raw, image, args.model_input_mode)
            else:
                model_input = np.clip(raw, 0, 1).astype(np.float32)
                scale = 1.0
                clip_stats = {
                    'clip_low_fraction': 0.0,
                    'clip_high_fraction': 0.0,
                }
            image_path = out_dir / 'images' / band.name / f'{sample_name}.png'
            save_png(image_path, model_input, args.png_compress_level)
            raw_path = out_dir / 'arrays' / 'raw' / band.name / f'{sample_name}.npy'
            if args.save_raw:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, raw.astype(np.float32), allow_pickle=False)
            visual_path = (
                out_dir / 'visualizations' / 'magnitude' /
                band.name / f'{sample_name}.png')
            if args.save_visualizations:
                save_png(
                    visual_path, magnitude_visualization(raw),
                    args.png_compress_level)
            variant = {
                'image_path': str(image_path.absolute()),
                'representation': args.model_input_mode,
                'frequency_unit': (
                    'cycles_per_pixel' if args.method == 'soft-cpp'
                    else 'normalized_corner_radius'),
                'low': band.low,
                'high': band.high if band.high is not None else 'max',
                'source_mean': vector(image.mean(axis=(0, 1))),
                'output_mean': vector(model_input.mean(axis=(0, 1))),
                'output_std': vector(model_input.std(axis=(0, 1))),
                'raw_mean': vector(raw.mean(axis=(0, 1))),
                'raw_std': vector(raw.std(axis=(0, 1))),
                'raw_centered_rms': centered_rms(raw),
                'model_input_scale': scale,
                'mask_mean': float(masks[band.name].mean()),
                'mask_fraction': float(masks[band.name].mean()),
                **clip_stats,
            }
            if args.save_raw:
                variant['raw_path'] = str(raw_path.absolute())
            if args.save_visualizations:
                variant['visualization_path'] = str(visual_path.absolute())
            variants[band.name] = variant

            qa_rows.append({
                'sample_index': int(row['sample_index']),
                'image_id': int(row['image_id']),
                'file_name': row['file_name'],
                'height': height,
                'width': width,
                'band': band.name,
                'low_cpp': band.low if args.method == 'soft-cpp' else '',
                'high_cpp': (
                    band.high if args.method == 'soft-cpp' and band.high is not None
                    else 'max' if args.method == 'soft-cpp' else ''),
                'mask_mean': float(masks[band.name].mean()),
                'raw_energy': float(np.mean(np.square(raw, dtype=np.float64))),
                'raw_energy_over_clean': float(
                    np.mean(np.square(raw, dtype=np.float64)) /
                    max(source_energy, 1e-12)),
                'raw_centered_rms': centered_rms(raw),
                'clean_centered_rms': source_centered_rms,
                'model_input_scale': scale,
                'clip_low_fraction': clip_stats['clip_low_fraction'],
                'clip_high_fraction': clip_stats['clip_high_fraction'],
                'reconstruction_mae': reconstruction_mae,
                'reconstruction_max_abs': reconstruction_max,
            })

            if args.save_band_stop and args.method == 'soft-cpp':
                stopped, stopped_stats = model_band_stop_input(raw, image)
                stopped_name = f'remove_{band.name}'
                stopped_path = (
                    out_dir / 'images' / stopped_name / f'{sample_name}.png')
                save_png(stopped_path, stopped, args.png_compress_level)
                variants[stopped_name] = {
                    'image_path': str(stopped_path.absolute()),
                    'representation': 'mean-preserved-band-stop',
                    'removed_band': band.name,
                    'frequency_unit': 'cycles_per_pixel',
                    'low': band.low,
                    'high': band.high if band.high is not None else 'max',
                    'output_mean': vector(stopped.mean(axis=(0, 1))),
                    'output_std': vector(stopped.std(axis=(0, 1))),
                    **stopped_stats,
                }

        output = dict(row)
        output['frequency_input_shape'] = [height, width]
        output['variants'] = variants
        output_rows.append(output)
        print(f'[{position}/{len(rows)}] {row["file_name"]}', flush=True)

    write_jsonl(out_dir / 'frequency_manifest.jsonl', output_rows)
    write_qa(out_dir / 'frequency_qa.tsv', qa_rows)
    write_json(out_dir / 'filter_config.json', {
        'schema_version': 3,
        'source_manifest': str(existing_file(args.manifest)),
        'method': args.method,
        'band_policy': args.band_policy,
        'dataset_energy_calibration': energy_calibration,
        'frequency_unit': (
            'cycles_per_pixel' if args.method == 'soft-cpp'
            else 'fftshift radial distance normalized by corner Nyquist'),
        'bands': [
            {
                'name': band.name,
                'low': band.low,
                'high': band.high if band.high is not None else 'max',
                'low_wavelength_pixels': (
                    1.0 / band.low if band.low > 0 and args.method == 'soft-cpp'
                    else None),
                'high_wavelength_pixels': (
                    1.0 / band.high
                    if band.high and args.method == 'soft-cpp' else None),
            }
            for band in bands
        ],
        'transition_ratio': args.transition_ratio,
        'resize': args.resize,
        'resize_interpolation': 'Pillow bilinear',
        'pad_fraction': args.pad_fraction,
        'padding_mode': 'reflect',
        'model_input_mode': args.model_input_mode,
        'raw_arrays': (
            'signed unclipped float32 before model-input mapping'
            if args.method == 'soft-cpp'
            else 'legacy reconstructed and clipped float32'),
        'model_png_quantization': 'clip [0,1], round to uint8',
        'visualization': 'per-image 1st-99th percentile of abs(centered raw band)',
        'save_raw': args.save_raw,
        'save_band_stop': args.save_band_stop,
        'save_visualizations': args.save_visualizations,
        'reconstruction_tolerance': args.reconstruction_tolerance,
        'maximum_reconstruction_error': maximum_reconstruction_error,
        'shape_counts': shape_counts,
    })
    print(f'Frequency variants completed: {out_dir}')
    print(f'Manifest: {out_dir / "frequency_manifest.jsonl"}')
    print(f'QA: {out_dir / "frequency_qa.tsv"}')


if __name__ == '__main__':
    main()
