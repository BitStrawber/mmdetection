#!/usr/bin/env python3
"""Render input-spectrum QA, band energy charts, and frequency-image panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import ensure_empty_or_create, read_json, read_jsonl, write_json
from .generate_frequency_bands import radial_frequency_cpp


COLORS = {
    'clean': '#222222', 'low': '#2f6fbb', 'mid': '#d29b20', 'high': '#d95f4c',
    'remove_low': '#79a7d3', 'remove_mid': '#e6bd58', 'remove_high': '#e99a8c',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--spectrum-variants', default='remove_low,remove_mid,remove_high')
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--panel-samples', type=int, default=6)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert('RGB'), dtype=np.float64) / 255.0


def power_histogram(value: np.ndarray, edges: np.ndarray) -> np.ndarray:
    centered = value - value.mean(axis=(0, 1), keepdims=True)
    window = np.hanning(value.shape[0])[:, None] * np.hanning(value.shape[1])[None, :]
    spectrum = np.fft.fftshift(
        np.fft.fft2(centered * window[:, :, None], axes=(0, 1)), axes=(0, 1))
    power = np.sum(np.square(np.abs(spectrum)), axis=2)
    radius = radial_frequency_cpp(value.shape[0], value.shape[1])
    histogram, _ = np.histogram(
        radius.reshape(-1), bins=edges, weights=power.reshape(-1))
    return histogram.astype(np.float64)


def mean_spectra(
    rows: Sequence[Mapping[str, object]], variants: Sequence[str], bins: int,
) -> List[Mapping[str, object]]:
    maximum = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    edges = np.linspace(0.0, maximum, bins + 1)
    accumulated: Dict[str, List[np.ndarray]] = {'clean': []}
    accumulated.update({f'difference_{name}': [] for name in variants})
    for index, row in enumerate(rows, start=1):
        payload = row['variants']
        clean = load_rgb(payload['clean']['image_path'])
        clean_power = power_histogram(clean, edges)
        denominator = max(float(clean_power.sum()), 1e-12)
        accumulated['clean'].append(clean_power / denominator)
        for variant in variants:
            changed = load_rgb(payload[variant]['image_path'])
            if changed.shape != clean.shape:
                raise ValueError(f'{variant}: input shape differs for sample {index}')
            difference_power = power_histogram(clean - changed, edges)
            accumulated[f'difference_{variant}'].append(
                difference_power / denominator)
        print(f'[spectrum {index}/{len(rows)}] {row["file_name"]}', flush=True)
    output = []
    for signal, values in accumulated.items():
        matrix = np.stack(values)
        mean = matrix.mean(axis=0)
        p05 = np.percentile(matrix, 5, axis=0)
        p95 = np.percentile(matrix, 95, axis=0)
        for bin_index in range(bins):
            output.append({
                'signal': signal,
                'bin_index': bin_index,
                'low_cpp': float(edges[bin_index]),
                'high_cpp': float(edges[bin_index + 1]),
                'center_cpp': float((edges[bin_index] + edges[bin_index + 1]) / 2),
                'mean_energy_fraction_of_clean': float(mean[bin_index]),
                'p05': float(p05[bin_index]),
                'p95': float(p95[bin_index]),
            })
    return output


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def render_spectrum(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row['signal']), []).append(row)
    figure, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for signal, values in grouped.items():
        x = np.asarray([float(row['center_cpp']) for row in values])
        y = np.maximum(np.asarray([
            float(row['mean_energy_fraction_of_clean']) for row in values]), 1e-12)
        label = signal.replace('difference_', 'clean - ')
        color = COLORS.get(signal.replace('difference_', ''), '#666666')
        axis.plot(x, y, label=label, color=color, linewidth=1.8)
    axis.set_yscale('log')
    axis.set_xlabel('Radial frequency (cycles/pixel)')
    axis.set_ylabel('Energy fraction relative to clean spectrum')
    axis.set_title('Input and degradation-difference spectra')
    axis.grid(True, color='#dddddd', linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    for suffix in ('png', 'pdf'):
        figure.savefig(out_dir / f'input_difference_spectra.{suffix}', dpi=240)
    plt.close(figure)


def render_energy(qa_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with qa_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    bands = []
    values = []
    for band in ('low', 'mid', 'high'):
        selected = np.asarray([
            (float(row['raw_centered_rms']) /
             max(float(row['clean_centered_rms']), 1e-12)) ** 2
            for row in rows if row['band'] == band])
        if selected.size:
            bands.append(band)
            values.append(selected)
    figure, axis = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    positions = np.arange(len(bands))
    means = [float(value.mean()) for value in values]
    errors = np.asarray([
        [mean - float(np.percentile(value, 5)) for mean, value in zip(means, values)],
        [float(np.percentile(value, 95)) - mean for mean, value in zip(means, values)],
    ])
    axis.bar(
        positions, means, yerr=errors, capsize=4,
        color=[COLORS[band] for band in bands], edgecolor='#333333', linewidth=0.8)
    axis.set_xticks(positions, bands)
    axis.set_ylabel('Centered band energy / centered clean energy')
    axis.set_title('Input energy distribution by frequency band')
    axis.grid(axis='y', color='#dddddd', linewidth=0.7)
    for suffix in ('png', 'pdf'):
        figure.savefig(out_dir / f'band_energy_distribution.{suffix}', dpi=240)
    plt.close(figure)


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.convert('RGB').copy()
    result.thumbnail(size)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2))
    return canvas


def render_panels(rows: Sequence[Mapping[str, object]], out_dir: Path, count: int) -> None:
    labels = ('clean', 'low', 'mid', 'high')
    tile_size = (320, 200)
    label_height = 28
    panel_root = out_dir / 'frequency_band_panels'
    panel_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows[:count]):
        tiles = []
        for label in labels:
            with Image.open(row['variants'][label]['image_path']) as opened:
                tile = fit(opened, tile_size)
            output = Image.new('RGB', (tile.width, tile.height + label_height), 'white')
            output.paste(tile, (0, label_height))
            ImageDraw.Draw(output).text((10, 7), label, fill='#222222')
            tiles.append(output)
        panel = Image.new('RGB', (sum(tile.width for tile in tiles), tiles[0].height), 'white')
        left = 0
        for tile in tiles:
            panel.paste(tile, (left, 0))
            left += tile.width
        panel.save(panel_root / f'{index:03d}_{int(row["image_id"])}.png')


def main() -> None:
    args = parse_args()
    root = Path(args.frequency_root).expanduser().resolve()
    rows = read_jsonl(root / 'frequency_manifest.jsonl')
    config = read_json(root / 'filter_config.json')
    variants = [item.strip() for item in args.spectrum_variants.split(',') if item.strip()]
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    spectra = mean_spectra(rows, variants, args.bins)
    write_tsv(out_dir / 'input_difference_spectra.tsv', spectra)
    render_spectrum(spectra, out_dir)
    render_energy(root / 'frequency_qa.tsv', out_dir)
    render_panels(rows, out_dir, min(args.panel_samples, len(rows)))
    write_json(out_dir / 'input_visualization_metadata.json', {
        'frequency_root': str(root),
        'band_policy': config.get('band_policy'),
        'bands': config.get('bands'),
        'samples': len(rows),
        'spectrum_variants': variants,
        'spectrum_normalization': 'each difference PSD divided by clean total PSD',
    })
    print(f'Frequency input visualizations: {out_dir}')


if __name__ == '__main__':
    main()
