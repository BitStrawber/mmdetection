#!/usr/bin/env python3
"""Stage and zip EigenCAM panels, individual CAMs, originals, and metadata."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List
import zipfile


GROUPS = (
    {
        'dataset': 'ruod',
        'backbone': 'r50',
        'layer': 'res4',
        'models': (
            'imagenet_dino100e_backbone',
            'realuw_dino100e_backbone',
            'synthetic5_dino100e_backbone',
        ),
    },
    {
        'dataset': 'ruod',
        'backbone': 'vits',
        'layer': 'block12',
        'models': (
            'imagenet_vits_backbone',
            'realuw_vits_backbone',
            'synthetic5_vits_backbone',
        ),
    },
    {
        'dataset': 'uiis10k',
        'backbone': 'r50',
        'layer': 'res4',
        'models': (
            'imagenet_dino100e_backbone',
            'realuw_dino100e_backbone',
            'synthetic5_dino100e_backbone',
        ),
    },
    {
        'dataset': 'uiis10k',
        'backbone': 'vits',
        'layer': 'block12',
        'models': (
            'imagenet_vits_backbone',
            'realuw_vits_backbone',
            'synthetic5_vits_backbone',
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def copy_files(source: Path, destination: Path, expected: int) -> List[str]:
    files = sorted(source.glob('*.png'))
    if len(files) != expected:
        raise RuntimeError(
            f'Expected {expected} PNG files in {source}, got {len(files)}')
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for path in files:
        shutil.copy2(path, destination / path.name)
        names.append(path.name)
    return names


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f'{path}:{line_number}: expected a JSON object')
            rows.append(value)
    return rows


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError('--samples must be positive')
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    archive = Path(args.archive).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    if archive.exists() and not args.overwrite:
        raise FileExistsError(f'Archive already exists: {archive}')
    out_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    index_rows = []
    panel_count = 0
    cam_count = 0
    original_count = 0
    copied_originals = set()

    for group in GROUPS:
        dataset = str(group['dataset'])
        backbone = str(group['backbone'])
        layer = str(group['layer'])
        models = tuple(str(value) for value in group['models'])
        group_root = run_root / dataset / backbone
        rendered = group_root / 'rendered_eigen'
        manifest_path = group_root.parent / f'manifest_{args.samples}.jsonl'
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_manifest(manifest_path)
        if len(manifest) != args.samples:
            raise RuntimeError(
                f'{manifest_path}: expected {args.samples} rows, '
                f'got {len(manifest)}')

        panel_destination = (
            out_dir / 'panels' / dataset / f'{backbone}_{layer}')
        copy_files(
            rendered / 'panels' / layer,
            panel_destination,
            args.samples,
        )
        panel_count += args.samples

        for model in models:
            cam_destination = (
                out_dir / 'individual_cam' / dataset /
                f'{backbone}_{layer}' / model)
            copy_files(
                rendered / model / 'without_boxes' / layer,
                cam_destination,
                args.samples,
            )
            cam_count += args.samples

        statistics_path = rendered / 'activation_statistics.tsv'
        if statistics_path.is_file():
            destination = (
                out_dir / 'metadata' /
                f'{dataset}_{backbone}_{layer}_activation_statistics.tsv')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(statistics_path, destination)

        for position, row in enumerate(manifest):
            image_id = int(row['image_id'])
            stem = f'{position:05d}_{image_id}'
            source_image = Path(row['image_path']).expanduser().resolve()
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            destination = (
                out_dir / 'original' / dataset /
                f'{stem}{source_image.suffix.lower()}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            original_key = (dataset, stem)
            if original_key not in copied_originals:
                shutil.copy2(source_image, destination)
                original_count += 1
                copied_originals.add(original_key)

            index_rows.append({
                'dataset': dataset,
                'backbone': backbone,
                'layer': layer,
                'sample_index': position,
                'image_id': image_id,
                'file_name': row.get('file_name', source_image.name),
                'stem': stem,
                'panel': (
                    f'panels/{dataset}/{backbone}_{layer}/{stem}.png'),
                'original': (
                    f'original/{dataset}/{stem}{source_image.suffix.lower()}'),
            })

    manifest_destination = out_dir / 'metadata'
    manifest_destination.mkdir(parents=True, exist_ok=True)
    complete_env = run_root / 'COMPLETE.env'
    if complete_env.is_file():
        shutil.copy2(complete_env, manifest_destination / 'COMPLETE.env')

    index_path = out_dir / 'index.tsv'
    with index_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(index_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(index_rows)

    expected_panels = len(GROUPS) * args.samples
    expected_cams = sum(
        len(group['models']) * args.samples for group in GROUPS)
    expected_originals = len({str(group['dataset']) for group in GROUPS}) * (
        args.samples)
    if (
        panel_count != expected_panels or
        cam_count != expected_cams or
        original_count != expected_originals
    ):
        raise RuntimeError(
            f'Unexpected counts: panels={panel_count}/{expected_panels}, '
            f'cams={cam_count}/{expected_cams}, '
            f'originals={original_count}/{expected_originals}')

    with zipfile.ZipFile(
        archive,
        'w',
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as handle:
        for path in sorted(out_dir.rglob('*')):
            if path.is_file():
                handle.write(
                    path,
                    arcname=(
                        Path(out_dir.name) /
                        path.relative_to(out_dir)).as_posix(),
                )

    print(f'Panels:    {panel_count}')
    print(f'CAM images:{cam_count}')
    print(f'Originals: {original_count}')
    print(f'Stage:     {out_dir}')
    print(f'Archive:   {archive}')
    print(f'Archive size: {archive.stat().st_size / (1024 ** 3):.2f} GiB')


if __name__ == '__main__':
    main()
