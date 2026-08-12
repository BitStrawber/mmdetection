#!/usr/bin/env python3
"""Validate fixed-GT CAM paths, detector specs and RUOD category compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-file', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--models-config', required=True)
    return parser.parse_args()


def require_file(path: Union[str, Path]) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return value


def main() -> None:
    args = parse_args()
    annotation_path = require_file(args.annotation_file)
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    config_path = require_file(args.models_config)
    with annotation_path.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)
    with config_path.open('r', encoding='utf-8') as handle:
        models_config = json.load(handle)
    categories = coco.get('categories', [])
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    missing_images = []
    for image in images:
        path = image_root / image['file_name']
        if not path.is_file() or path.stat().st_size <= 0:
            missing_images.append(str(path))
    models = models_config.get('models', [])
    if not models:
        raise ValueError('models-config contains no models')
    print('============================================================')
    print('Fixed-GT CAM input validation')
    print('============================================================')
    print(f'RUOD images:       {len(images)}')
    print(f'RUOD annotations:  {len(annotations)}')
    print(f'RUOD categories:   {len(categories)}')
    print(f'Missing images:    {len(missing_images)}')
    print('Category order:')
    for index, category in enumerate(categories):
        print(f'  label={index:2d} coco_id={int(category["id"]):3d} '
              f'name={category["name"]}')
    if missing_images:
        raise FileNotFoundError(f'First missing image: {missing_images[0]}')
    print('\nDetector models:')
    seen = set()
    for spec in models:
        model_id = str(spec['id'])
        if model_id in seen:
            raise ValueError(f'Duplicate model ID: {model_id}')
        seen.add(model_id)
        if str(spec.get('kind', 'detector')) != 'detector':
            raise ValueError(f'{model_id}: kind must be detector')
        detector_config = require_file(spec['config'])
        checkpoint = require_file(spec['checkpoint'])
        layers = spec.get('layers', {})
        if not layers:
            raise ValueError(f'{model_id}: no target layers configured')
        print(f'  [{model_id}]')
        print(f'    config:     {detector_config}')
        print(f'    checkpoint: {checkpoint} ({checkpoint.stat().st_size} bytes)')
        print(f'    layers:     {layers}')
    print('\nInput path validation: OK')
    print('Runtime model/head compatibility is checked during extraction.')


if __name__ == '__main__':
    main()
