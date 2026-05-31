#!/usr/bin/env python3
"""Inspect annotation files/directories and report conversion evidence.

This utility is intentionally conservative: it verifies whether a JSON file
looks like COCO detection/instance annotations and, for non-COCO paths, checks
common dataset layouts used in this workspace. It also samples TXT annotation
files to distinguish YOLO-style labels from tracking groundtruth boxes.

The main output is a structured report that can be pasted into an LLM prompt to
generate a dataset-specific converter.
"""

import argparse
import json
import os
import re
from pathlib import Path
from pprint import pformat


COCO_TOP_KEYS = {'images', 'annotations', 'categories'}
COCO_IMAGE_KEYS = {'id', 'file_name', 'width', 'height'}
COCO_ANN_KEYS = {'id', 'image_id', 'category_id', 'bbox'}
COCO_CAT_KEYS = {'id', 'name'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'paths',
        nargs='+',
        help='Annotation JSON files or dataset directories to inspect.')
    parser.add_argument(
        '--max-json-samples',
        type=int,
        default=3,
        help='Number of JSON files to sample from a non-COCO directory.')
    parser.add_argument(
        '--max-txt-samples',
        type=int,
        default=5,
        help='Number of TXT files to sample from a non-COCO directory.')
    parser.add_argument(
        '--max-lines',
        type=int,
        default=5,
        help='Number of lines to show for each sampled TXT file.')
    return parser.parse_args()


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        return None, str(exc)


def inspect_coco_json(path):
    data, error = load_json(path)
    if data is None:
        return {
            'is_json': False,
            'is_coco': False,
            'format_guess': 'invalid_json',
            'reason': f'cannot parse json: {error}',
            'conversion_info_needed': ['valid annotation schema or parser rule'],
        }
    if not isinstance(data, dict):
        return {
            'is_json': True,
            'is_coco': False,
            'format_guess': 'json_non_object',
            'reason': 'top-level JSON is not an object',
            'conversion_info_needed': ['meaning of top-level JSON values'],
        }

    missing_top = sorted(COCO_TOP_KEYS - set(data.keys()))
    if missing_top:
        return {
            'is_json': True,
            'is_coco': False,
            'format_guess': 'non_coco_json',
            'reason': f'missing top-level keys: {missing_top}',
            'top_keys': sorted(data.keys()),
            'sample': sample_json_value(data),
            'conversion_info_needed': [
                'image records and their file names',
                'annotation records and bbox/mask fields',
                'category names or category id mapping',
            ],
        }

    images = data.get('images') or []
    annotations = data.get('annotations') or []
    categories = data.get('categories') or []
    image_missing = sorted(COCO_IMAGE_KEYS - set(images[0].keys())) if images else []
    ann_missing = sorted(COCO_ANN_KEYS - set(annotations[0].keys())) if annotations else []
    cat_missing = sorted(COCO_CAT_KEYS - set(categories[0].keys())) if categories else []

    is_coco = not image_missing and not ann_missing and not cat_missing
    has_segmentation = bool(annotations and 'segmentation' in annotations[0])
    has_bbox = bool(annotations and 'bbox' in annotations[0])

    result = {
        'is_json': True,
        'is_coco': is_coco,
        'format_guess': 'coco_json' if is_coco else 'coco_like_json',
        'num_images': len(images),
        'num_annotations': len(annotations),
        'num_categories': len(categories),
        'has_bbox': has_bbox,
        'has_segmentation': has_segmentation,
        'image_missing': image_missing,
        'annotation_missing': ann_missing,
        'category_missing': cat_missing,
        'image_sample': images[0] if images else None,
        'annotation_sample': annotations[0] if annotations else None,
        'category_sample': categories[:5],
    }
    if is_coco and has_segmentation:
        result['coco_type'] = 'COCO instance segmentation; usable for bbox filtering if bbox is valid'
        result['suggestion'] = (
            'Use directly with tools/filter_coco_large_objects.py, or convert to '
            'detection-only with tools/convert_uiis10k_seg_to_det.py if needed.')
    elif is_coco:
        result['coco_type'] = 'COCO detection'
        result['suggestion'] = 'Use directly with tools/filter_coco_large_objects.py.'
    else:
        result['suggestion'] = 'JSON is close to COCO but missing required fields.'
        result['conversion_info_needed'] = [
            'how to derive missing image width/height/file_name fields',
            'how to derive bbox as COCO [x,y,width,height]',
            'category id to category name mapping',
        ]
    return result


def sample_json_value(data):
    if isinstance(data, dict):
        sample = {}
        for key in list(data.keys())[:5]:
            value = data[key]
            if isinstance(value, list):
                sample[key] = value[:2]
            elif isinstance(value, dict):
                sample[key] = {k: value[k] for k in list(value.keys())[:5]}
            else:
                sample[key] = value
        return sample
    if isinstance(data, list):
        return data[:2]
    return data


def list_child_names(path):
    try:
        return {p.name for p in Path(path).iterdir()}
    except OSError:
        return set()


def find_jsons(path, max_samples):
    return sorted(Path(path).rglob('*.json'))[:max_samples]


def parse_numeric_line(line):
    line = line.strip()
    if not line:
        return None
    parts = re.split(r'[\s,]+', line)
    try:
        return [float(x) for x in parts if x != '']
    except ValueError:
        return None


def classify_txt_file(path, max_lines=20, preview_lines=5):
    numeric_rows = []
    raw_lines = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(raw_lines) < preview_lines:
                    raw_lines.append(line.rstrip('\n'))
                row = parse_numeric_line(line)
                if row is not None:
                    numeric_rows.append(row)
                if len(numeric_rows) >= max_lines:
                    break
    except OSError as exc:
        return {
            'path': str(path),
            'format_guess': 'unreadable_txt',
            'reason': str(exc),
            'conversion_info_needed': ['readable annotation file'],
        }

    if not numeric_rows:
        return {
            'path': str(path),
            'format_guess': 'unknown_txt',
            'reason': 'no numeric rows sampled',
            'raw_preview': raw_lines,
            'conversion_info_needed': [
                'line grammar',
                'field meanings',
                'image-to-annotation mapping',
                'category mapping',
            ],
        }

    lengths = sorted({len(row) for row in numeric_rows})
    first = numeric_rows[0]

    if all(len(row) == 5 for row in numeric_rows):
        class_like = all(float(row[0]).is_integer() and row[0] >= 0 for row in numeric_rows)
        normalized = all(0.0 <= v <= 1.0 for row in numeric_rows for v in row[1:5])
        if class_like and normalized:
            return {
                'path': str(path),
                'format_guess': 'yolo_txt',
                'lengths': lengths,
                'sample': first,
                'raw_preview': raw_lines,
                'convertible': True,
                'field_guess': ['class_id', 'x_center_norm', 'y_center_norm', 'width_norm', 'height_norm'],
                'conversion_info_needed': [
                    'image file paired with each txt file',
                    'image width and height',
                    'class_id to category name mapping',
                    'output split policy',
                ],
            }

    if all(len(row) >= 4 for row in numeric_rows):
        # Tracking groundtruth is commonly x,y,w,h per frame, sometimes with
        # extra flags. Values are usually pixel coordinates, not normalized.
        coordinate_like = all(row[2] >= 0 and row[3] >= 0 for row in numeric_rows)
        if coordinate_like:
            return {
                'path': str(path),
                'format_guess': 'tracking_bbox_txt',
                'lengths': lengths,
                'sample': first,
                'raw_preview': raw_lines,
                'convertible': True,
                'field_guess': ['x', 'y', 'width', 'height', 'optional_extra_fields'],
                'conversion_info_needed': [
                    'frame image directory and frame ordering',
                    'whether coordinates are 0-based or 1-based',
                    'category name/id to assign to tracked object',
                    'how to handle absent/lost/invalid frames if extra fields exist',
                ],
            }

    return {
        'path': str(path),
        'format_guess': 'unknown_txt',
        'lengths': lengths,
        'sample': first,
        'raw_preview': raw_lines,
        'convertible': None,
        'conversion_info_needed': [
            'line grammar',
            'field meanings',
            'image-to-annotation mapping',
            'category mapping',
        ],
    }


def find_txts(path, max_samples):
    names = []
    priority_patterns = ('groundtruth', 'gt', 'label', 'anno', 'annotation')
    all_txts = sorted(Path(path).rglob('*.txt'))
    for txt in all_txts:
        stem = txt.stem.lower()
        if any(pattern in stem for pattern in priority_patterns):
            names.append(txt)
    for txt in all_txts:
        if txt not in names:
            names.append(txt)
    return names[:max_samples]


def preview_tree(path, max_entries=50):
    root = Path(path)
    entries = []
    for idx, child in enumerate(sorted(root.rglob('*'))):
        if idx >= max_entries:
            entries.append('...')
            break
        try:
            rel = child.relative_to(root)
        except ValueError:
            rel = child
        suffix = '/' if child.is_dir() else ''
        entries.append(str(rel) + suffix)
    return entries


def inspect_directory(path, max_json_samples, max_txt_samples, max_lines):
    root = Path(path)
    names = list_child_names(root)
    common = {
        'top_level_entries': sorted(names),
        'tree_preview': preview_tree(path),
    }

    if {'train', 'test'}.issubset(names):
        train_names = list_child_names(root / 'train')
        test_names = list_child_names(root / 'test')
        if {'images', 'jsons'}.issubset(train_names) or {'images', 'jsons'}.issubset(test_names):
            return {
                **common,
                'layout': 'CoralSCOP-style per-image JSON directory',
                'is_coco': False,
                'convertible': True,
                'format_guess': 'per_image_json_segmentation',
                'conversion_info_needed': [
                    'per-image JSON schema',
                    'mask/polygon fields and category field',
                    'image dimensions from image files or JSON',
                    'category mapping to desired COCO categories',
                ],
                'existing_converter_hint': f'python tools/convert_coralscop.py --data-dir {path}',
            }

    if {'annotations', 'images'}.issubset(names):
        ann_jsons = sorted((root / 'annotations').glob('*.json'))
        coco_jsons = []
        non_coco_jsons = []
        for json_path in ann_jsons[:max_json_samples]:
            info = inspect_coco_json(json_path)
            if info.get('is_coco'):
                coco_jsons.append(str(json_path))
            else:
                non_coco_jsons.append((str(json_path), info.get('reason')))
        if coco_jsons:
            return {
                **common,
                'layout': 'directory with COCO-like annotations/images',
                'is_coco': True,
                'convertible': False,
                'format_guess': 'coco_json',
                'sample_coco_jsons': coco_jsons,
                'conversion_info_needed': [],
            }
        if {'train', 'test'}.intersection(names):
            return {
                **common,
                'layout': 'tracking-style dataset with annotations/images/train/test',
                'is_coco': False,
                'convertible': True,
                'format_guess': 'tracking_dataset',
                'sample_non_coco_jsons': non_coco_jsons,
                'conversion_info_needed': [
                    'annotation file schema in annotations/',
                    'frame image paths in train/test/images',
                    'frame ordering',
                    'category mapping',
                ],
                'existing_converter_hint': f'Try tools/convert_uvot400_v2.py after checking its BASE path for {path}.',
            }

    sampled_jsons = find_jsons(root, max_json_samples)
    if sampled_jsons:
        samples = []
        any_coco = False
        for json_path in sampled_jsons:
            info = inspect_coco_json(json_path)
            any_coco = any_coco or bool(info.get('is_coco'))
            samples.append({
                'path': str(json_path),
                'is_coco': info.get('is_coco'),
                'reason': info.get('reason'),
                'format_guess': info.get('format_guess'),
                'num_images': info.get('num_images'),
                'num_annotations': info.get('num_annotations'),
                'sample': info.get('sample') or info.get('annotation_sample'),
            })
        return {
            **common,
            'layout': 'directory with JSON files',
            'is_coco': any_coco,
            'convertible': None,
            'format_guess': 'mixed_or_unknown_json_directory',
            'samples': samples,
            'conversion_info_needed': [
                'which JSON files are annotation files',
                'schema of non-COCO JSON files',
                'image root directory',
                'category mapping',
            ],
        }

    sampled_txts = find_txts(root, max_txt_samples)
    if sampled_txts:
        txt_samples = [
            classify_txt_file(txt, preview_lines=max_lines) for txt in sampled_txts
        ]
        formats = sorted({sample['format_guess'] for sample in txt_samples})
        convertible = None
        needed = [
            'image root directory',
            'image-to-txt matching rule',
            'category mapping',
        ]
        if 'tracking_bbox_txt' in formats:
            convertible = True
            needed = [
                'frame image directory and frame ordering',
                'coordinate convention: x,y,w,h and 0-based/1-based',
                'category assigned to tracked objects',
                'handling rules for absent/lost frames',
            ]
        elif 'yolo_txt' in formats:
            convertible = True
            needed = [
                'class_id to category name mapping',
                'image dimensions',
                'image-to-label matching rule',
                'split policy',
            ]
        return {
            **common,
            'layout': 'directory with TXT annotations',
            'is_coco': False,
            'convertible': convertible,
            'format_guess': '+'.join(formats),
            'txt_formats': formats,
            'samples': txt_samples,
            'conversion_info_needed': needed,
        }

    return {
        **common,
        'layout': 'unknown',
        'is_coco': False,
        'convertible': None,
        'format_guess': 'unknown_directory',
        'conversion_info_needed': [
            'annotation file locations',
            'annotation schema',
            'image root directory',
            'category mapping',
        ],
    }


def print_result(path, result):
    print('=' * 80)
    print(path)
    for key, value in result.items():
        if isinstance(value, (dict, list, tuple)):
            print(f'{key}: {pformat(value, width=100)}')
        else:
            print(f'{key}: {value}')


def main():
    args = parse_args()
    for input_path in args.paths:
        path = Path(input_path)
        if path.is_file():
            if path.suffix.lower() == '.json':
                result = inspect_coco_json(path)
            elif path.suffix.lower() == '.txt':
                result = classify_txt_file(path)
                result['is_coco'] = False
            else:
                result = {
                    'is_coco': False,
                    'convertible': None,
                    'suggestion': f'Unsupported file extension: {path.suffix}',
                }
        elif path.is_dir():
            result = inspect_directory(
                path, args.max_json_samples, args.max_txt_samples, args.max_lines)
        else:
            result = {
                'is_coco': False,
                'convertible': None,
                'suggestion': 'Path does not exist.',
            }
        print_result(input_path, result)


if __name__ == '__main__':
    main()
