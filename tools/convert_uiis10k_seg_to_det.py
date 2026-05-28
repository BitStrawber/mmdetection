#!/usr/bin/env python3
"""Convert UIIS10K COCO instance segmentation annotations to detection COCO.

The output keeps images/categories and writes bbox-only annotations. If an
annotation already has a valid COCO bbox, it is reused. Otherwise the bbox is
derived from polygon/RLE segmentation.
"""

import argparse
import json
import os
from copy import deepcopy

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover
    mask_utils = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True, help='Input COCO segmentation json.')
    parser.add_argument('--out', required=True, help='Output detection json.')
    parser.add_argument(
        '--drop-segmentation',
        action='store_true',
        default=True,
        help='Drop segmentation fields in output annotations.')
    parser.add_argument(
        '--min-area',
        type=float,
        default=1.0,
        help='Drop boxes whose bbox area is smaller than this value.')
    return parser.parse_args()


def bbox_from_polygon(segmentation):
    xs = []
    ys = []
    for poly in segmentation:
        if not poly or len(poly) < 6:
            continue
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
    if not xs or not ys:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def bbox_from_segmentation(segmentation, height, width):
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        return bbox_from_polygon(segmentation)
    if isinstance(segmentation, dict):
        if mask_utils is None:
            raise RuntimeError('pycocotools is required to decode RLE segmentation.')
        rle = segmentation
        if isinstance(rle.get('counts'), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        bbox = mask_utils.toBbox(rle).tolist()
        return [float(v) for v in bbox]
    return None


def valid_bbox(bbox, min_area):
    if not bbox or len(bbox) != 4:
        return False
    _, _, w, h = bbox
    return w > 0 and h > 0 and w * h >= min_area


def main():
    args = parse_args()
    with open(args.ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    image_size = {
        img['id']: (img.get('height', 0), img.get('width', 0))
        for img in coco.get('images', [])
    }

    converted = deepcopy(coco)
    converted_annotations = []
    skipped = 0

    for new_id, ann in enumerate(coco.get('annotations', [])):
        new_ann = deepcopy(ann)
        bbox = new_ann.get('bbox')
        if not valid_bbox(bbox, args.min_area):
            height, width = image_size.get(new_ann['image_id'], (0, 0))
            bbox = bbox_from_segmentation(new_ann.get('segmentation'), height, width)
        if not valid_bbox(bbox, args.min_area):
            skipped += 1
            continue

        new_ann['id'] = new_id
        new_ann['bbox'] = [round(float(v), 3) for v in bbox]
        new_ann['area'] = float(new_ann['bbox'][2] * new_ann['bbox'][3])
        new_ann['iscrowd'] = int(new_ann.get('iscrowd', 0))
        new_ann.pop('segmentation', None)
        converted_annotations.append(new_ann)

    converted['annotations'] = converted_annotations

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(converted, f)

    print(f'Input annotations: {len(coco.get("annotations", []))}')
    print(f'Output annotations: {len(converted_annotations)}')
    print(f'Skipped annotations: {skipped}')
    print(f'Output: {args.out}')


if __name__ == '__main__':
    main()
