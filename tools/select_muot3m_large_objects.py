#!/usr/bin/env python3
"""Select large-object frames from MUOT_3M without extracting frames.

MUOT_3M is stored as video sequences:

    split/Video_xxx/Video_xxx.mp4
    split/Video_xxx/groundtruth.txt
    split/Video_xxx/captions.txt

This script reads bounding boxes from ``groundtruth.txt``, obtains video
metadata, and writes a COCO-style JSON plus a JSONL manifest for frames whose
box area ratio satisfies ``bbox_area / frame_area >= threshold``. Actual frame
extraction can be performed later from the manifest.
"""

import argparse
import json
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default='/media/HDD1/XCX/exp_2/MUOT_3M',
        help='MUOT_3M root containing train/ and test/.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test'],
        help='Splits to process.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
        help='Minimum bbox area / image area ratio.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output annotation directory. Defaults to ROOT/annotations.')
    parser.add_argument(
        '--manifest-dir',
        default=None,
        help='Output manifest directory. Defaults to ROOT/manifests.')
    parser.add_argument(
        '--category-name',
        default='object',
        help='Single COCO category name.')
    parser.add_argument(
        '--keep-invalid',
        action='store_true',
        help='Keep invalid boxes in stats only; invalid boxes are never selected.')
    return parser.parse_args()


def parse_bbox_line(line):
    parts = [part for part in re.split(r'[\s,]+', line.strip()) if part]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = [float(value) for value in parts[:4]]
    except ValueError:
        return None
    return [x, y, w, h]


def read_bboxes(path):
    bboxes = []
    invalid_lines = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            if not line.strip():
                continue
            bbox = parse_bbox_line(line)
            if bbox is None:
                invalid_lines += 1
                continue
            bboxes.append(bbox)
    return bboxes, invalid_lines


def read_caption(path):
    if not path.is_file():
        return ''
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read().strip()


def find_video(seq_dir):
    videos = [
        path for path in seq_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        return None
    exact = seq_dir / '{}.mp4'.format(seq_dir.name)
    if exact.is_file():
        return exact
    return sorted(videos, key=lambda path: (-path.stat().st_size, path.name))[0]


def video_info(video_path):
    try:
        import cv2
    except ImportError:
        raise RuntimeError('opencv-python is required to read video metadata.')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if width <= 0 or height <= 0:
        return None
    return {
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'fps': fps,
    }


def clamp_bbox(bbox, width, height):
    x, y, w, h = bbox
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + max(0.0, w))
    y2 = min(float(height), y + max(0.0, h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_ratio(bbox, width, height):
    if width <= 0 or height <= 0:
        return 0.0
    return (max(0.0, bbox[2]) * max(0.0, bbox[3])) / float(width * height)


def sequence_dirs(split_dir):
    return [
        path for path in sorted(split_dir.iterdir())
        if path.is_dir() and path.name.startswith('Video_')
    ]


def process_split(root, split, out_dir, manifest_dir, threshold, category_name):
    split_dir = root / split
    if not split_dir.is_dir():
        raise RuntimeError('Missing split directory: {}'.format(split_dir))

    coco = {
        'info': {
            'description': 'MUOT_3M large-object frame selection',
            'source_root': str(root),
            'split': split,
            'threshold': threshold,
            'criterion': 'bbox_area / frame_area >= threshold',
            'note': 'Frame files are not extracted yet; use manifest JSONL.',
        },
        'licenses': [],
        'categories': [{'id': 1, 'name': category_name}],
        'images': [],
        'annotations': [],
    }

    manifest_path = manifest_dir / 'muot3m_{}_bbox{:02d}pct_frames.jsonl'.format(
        split, int(round(threshold * 100)))
    out_path = out_dir / 'instances_{}_bbox{:02d}pct.json'.format(
        split, int(round(threshold * 100)))

    stats = {
        'split': split,
        'sequences': 0,
        'missing_gt': 0,
        'missing_video': 0,
        'bad_video': 0,
        'invalid_gt_lines': 0,
        'invalid_boxes': 0,
        'total_bboxes': 0,
        'selected_frames': 0,
        'frame_count_mismatches': 0,
    }

    image_id = 1
    ann_id = 1
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        for seq in tqdm(sequence_dirs(split_dir), desc='MUOT_3M {}'.format(split), unit='seq'):
            stats['sequences'] += 1
            gt_path = seq / 'groundtruth.txt'
            if not gt_path.is_file():
                stats['missing_gt'] += 1
                continue

            video_path = find_video(seq)
            if video_path is None:
                stats['missing_video'] += 1
                continue

            info = video_info(video_path)
            if info is None:
                stats['bad_video'] += 1
                continue

            bboxes, invalid_lines = read_bboxes(gt_path)
            stats['invalid_gt_lines'] += invalid_lines
            stats['total_bboxes'] += len(bboxes)
            if info['frame_count'] and info['frame_count'] != len(bboxes):
                stats['frame_count_mismatches'] += 1

            caption = read_caption(seq / 'captions.txt')
            pair_count = len(bboxes)
            if info['frame_count']:
                pair_count = min(pair_count, info['frame_count'])

            for frame_index, bbox in enumerate(bboxes[:pair_count], start=1):
                clipped = clamp_bbox(bbox, info['width'], info['height'])
                ratio = bbox_ratio(clipped, info['width'], info['height'])
                if clipped[2] <= 0 or clipped[3] <= 0:
                    stats['invalid_boxes'] += 1
                    continue
                if ratio < threshold:
                    continue

                file_name = '{}/{}/{:08d}.jpg'.format(split, seq.name, frame_index)
                coco['images'].append({
                    'id': image_id,
                    'file_name': file_name,
                    'width': info['width'],
                    'height': info['height'],
                    'sequence': seq.name,
                    'frame_index': frame_index,
                    'video_path': str(video_path),
                    'caption': caption,
                })
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [round(float(value), 3) for value in clipped],
                    'area': round(float(clipped[2] * clipped[3]), 3),
                    'iscrowd': 0,
                    'bbox_ratio': round(float(ratio), 6),
                })

                manifest_file.write(json.dumps({
                    'split': split,
                    'sequence': seq.name,
                    'video_path': str(video_path),
                    'frame_index': frame_index,
                    'output_file': file_name,
                    'width': info['width'],
                    'height': info['height'],
                    'bbox': [round(float(value), 3) for value in clipped],
                    'bbox_ratio': round(float(ratio), 6),
                    'caption': caption,
                }, ensure_ascii=False) + '\n')

                image_id += 1
                ann_id += 1
                stats['selected_frames'] += 1

    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(coco, file)

    print(split)
    print('  sequences:', stats['sequences'])
    print('  total_bboxes:', stats['total_bboxes'])
    print('  selected_frames:', stats['selected_frames'])
    print('  missing_gt:', stats['missing_gt'])
    print('  missing_video:', stats['missing_video'])
    print('  bad_video:', stats['bad_video'])
    print('  invalid_gt_lines:', stats['invalid_gt_lines'])
    print('  invalid_boxes:', stats['invalid_boxes'])
    print('  frame_count_mismatches:', stats['frame_count_mismatches'])
    print('  coco:', out_path)
    print('  manifest:', manifest_path)
    return stats


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'annotations'
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else root / 'manifests'

    all_stats = []
    for split in args.splits:
        all_stats.append(process_split(
            root=root,
            split=split,
            out_dir=out_dir,
            manifest_dir=manifest_dir,
            threshold=args.threshold,
            category_name=args.category_name,
        ))

    print('\nSummary')
    print('=' * 80)
    total_bboxes = sum(item['total_bboxes'] for item in all_stats)
    selected_frames = sum(item['selected_frames'] for item in all_stats)
    missing_gt = sum(item['missing_gt'] for item in all_stats)
    print('splits:', ', '.join(item['split'] for item in all_stats))
    print('total_bboxes:', total_bboxes)
    print('selected_frames:', selected_frames)
    print('missing_gt:', missing_gt)
    if total_bboxes:
        print('keep_ratio: {:.2f}%'.format(selected_frames / total_bboxes * 100.0))


if __name__ == '__main__':
    main()
