#!/usr/bin/env python3
"""Extract MUOT_3M frames referenced by COCO large-object annotations.

``select_muot3m_large_objects.py`` writes COCO annotations without extracting
frames. Each image entry stores:

    video_path: absolute source video path
    frame_index: 1-based frame number
    file_name: output image path relative to the MUOT_3M root

This script materializes those frame images so the COCO files can be visualized
or used by tools that expect image files on disk.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_ANNS = [
    'annotations/instances_train_bbox20pct.json',
    'annotations/instances_test_bbox20pct.json',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default='/media/HDD1/XCX/exp_2/MUOT_3M',
        help='MUOT_3M root. COCO file_name paths are resolved under this root.')
    parser.add_argument(
        '--ann',
        nargs='+',
        default=DEFAULT_ANNS,
        help='COCO json files. Relative paths are resolved under --root.')
    parser.add_argument(
        '--frame-base',
        type=int,
        choices=[0, 1],
        default=1,
        help='Whether image frame_index values are 0-based or 1-based.')
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality for extracted frames.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing extracted images.')
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Debug limit for number of images to extract. 0 means no limit.')
    parser.add_argument(
        '--summary',
        default=None,
        help='Optional output summary JSON. Defaults to ROOT/annotations/extract_muot3m_bbox_frames_summary.json.')
    return parser.parse_args()


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def load_tasks(root, ann_paths, limit=0):
    tasks_by_video = defaultdict(list)
    image_count = 0
    missing_fields = 0

    for ann_path in ann_paths:
        with open(ann_path, 'r', encoding='utf-8') as file:
            coco = json.load(file)

        for image in coco.get('images', []):
            video_path = image.get('video_path')
            frame_index = image.get('frame_index')
            file_name = image.get('file_name')
            if not video_path or frame_index is None or not file_name:
                missing_fields += 1
                continue

            out_path = root / str(file_name)
            tasks_by_video[str(video_path)].append({
                'image_id': image.get('id'),
                'frame_index': int(frame_index),
                'file_name': file_name,
                'out_path': out_path,
            })
            image_count += 1
            if limit and image_count >= limit:
                return tasks_by_video, image_count, missing_fields

    return tasks_by_video, image_count, missing_fields


def extract_frame(cap, frame_index, frame_base):
    cv2_index = frame_index - frame_base
    if cv2_index < 0:
        cv2_index = 0
    cap.set(1, cv2_index)  # cv2.CAP_PROP_POS_FRAMES
    ok, frame = cap.read()
    return ok, frame


def main():
    args = parse_args()
    root = Path(args.root)
    ann_paths = [resolve_path(root, path) for path in args.ann]
    for ann_path in ann_paths:
        if not ann_path.is_file():
            raise FileNotFoundError('Missing annotation file: {}'.format(ann_path))

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('opencv-python is required to extract MUOT_3M frames.') from exc

    tasks_by_video, requested_images, missing_fields = load_tasks(
        root=root,
        ann_paths=ann_paths,
        limit=args.limit,
    )

    print('root:', root)
    print('annotation files:', len(ann_paths))
    for ann_path in ann_paths:
        print('  ', ann_path)
    print('videos:', len(tasks_by_video))
    print('requested images:', requested_images)
    print('missing image fields:', missing_fields)
    print('frame_base:', args.frame_base)

    stats = {
        'root': str(root),
        'annotations': [str(path) for path in ann_paths],
        'videos': len(tasks_by_video),
        'requested_images': requested_images,
        'missing_image_fields': missing_fields,
        'written': 0,
        'skipped_exists': 0,
        'missing_video': 0,
        'bad_video': 0,
        'failed_frames': 0,
        'frame_base': args.frame_base,
        'quality': args.quality,
        'overwrite': args.overwrite,
        'limit': args.limit,
    }

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]
    for video_path_text, tasks in tqdm(
            sorted(tasks_by_video.items()),
            desc='extract MUOT_3M',
            unit='video'):
        video_path = Path(video_path_text)
        if not video_path.is_file():
            stats['missing_video'] += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats['bad_video'] += 1
            continue

        # One frame can be referenced once per split; keep the first output path.
        tasks_by_frame = {}
        for task in tasks:
            tasks_by_frame.setdefault(task['frame_index'], task)

        for frame_index, task in sorted(tasks_by_frame.items()):
            out_path = Path(task['out_path'])
            if out_path.is_file() and not args.overwrite:
                stats['skipped_exists'] += 1
                continue

            ok, frame = extract_frame(cap, frame_index, args.frame_base)
            if not ok:
                stats['failed_frames'] += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), frame, encode_params):
                stats['failed_frames'] += 1
                continue
            stats['written'] += 1

        cap.release()

    summary_path = (
        Path(args.summary)
        if args.summary
        else root / 'annotations' / 'extract_muot3m_bbox_frames_summary.json'
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(stats, file, indent=2)

    print('\nSummary')
    print('=' * 80)
    for key in (
            'requested_images',
            'written',
            'skipped_exists',
            'missing_video',
            'bad_video',
            'failed_frames',
            'missing_image_fields'):
        print('{}: {}'.format(key, stats[key]))
    print('summary:', summary_path)


if __name__ == '__main__':
    main()
