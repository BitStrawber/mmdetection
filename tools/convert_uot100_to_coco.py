#!/usr/bin/env python3
"""Convert UOT100 tracking annotations to one COCO detection JSON.

UOT100 contains sequence folders with ``groundtruth_rect.txt``. Some sequences
already contain extracted frames in ``img/`` while others only contain an mp4.
Use ``--extract-frames`` to generate frame images from mp4 files when frames are
missing or their count does not match the number of annotation lines.
"""

import argparse
import json
import os
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
FRAME_DIR_NAMES = ('img', 'imgs', 'images')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        required=True,
        help='Root directory containing UOT100 sequence folders.')
    parser.add_argument(
        '--out',
        required=True,
        help='Output COCO json path, e.g. annotations/instances_all.json.')
    parser.add_argument(
        '--keep-trailing-digits',
        action='store_true',
        help='Keep sequence names such as ArmyDiver1 as separate categories.')
    parser.add_argument(
        '--skip-invalid',
        action='store_true',
        default=True,
        help='Skip invalid or zero-area boxes.')
    parser.add_argument(
        '--extract-frames',
        action='store_true',
        help='Extract mp4 frames when frame images are missing or mismatched.')
    parser.add_argument(
        '--force-extract',
        action='store_true',
        help='Regenerate extracted frames even if an extracted frame dir exists.')
    parser.add_argument(
        '--extracted-dir-name',
        default='img_extracted',
        help='Subdirectory name used for generated frames.')
    parser.add_argument(
        '--jpg-quality',
        type=int,
        default=95,
        help='JPEG quality for extracted frames.')
    return parser.parse_args()


def natural_key(path):
    stem = Path(path).stem
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', stem)]


def category_name_from_sequence(sequence_name, keep_trailing_digits=False):
    if keep_trailing_digits:
        return sequence_name
    return re.sub(r'\d+$', '', sequence_name)


def parse_bbox_line(line):
    parts = [x for x in re.split(r'[\s,]+', line.strip()) if x]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in parts[:4]]
    except ValueError:
        return None
    return [x, y, w, h]


def read_bboxes(path):
    bboxes = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            bbox = parse_bbox_line(line)
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def list_frames(img_dir):
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        return []
    frames = [
        path for path in img_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_SUFFIXES
    ]
    return sorted(frames, key=natural_key)


def image_size(path):
    with Image.open(path) as img:
        return img.width, img.height


def clamp_bbox(bbox, width, height):
    x, y, w, h = bbox
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + max(0.0, w))
    y2 = min(float(height), y + max(0.0, h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def find_video(seq_dir):
    videos = [
        path for path in Path(seq_dir).iterdir()
        if path.is_file() and path.suffix in VIDEO_SUFFIXES
    ]
    if not videos:
        return None
    return sorted(videos, key=lambda p: (-p.stat().st_size, p.name))[0]


def sequence_dirs(root):
    return [
        path for path in sorted(Path(root).iterdir())
        if path.is_dir() and (path / 'groundtruth_rect.txt').is_file()
    ]


def clean_generated_frames(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMG_SUFFIXES:
            path.unlink()


def sample_indices(source_count, target_count):
    if source_count <= 0 or target_count <= 0:
        return []
    if target_count == 1:
        return [0]
    if source_count == target_count:
        return list(range(source_count))
    return [
        min(source_count - 1, int(round(i * (source_count - 1) / (target_count - 1))))
        for i in range(target_count)
    ]


def extract_frames_from_video(video_path, out_dir, target_count, force=False, jpg_quality=95):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'OpenCV is required for --extract-frames. Install opencv-python '
            'or run conversion without frame extraction.') from exc

    out_dir = Path(out_dir)
    existing = list_frames(out_dir)
    if existing and len(existing) == target_count and not force:
        return existing, 'reuse'

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_frames(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(source_count, target_count)
    if not indices:
        cap.release()
        return [], 'empty'

    written = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    iterator = tqdm(
        enumerate(indices, start=1),
        total=len(indices),
        desc=f'extract {Path(video_path).parent.name}',
        unit='frame',
        leave=False)
    for out_index, source_index in iterator:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = out_dir / f'{out_index:06d}.jpg'
        cv2.imwrite(str(out_path), frame, encode_params)
        written.append(out_path)
    cap.release()

    frames = list_frames(out_dir)
    if len(frames) != target_count:
        print(
            f'Warning: extracted {len(frames)} frames from {video_path}, '
            f'expected {target_count}.')
    return frames, 'extract'


def choose_frames(seq, bbox_count, args):
    candidates = []
    for dir_name in FRAME_DIR_NAMES:
        frame_dir = seq / dir_name
        frames = list_frames(frame_dir)
        if frames:
            candidates.append((frame_dir, frames, 'existing'))

    extracted_dir = seq / args.extracted_dir_name
    extracted = list_frames(extracted_dir)
    if extracted:
        candidates.insert(0, (extracted_dir, extracted, 'extracted-existing'))

    for _, frames, source in candidates:
        if len(frames) == bbox_count:
            return frames, source, 0

    video = find_video(seq)
    if args.extract_frames and video is not None:
        frames, source = extract_frames_from_video(
            video_path=video,
            out_dir=extracted_dir,
            target_count=bbox_count,
            force=args.force_extract,
            jpg_quality=args.jpg_quality)
        return frames, source, abs(len(frames) - bbox_count)

    if candidates:
        best_dir, best_frames, source = max(candidates, key=lambda item: len(item[1]))
        return best_frames, f'{source}:{best_dir.name}', abs(len(best_frames) - bbox_count)

    return [], 'missing', bbox_count


def main():
    args = parse_args()
    root = Path(args.root)
    out_path = Path(args.out)

    sequences = sequence_dirs(root)
    if not sequences:
        raise RuntimeError(f'No UOT100 sequence folders found under {root}')

    category_names = sorted({
        category_name_from_sequence(seq.name, args.keep_trailing_digits)
        for seq in sequences
    })
    cat_id_by_name = {name: idx + 1 for idx, name in enumerate(category_names)}

    coco = {
        'info': {
            'description': 'UOT100 tracking converted to COCO detection',
            'root': str(root),
            'extract_frames': args.extract_frames,
            'extracted_dir_name': args.extracted_dir_name,
        },
        'licenses': [],
        'categories': [
            {'id': cat_id_by_name[name], 'name': name}
            for name in category_names
        ],
        'images': [],
        'annotations': [],
    }

    image_id = 1
    ann_id = 1
    skipped_invalid = 0
    skipped_mismatch = 0
    skipped_no_frames = 0
    extracted_sequences = 0
    reused_extracted_sequences = 0
    exact_existing_sequences = 0
    fallback_mismatch_sequences = 0

    for seq in tqdm(sequences, desc='convert UOT100', unit='seq'):
        category_name = category_name_from_sequence(seq.name, args.keep_trailing_digits)
        category_id = cat_id_by_name[category_name]
        bboxes = read_bboxes(seq / 'groundtruth_rect.txt')
        frames, frame_source, mismatch = choose_frames(seq, len(bboxes), args)

        if not frames:
            skipped_no_frames += 1
            continue
        if frame_source == 'extract':
            extracted_sequences += 1
        elif frame_source == 'reuse':
            reused_extracted_sequences += 1
        elif frame_source in ('existing', 'extracted-existing'):
            exact_existing_sequences += 1
        if mismatch:
            fallback_mismatch_sequences += 1

        pair_count = min(len(frames), len(bboxes))
        skipped_mismatch += abs(len(frames) - len(bboxes))

        for frame_path, bbox in zip(frames[:pair_count], bboxes[:pair_count]):
            width, height = image_size(frame_path)
            clipped_bbox = clamp_bbox(bbox, width, height)
            area = clipped_bbox[2] * clipped_bbox[3]
            if args.skip_invalid and area <= 0:
                skipped_invalid += 1
                continue

            rel_file = frame_path.relative_to(root).as_posix()
            coco['images'].append({
                'id': image_id,
                'file_name': rel_file,
                'width': width,
                'height': height,
                'sequence': seq.name,
                'frame_source': frame_source,
            })
            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [round(float(v), 3) for v in clipped_bbox],
                'area': round(float(area), 3),
                'iscrowd': 0,
            })
            image_id += 1
            ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f'Sequences found: {len(sequences)}')
    print(f'Categories: {len(coco["categories"])}')
    print(f'Images: {len(coco["images"])}')
    print(f'Annotations: {len(coco["annotations"])}')
    print(f'Exact existing-frame sequences: {exact_existing_sequences}')
    print(f'Extracted sequences: {extracted_sequences}')
    print(f'Reused extracted sequences: {reused_extracted_sequences}')
    print(f'Fallback mismatched sequences: {fallback_mismatch_sequences}')
    print(f'Skipped sequences without frames: {skipped_no_frames}')
    print(f'Skipped invalid boxes: {skipped_invalid}')
    print(f'Frame/bbox count mismatches skipped: {skipped_mismatch}')
    print(f'Output: {out_path}')


if __name__ == '__main__':
    main()
