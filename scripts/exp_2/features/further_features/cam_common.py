#!/usr/bin/env python3
"""Shared, dependency-light helpers for detector CAM visualizations."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPSILON = 1e-12


def existing_file(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f'Required file is missing or empty: {path}')
    return path


def read_json(path: Union[str, Path]) -> Any:
    with existing_file(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with existing_file(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: expected JSON object')
            rows.append(value)
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def atomic_write_json(path: Union[str, Path], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(str(temporary), str(destination))


def atomic_write_jsonl(
    path: Union[str, Path], rows: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    count = 0
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write('\n')
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError('Refusing to write an empty JSONL file')
    os.replace(str(temporary), str(destination))


def atomic_save_npz(path: Union[str, Path], **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(destination))


def write_tsv(
    path: Union[str, Path], rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f'Refusing to write empty TSV: {path}')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(destination))


def parse_csv(value: Optional[str]) -> List[str]:
    return [item.strip() for item in (value or '').split(',') if item.strip()]


def clean_name(value: Any) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')


def instance_key(image_id: int, annotation_id: int, layer: str) -> str:
    return f'{int(image_id)}:{int(annotation_id)}:{layer}'


def raw_cam_path(
    root: Path,
    model_id: str,
    image_id: int,
    annotation_id: int,
    layer: str,
) -> Path:
    return (
        root / 'raw_cam' / clean_name(model_id) /
        f'image_{int(image_id):08d}' / f'ann_{int(annotation_id):08d}' /
        f'{clean_name(layer)}.npz'
    )


def instance_metadata_path(
    root: Path,
    model_id: str,
    image_id: int,
    annotation_id: int,
) -> Path:
    return (
        root / 'raw_cam' / clean_name(model_id) /
        f'image_{int(image_id):08d}' / f'ann_{int(annotation_id):08d}' /
        'instance.json'
    )


def finite_percentiles(
    array: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[float, float]:
    values = np.asarray(array, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    return low, high


def normalize_with_limits(array: np.ndarray, low: float, high: float) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(value.shape, dtype=np.float32)
    result = (value - float(low)) / float(high - low)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def resize_map(array: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(array, dtype=np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)


def blue_yellow_rgb(normalized: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Map [0, 1] to a blue-cyan-yellow palette without a red endpoint."""
    value = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    if gamma != 1.0:
        value = np.power(value, gamma)
    anchors = np.asarray([
        [3, 18, 74],
        [0, 72, 170],
        [0, 175, 220],
        [113, 219, 174],
        [255, 238, 35],
    ], dtype=np.float32)
    scaled = value * float(len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def jet_rgb(normalized: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Map [0, 1] to OpenCV JET after an optional display-only gamma."""
    value = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    if gamma != 1.0:
        value = np.power(value, gamma)
    bgr = cv2.applyColorMap(
        np.uint8(np.clip(value, 0.0, 1.0) * 255.0), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_rgb(path: Union[str, Path]) -> np.ndarray:
    with Image.open(existing_file(path)) as image:
        return np.asarray(image.convert('RGB'))


def save_rgb(
    path: Union[str, Path], rgb: np.ndarray, compress_level: int = 3,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB').save(
        destination, format='PNG', compress_level=int(compress_level))


def overlay_heatmap(
    image: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    base = np.asarray(image, dtype=np.float32)
    heat = np.asarray(heatmap_rgb, dtype=np.float32)
    return np.clip((1.0 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)


def default_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        'C:/Windows/Fonts/arial.ttf',
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_box(
    rgb: np.ndarray,
    box_xyxy: Sequence[float],
    label: str = '',
    color: Tuple[int, int, int] = (255, 230, 20),
    width: int = 3,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    if label:
        font = default_font(17)
        bounds = draw.textbbox((0, 0), label, font=font)
        text_w = bounds[2] - bounds[0]
        text_h = bounds[3] - bounds[1]
        top = max(0, int(y1) - text_h - 8)
        left = max(0, int(x1))
        draw.rectangle(
            (left, top, left + text_w + 8, top + text_h + 6), fill=color)
        draw.text((left + 4, top + 2), label, fill=(0, 0, 0), font=font)
    return np.asarray(image)


def labeled_tile(
    rgb: np.ndarray,
    label: str,
    tile_width: int,
    tile_height: int,
    label_height: int = 34,
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
    image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
    canvas = Image.new(
        'RGB', (tile_width, tile_height + label_height), (255, 255, 255))
    left = (tile_width - image.width) // 2
    top = (tile_height - image.height) // 2
    canvas.paste(image, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = default_font(16)
    bounds = draw.textbbox((0, 0), label, font=font)
    text_w = bounds[2] - bounds[0]
    draw.text(
        (max(4, (tile_width - text_w) // 2), tile_height + 7),
        label,
        fill=(15, 15, 15),
        font=font,
    )
    return canvas


def compose_grid(rows: Sequence[Sequence[Image.Image]], gap: int = 3) -> Image.Image:
    if not rows or not rows[0]:
        raise ValueError('Cannot compose an empty grid')
    columns = max(len(row) for row in rows)
    cell_w = max(tile.width for row in rows for tile in row)
    cell_h = max(tile.height for row in rows for tile in row)
    canvas = Image.new(
        'RGB',
        (columns * cell_w + (columns - 1) * gap,
         len(rows) * cell_h + (len(rows) - 1) * gap),
        (255, 255, 255),
    )
    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            canvas.paste(
                tile,
                (column_index * (cell_w + gap), row_index * (cell_h + gap)),
            )
    return canvas


def clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(x1, 0.0), max(float(width - 1), 0.0))
    y1 = min(max(y1, 0.0), max(float(height - 1), 0.0))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return [x1, y1, x2, y2]


def box_mask(box: Sequence[float], width: int, height: int) -> np.ndarray:
    clipped = clip_box(box, width, height)
    x1, y1 = int(np.floor(clipped[0])), int(np.floor(clipped[1]))
    x2, y2 = int(np.ceil(clipped[2])), int(np.ceil(clipped[3]))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def cam_metrics(
    cam_original_size: np.ndarray,
    target_box: Sequence[float],
    all_boxes: Sequence[Sequence[float]],
) -> Dict[str, Union[float, int]]:
    cam = np.maximum(
        np.nan_to_num(np.asarray(cam_original_size, dtype=np.float64)), 0.0)
    height, width = cam.shape
    target = box_mask(target_box, width, height)
    any_gt = np.zeros((height, width), dtype=bool)
    for box in all_boxes:
        any_gt |= box_mask(box, width, height)
    scene_background = ~any_gt
    target_values = cam[target]
    background_values = cam[scene_background]
    total_energy = float(cam.sum())
    target_energy = float(target_values.sum())
    any_gt_energy = float(cam[any_gt].sum())
    target_mean = float(target_values.mean()) if target_values.size else 0.0
    background_mean = (
        float(background_values.mean()) if background_values.size else 0.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    x1, y1, x2, y2 = [float(value) for value in target_box]
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), EPSILON)
    peak_distance = float(np.hypot(peak_x - center_x, peak_y - center_y))
    flattened = cam.reshape(-1)
    positive_indices = np.flatnonzero(flattened > 0)
    if positive_indices.size:
        selected_count = min(
            positive_indices.size,
            max(1, int(np.ceil(0.20 * flattened.size))),
        )
        positive_values = flattened[positive_indices]
        selected_positive = np.argpartition(
            positive_values, -selected_count)[-selected_count:]
        top_indices = positive_indices[selected_positive]
        top_mask_flat = np.zeros(flattened.shape, dtype=bool)
        top_mask_flat[top_indices] = True
        top_mask = top_mask_flat.reshape(cam.shape)
        threshold = float(flattened[top_indices].min())
    else:
        top_mask = np.zeros(cam.shape, dtype=bool)
        threshold = 0.0
    intersection = int(np.logical_and(top_mask, target).sum())
    union = int(np.logical_or(top_mask, target).sum())
    probability = cam.reshape(-1) / max(total_energy, EPSILON)
    probability = probability[probability > 0]
    entropy = float(-(probability * np.log(probability)).sum()) if probability.size else 0.0
    normalized_entropy = entropy / max(float(np.log(cam.size)), EPSILON)
    return {
        'energy_in_target_box': target_energy / max(total_energy, EPSILON),
        'energy_in_any_gt_box': any_gt_energy / max(total_energy, EPSILON),
        'target_mean_response': target_mean,
        'scene_background_mean_response': background_mean,
        'target_to_background_ratio': target_mean / max(background_mean, EPSILON),
        'pointing_game_hit': int(bool(target[peak_y, peak_x])),
        'top20_iou_with_target': intersection / float(max(union, 1)),
        'top20_area_fraction': float(top_mask.mean()),
        'normalized_entropy': normalized_entropy,
        'peak_distance_over_box_diagonal': peak_distance / diagonal,
        'raw_cam_sum': total_energy,
        'raw_cam_max': float(cam.max()),
        'top20_threshold': threshold,
    }
