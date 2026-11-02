from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_boxes(
    boxes: np.ndarray,
    image_height: int,
    image_width: int,
    tolerance: float = 1e-4,
) -> None:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected N x 4 xyxy boxes, got {boxes.shape}")
    if not np.isfinite(boxes).all():
        raise ValueError("Boxes contain NaN or Inf")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("Every box must satisfy x2 > x1 and y2 > y1")
    if np.any(boxes[:, 0] < -tolerance) or np.any(boxes[:, 1] < -tolerance):
        raise ValueError("Box coordinates must be non-negative")
    if np.any(boxes[:, 2] > image_width + tolerance):
        raise ValueError("Box x2 exceeds the network input width")
    if np.any(boxes[:, 3] > image_height + tolerance):
        raise ValueError("Box y2 exceeds the network input height")


def clip_boxes_to_image(
    boxes: np.ndarray,
    image_height: int,
    image_width: int,
    max_overshoot: float = 2.0,
) -> tuple[np.ndarray, float]:
    values = np.asarray(boxes, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected N x 4 xyxy boxes, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Boxes contain NaN or Inf")
    overshoot = max(
        0.0,
        float(np.max(-values[:, [0, 1]], initial=0.0)),
        float(np.max(values[:, [2]] - image_width, initial=0.0)),
        float(np.max(values[:, [3]] - image_height, initial=0.0)),
    )
    if overshoot > float(max_overshoot):
        raise ValueError(
            f"Box overshoot {overshoot:.4f} exceeds tolerance {max_overshoot}"
        )
    values[:, [0, 2]] = np.clip(values[:, [0, 2]], 0.0, image_width)
    values[:, [1, 3]] = np.clip(values[:, [1, 3]], 0.0, image_height)
    validate_boxes(values, image_height, image_width)
    return values, overshoot


def project_boxes_to_feature(
    boxes: np.ndarray,
    image_hw: Sequence[int],
    feature_hw: Sequence[int],
    clip: bool = True,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    image_h, image_w = (int(image_hw[0]), int(image_hw[1]))
    feature_h, feature_w = (int(feature_hw[0]), int(feature_hw[1]))
    if min(image_h, image_w, feature_h, feature_w) <= 0:
        raise ValueError("Image and feature dimensions must be positive")
    validate_boxes(boxes, image_h, image_w)
    scale = np.asarray(
        [
            feature_w / float(image_w),
            feature_h / float(image_h),
            feature_w / float(image_w),
            feature_h / float(image_h),
        ],
        dtype=np.float64,
    )
    projected = boxes * scale
    if clip:
        projected[:, [0, 2]] = np.clip(projected[:, [0, 2]], 0.0, feature_w)
        projected[:, [1, 3]] = np.clip(projected[:, [1, 3]], 0.0, feature_h)
    if np.any(projected[:, 2] <= projected[:, 0]):
        raise ValueError("A projected box has zero width")
    if np.any(projected[:, 3] <= projected[:, 1]):
        raise ValueError("A projected box has zero height")
    return projected


def unproject_boxes_from_feature(
    boxes: np.ndarray,
    image_hw: Sequence[int],
    feature_hw: Sequence[int],
) -> np.ndarray:
    image_h, image_w = (int(image_hw[0]), int(image_hw[1]))
    feature_h, feature_w = (int(feature_hw[0]), int(feature_hw[1]))
    scale = np.asarray(
        [
            image_w / float(feature_w),
            image_h / float(feature_h),
            image_w / float(feature_w),
            image_h / float(feature_h),
        ],
        dtype=np.float64,
    )
    return np.asarray(boxes, dtype=np.float64) * scale


def roundtrip_error(
    boxes: np.ndarray,
    image_hw: Sequence[int],
    feature_hw: Sequence[int],
) -> np.ndarray:
    projected = project_boxes_to_feature(boxes, image_hw, feature_hw)
    restored = unproject_boxes_from_feature(projected, image_hw, feature_hw)
    return np.abs(restored - np.asarray(boxes, dtype=np.float64))


def normalized_box_centers(
    boxes: np.ndarray,
    feature_hw: Sequence[int],
) -> np.ndarray:
    feature_h, feature_w = int(feature_hw[0]), int(feature_hw[1])
    boxes = np.asarray(boxes, dtype=np.float64)
    centers = np.empty((len(boxes), 2), dtype=np.float64)
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / (2.0 * feature_w)
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / (2.0 * feature_h)
    return centers


def coordinate_feature_map(feature_hw: Sequence[int]) -> np.ndarray:
    feature_h, feature_w = int(feature_hw[0]), int(feature_hw[1])
    x = np.linspace(0.0, 1.0, feature_w, dtype=np.float64)
    y = np.linspace(0.0, 1.0, feature_h, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.stack([x_grid, y_grid], axis=0)
