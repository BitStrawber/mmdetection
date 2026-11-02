from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .bbox_mapping import (
    normalized_box_centers,
    project_boxes_to_feature,
    unproject_boxes_from_feature,
)


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.nanpercentile(values, 1))
    high = float(np.nanpercentile(values, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def render_mapping_debug(
    image_path: str | Path,
    output_path: str | Path,
    original_box: Sequence[float],
    original_hw: Sequence[int],
    network_box: Sequence[float],
    img_hw: Sequence[int],
    pad_hw: Sequence[int],
    feature_box: Sequence[float],
    feature_heatmap: np.ndarray,
) -> Path:
    """Render image-space and feature-space box projections side by side."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((int(img_hw[1]), int(img_hw[0])))
    canvas = Image.new("RGB", (int(pad_hw[1]), int(pad_hw[0])), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([float(value) for value in network_box], outline=(255, 0, 0), width=4)
    original_scale = np.asarray(
        [
            int(img_hw[1]) / float(original_hw[1]),
            int(img_hw[0]) / float(original_hw[0]),
            int(img_hw[1]) / float(original_hw[1]),
            int(img_hw[0]) / float(original_hw[0]),
        ],
        dtype=np.float64,
    )
    original_scaled = np.asarray(original_box, dtype=np.float64) * original_scale
    draw.rectangle(
        [float(value) for value in original_scaled],
        outline=(0, 170, 255),
        width=2,
    )

    heatmap = _normalize_heatmap(feature_heatmap)
    feature_image = Image.fromarray(heatmap).convert("RGB")
    feature_image = feature_image.resize(
        (max(canvas.width, 64), max(canvas.height, 64)),
        resample=Image.Resampling.NEAREST,
    )
    feature_draw = ImageDraw.Draw(feature_image)
    scale_x = feature_image.width / float(feature_heatmap.shape[1])
    scale_y = feature_image.height / float(feature_heatmap.shape[0])
    feature_draw.rectangle(
        [
            float(feature_box[0]) * scale_x,
            float(feature_box[1]) * scale_y,
            float(feature_box[2]) * scale_x,
            float(feature_box[3]) * scale_y,
        ],
        outline=(255, 0, 0),
        width=4,
    )
    combined = Image.new(
        "RGB",
        (canvas.width + feature_image.width, max(canvas.height, feature_image.height)),
        "white",
    )
    combined.paste(canvas, (0, 0))
    combined.paste(feature_image, (canvas.width, 0))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.save(destination)
    return destination


def validate_coordinate_feature(
    feature_box: Sequence[float],
    feature_hw: Sequence[int],
    coordinate_roi_mean: Sequence[float],
    tolerance: float = 0.01,
) -> None:
    expected = normalized_box_centers(
        np.asarray([feature_box], dtype=np.float64), feature_hw
    )[0]
    actual = np.asarray(coordinate_roi_mean, dtype=np.float64)
    error = np.abs(actual - expected)
    if np.any(error > tolerance):
        raise AssertionError(
            f"Coordinate ROI mismatch: expected {expected}, got {actual}"
        )


def validate_mapping_record(
    network_box: Sequence[float],
    img_hw: Sequence[int],
    feature_hw: Sequence[int],
    tolerance: float = 0.5,
) -> np.ndarray:
    projected = project_boxes_to_feature(
        np.asarray([network_box], dtype=np.float64), img_hw, feature_hw
    )
    restored = unproject_boxes_from_feature(projected, img_hw, feature_hw)[0]
    network = np.asarray(network_box, dtype=np.float64)
    error = np.abs(restored - network)
    if float(error.max()) > tolerance:
        raise AssertionError(
            f"Box round-trip error {error.max():.4f} exceeds {tolerance}"
        )
    return error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate coordinate, image, and feature mappings."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a deterministic coordinate-mapping smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.self_test:
        raise SystemExit("Use --self-test or call render_mapping_debug from extraction.")
    image_hw = (500, 1000)
    feature_hw = (32, 64)
    boxes = np.asarray(
        [[0.0, 0.0, 250.0, 125.0], [100.0, 50.0, 900.0, 450.0]]
    )
    projected = project_boxes_to_feature(boxes, image_hw, feature_hw)
    restored = unproject_boxes_from_feature(projected, image_hw, feature_hw)
    error = np.abs(restored - boxes).max()
    if error > 1e-6:
        raise AssertionError(f"Round-trip error is too large: {error}")
    print(f"PASS box mapping round-trip max_error={error:.8f}")


if __name__ == "__main__":
    main()
