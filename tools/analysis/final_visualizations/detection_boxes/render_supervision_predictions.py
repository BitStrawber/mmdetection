#!/usr/bin/env python3
"""Render MMDetection predictions with the official supervision annotators.

The script accepts a newline-delimited image manifest, runs one MMDetection
model, and writes either box-only output or box, mask, and box+mask outputs.
Boxes and masks are always drawn by supervision; no prediction overlay is
manually painted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SEGMENT_VARIANTS = ("box", "mask", "box_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=15)
    parser.add_argument(
        "--variants",
        default="box,mask,box_mask",
        help="Comma-separated subset of box,mask,box_mask for segment models.",
    )
    return parser.parse_args()


def image_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"Image list is empty: {path}")
    missing = [str(item) for item in paths if not item.is_file()]
    invalid = [str(item) for item in paths if item.suffix.lower() not in IMAGE_SUFFIXES]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) are missing: {missing[:5]}")
    if invalid:
        raise ValueError(f"{len(invalid)} path(s) are not supported images: {invalid[:5]}")
    return paths


def parse_variants(value: str, task: str) -> tuple[str, ...]:
    if task == "detect":
        return ("box",)
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(variants) - set(SEGMENT_VARIANTS)
    if not variants or unknown:
        raise ValueError(
            f"Segment variants must be a subset of {SEGMENT_VARIANTS}; got {variants}"
        )
    return variants


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predicted_masks(instances: Any) -> np.ndarray | None:
    masks = getattr(instances, "masks", None)
    if masks is None:
        return None
    if hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    else:
        masks = as_numpy(masks)
    return np.asarray(masks, dtype=np.bool_)


def selected_indices(
    scores: np.ndarray,
    threshold: float,
    maximum: int,
) -> np.ndarray:
    indices = np.flatnonzero(scores >= float(threshold))
    if indices.size == 0:
        return indices
    order = np.argsort(-scores[indices], kind="stable")
    return indices[order[: int(maximum)]]


def annotation_labels(
    class_names: list[str],
    class_ids: np.ndarray,
    scores: np.ndarray,
) -> list[str]:
    labels: list[str] = []
    for class_id, score in zip(class_ids.tolist(), scores.tolist()):
        name = (
            class_names[int(class_id)]
            if 0 <= int(class_id) < len(class_names)
            else f"class_{int(class_id)}"
        )
        labels.append(f"{name} {float(score):.2f}")
    return labels


def annotate(
    image: np.ndarray,
    detections: Any,
    labels: list[str],
    variant: str,
    annotators: dict[str, Any],
) -> np.ndarray:
    scene = image.copy()
    if variant in {"mask", "box_mask"}:
        scene = annotators["mask"].annotate(scene=scene, detections=detections)
    if variant in {"box", "box_mask"}:
        scene = annotators["box"].annotate(scene=scene, detections=detections)
        scene = annotators["label"].annotate(
            scene=scene,
            detections=detections,
            labels=labels,
        )
    return scene


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        import cv2
        import supervision as sv
        from mmdet.apis import inference_detector, init_detector
    except ImportError as error:
        raise SystemExit(
            "MMDetection, OpenCV, and supervision are required in the active "
            f"environment: {error}"
        ) from error

    config = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    images = image_manifest(Path(args.image_list).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    variants = parse_variants(args.variants, args.task)

    for required in (config, checkpoint):
        if not required.is_file() or required.stat().st_size <= 0:
            raise FileNotFoundError(f"Required file is missing or empty: {required}")

    model = init_detector(str(config), str(checkpoint), device=args.device)
    class_names = list((model.dataset_meta or {}).get("classes", []))
    if not class_names:
        raise RuntimeError("The model checkpoint/config has no dataset_meta classes")

    annotators = {
        "box": sv.BoxAnnotator(),
        "label": sv.LabelAnnotator(),
        "mask": sv.MaskAnnotator(),
    }
    for variant in variants:
        (output_root / variant).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image: {image_path}")

        result = inference_detector(model, str(image_path))
        instances = result.pred_instances.cpu()
        scores = as_numpy(instances.scores).astype(np.float32, copy=False)
        class_ids = as_numpy(instances.labels).astype(np.int64, copy=False)
        boxes = as_numpy(instances.bboxes).astype(np.float32, copy=False)
        masks = predicted_masks(instances) if args.task == "segment" else None
        keep = selected_indices(scores, args.score_threshold, args.max_detections)

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if masks is not None:
            masks = masks[keep]
        if args.task == "segment" and masks is None:
            raise RuntimeError("The segmentation model did not return prediction masks")

        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
            mask=masks,
        )
        labels = annotation_labels(class_names, class_ids, scores)

        output_paths: dict[str, str] = {}
        for variant in variants:
            rendered = annotate(image, detections, labels, variant, annotators)
            destination = output_root / variant / f"{index:03d}_{image_path.stem}.png"
            if not cv2.imwrite(str(destination), rendered):
                raise RuntimeError(f"Failed to write visualization: {destination}")
            output_paths[variant] = str(destination)

        records.append(
            {
                "index": index,
                "image": str(image_path),
                "predictions_at_threshold": int(len(keep)),
                "score_threshold": float(args.score_threshold),
                "max_detections": int(args.max_detections),
                "outputs": output_paths,
            }
        )
        if index % 10 == 0 or index == len(images):
            print(f"processed={index}/{len(images)}", flush=True)

    write_jsonl(output_root / "predictions.jsonl", records)
    summary = {
        "model_name": args.model_name,
        "task": args.task,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "images": len(images),
        "classes": class_names,
        "score_threshold": float(args.score_threshold),
        "max_detections": int(args.max_detections),
        "variants": variants,
        "drawing_backend": "supervision",
        "inference_api": "mmdet.apis.inference_detector",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RESULT=PASS_SUPERVISION_VISUALIZATION output={output_root}")


if __name__ == "__main__":
    main()
