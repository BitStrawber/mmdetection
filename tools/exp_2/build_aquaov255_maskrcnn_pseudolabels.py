#!/usr/bin/env python3
"""Build a DFUI-like AquaOV255 pseudo-instance dataset from a frozen Mask R-CNN.

Selection is independent of AquaOV255 ground-truth masks.  An image is retained
only when the frozen teacher predicts at least one instance with score >= the
configured threshold. The resulting masks, boxes, teacher labels, and scores are
written as COCO annotations for MMDetection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Prepared AquaOV255 images directory.")
    parser.add_argument("--config", type=Path, required=True, help="Frozen Mask R-CNN MMDetection config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen Mask R-CNN checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="New output root; must not exist.")
    parser.add_argument("--score-thr", type=float, default=0.6, help="Inclusive confidence threshold (default: 0.6).")
    parser.add_argument("--device", default="cuda:0", help="MMDetection inference device.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Deterministic validation fraction of retained images.")
    parser.add_argument("--storage-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--max-images", type=int, default=0, help="Positive cap for a smoke test; 0 uses all images.")
    parser.add_argument("--skip-existing", action="store_true", help="Allow reusing an output root with .complete.")
    return parser.parse_args()


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def deterministic_split(name: str, val_ratio: float) -> str:
    fraction = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    return "val" if fraction < val_ratio else "train"


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    result = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    result["counts"] = result["counts"].decode("ascii")
    return result


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        shutil.copy2(source, destination)


def coco(categories: list[dict[str, Any]]) -> dict[str, Any]:
    return {"images": [], "annotations": [], "categories": categories}


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_thr <= 1:
        die("--score-thr must be in [0, 1]")
    if not 0 < args.val_ratio < 1:
        die("--val-ratio must be in (0, 1)")
    for path, option in ((args.images, "--images"), (args.config, "--config"), (args.checkpoint, "--checkpoint")):
        if not path.exists():
            die(f"{option} does not exist: {path}")
    if args.output.exists():
        if args.skip_existing and (args.output / ".complete").is_file():
            print(f"REUSE_COMPLETE={args.output}")
            return
        die(f"--output already exists: {args.output}")

    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as error:
        die(f"MMDetection inference dependencies are unavailable: {error}")
    try:
        import pycocotools.mask  # noqa: F401
    except ImportError as error:
        die(f"pycocotools is required for COCO mask export: {error}")

    images = sorted(path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        die(f"no image files found in {args.images}")
    if args.max_images > 0:
        images = images[: args.max_images]

    args.output.mkdir(parents=True)
    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    classes = tuple(model.dataset_meta.get("classes", ()))
    if not classes:
        die("teacher config/checkpoint has no dataset_meta.classes")
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(classes)]
    datasets = {"train": coco(categories), "val": coco(categories)}
    next_image_id = {"train": 1, "val": 1}
    next_annotation_id = {"train": 1, "val": 1}
    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for ordinal, source in enumerate(images, start=1):
        result = inference_detector(model, str(source))
        instances = result.pred_instances.cpu()
        scores = instances.scores.numpy()
        labels = instances.labels.numpy()
        boxes = instances.bboxes.numpy()
        masks = instances.masks.numpy() if hasattr(instances, "masks") else None
        selected = np.asarray(
            [
                index
                for index in np.flatnonzero(scores >= args.score_thr)
                if float(boxes[index][2]) > float(boxes[index][0])
                and float(boxes[index][3]) > float(boxes[index][1])
            ]
        )
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "image_name": source.name,
            "source_image": str(source.resolve()),
            "predicted_instances_total": int(len(scores)),
            "predicted_instances_at_or_above_threshold": int(len(selected)),
            "score_threshold": args.score_thr,
            "retained": bool(len(selected)),
        }
        if not len(selected):
            records.append(record)
            continue

        split = deterministic_split(source.name, args.val_ratio)
        with Image.open(source) as image:
            width, height = image.size
        image_id = next_image_id[split]
        next_image_id[split] += 1
        output_name = f"{image_id:06d}__{source.name}"
        destination = args.output / split / "images" / output_name
        link_or_copy(source, destination, args.storage_mode)
        datasets[split]["images"].append({"id": image_id, "file_name": output_name, "width": width, "height": height})

        names: list[str] = []
        for index in selected:
            label = int(labels[index])
            if not 0 <= label < len(classes):
                die(f"teacher emitted class index {label}, but has {len(classes)} configured classes")
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if not box_width or not box_height:
                continue
            annotation: dict[str, Any] = {
                "id": next_annotation_id[split],
                "image_id": image_id,
                "category_id": label + 1,
                "bbox": [x1, y1, box_width, box_height],
                "score": float(scores[index]),
                "iscrowd": 0,
            }
            if masks is not None:
                mask = masks[index].astype(bool)
                annotation["segmentation"] = encode_mask(mask)
                annotation["area"] = int(mask.sum())
            else:
                annotation["segmentation"] = []
                annotation["area"] = box_width * box_height
            datasets[split]["annotations"].append(annotation)
            next_annotation_id[split] += 1
            class_counts[classes[label]] += 1
            names.append(classes[label])

        record.update({"split": split, "output_image": str(destination.relative_to(args.output)), "teacher_categories": ";".join(names)})
        records.append(record)
        if ordinal % 100 == 0 or ordinal == len(images):
            retained = sum(item["retained"] for item in records)
            print(f"processed={ordinal}/{len(images)} retained_images={retained}", flush=True)

    for split, data in datasets.items():
        annotation_dir = args.output / split / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        (annotation_dir / f"instances_{split}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    columns = ("ordinal", "image_name", "source_image", "predicted_instances_total", "predicted_instances_at_or_above_threshold", "score_threshold", "retained", "split", "output_image", "teacher_categories")
    with (args.output / "selection_manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for record in records:
            handle.write("\t".join(str(record.get(column, "")) for column in columns) + "\n")

    summary = {
        "source_images_examined": len(images),
        "retained_images": sum(item["retained"] for item in records),
        "rejected_images": sum(not item["retained"] for item in records),
        "score_threshold": args.score_thr,
        "teacher_config": str(args.config.resolve()),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_classes": list(classes),
        "storage_mode": args.storage_mode,
        "train_images": len(datasets["train"]["images"]),
        "train_annotations": len(datasets["train"]["annotations"]),
        "val_images": len(datasets["val"]["images"]),
        "val_annotations": len(datasets["val"]["annotations"]),
        "teacher_category_instance_counts": dict(sorted(class_counts.items())),
        "ground_truth_masks_used_for_selection": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={args.output}")
    print("RESULT=PASS_AQUAOV255_MASKRCNN_PSEUDOLABELS")


if __name__ == "__main__":
    main()
