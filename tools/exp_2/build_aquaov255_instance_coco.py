#!/usr/bin/env python3
"""Convert prepared AquaOV255 semantic masks into a deterministic COCO instance split.

The official AquaOV255 layout uses one image/mask pair per fine-grained class.
This tool deliberately verifies that each mask has exactly one non-background
label before treating its non-zero region as an instance.  A mismatch stops the
pipeline instead of silently assigning an incorrect category.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--storage-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def class_name_from_stem(stem: str) -> str:
    prefix, separator, suffix = stem.rpartition("_")
    if not separator or not prefix or not suffix.isdigit():
        die(f"Cannot infer an AquaOV255 class from filename stem: {stem}")
    return prefix


def list_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def stable_is_validation(stem: str, seed: int, val_ratio: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_ratio


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def rle_for(mask: np.ndarray) -> tuple[dict[str, Any], list[float], float]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = [float(value) for value in mask_utils.toBbox(encoded).tolist()]
    area = float(mask_utils.area(encoded))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded, bbox, area


def main() -> int:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        die("--val-ratio must be strictly between 0 and 1")
    if not args.images.is_dir() or not args.masks.is_dir():
        die("--images and --masks must both be existing directories")
    if not args.check_only and args.output.exists():
        die(f"Output already exists: {args.output}")

    images = list_files(args.images)
    if not images:
        die(f"No images found in {args.images}")

    masks_by_stem: dict[str, Path] = {}
    for path in list_files(args.masks):
        if path.stem in masks_by_stem:
            die(f"Duplicate mask stem: {path.stem}")
        masks_by_stem[path.stem] = path

    records: list[dict[str, Any]] = []
    for image_path in images:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            die(f"No mask with matching stem for image: {image_path.name}")
        records.append(
            {
                "image": image_path,
                "mask": mask_path,
                "class_name": class_name_from_stem(image_path.stem),
                "split": "val" if stable_is_validation(image_path.stem, args.seed, args.val_ratio) else "train",
            }
        )

    image_stems = {item["image"].stem for item in records}
    unmatched_masks = sorted(set(masks_by_stem) - image_stems)
    if unmatched_masks:
        die(f"Found {len(unmatched_masks)} masks without an image; first: {unmatched_masks[0]}")

    classes = sorted({str(item["class_name"]) for item in records})
    class_ids = {name: index for index, name in enumerate(classes, start=1)}
    split_counts = Counter(str(item["split"]) for item in records)
    class_split_counts = Counter((str(item["split"]), str(item["class_name"])) for item in records)
    missing_val_classes = [name for name in classes if not class_split_counts[("val", name)]]
    missing_train_classes = [name for name in classes if not class_split_counts[("train", name)]]
    if missing_val_classes or missing_train_classes:
        die(
            "Deterministic split left a class empty: "
            f"missing_train={missing_train_classes[:5]} missing_val={missing_val_classes[:5]}. "
            "Use a different --seed or a larger --val-ratio."
        )

    if args.check_only:
        print(f"images={len(records)}")
        print(f"classes={len(classes)}")
        print(f"train_images={split_counts['train']}")
        print(f"val_images={split_counts['val']}")
        print("RESULT=PASS_AQUAOV255_PAIR_AND_SPLIT_PREFLIGHT")
        return 0

    payloads: dict[str, dict[str, Any]] = {
        split: {"images": [], "annotations": [], "categories": [
            {"id": category_id, "name": name, "supercategory": "aquaov255"}
            for name, category_id in class_ids.items()
        ]}
        for split in ("train", "val")
    }
    annotation_id = 1
    manifest_rows = [
        "split\timage_id\tannotation_id\timage_name\tmask_name\tclass_name\tclass_id\tforeground_value\tarea\n"
    ]

    for image_id, record in enumerate(records, start=1):
        image_path = Path(record["image"])
        mask_path = Path(record["mask"])
        split = str(record["split"])
        class_name = str(record["class_name"])

        with Image.open(image_path) as image:
            width, height = image.size
        mask_array = np.asarray(Image.open(mask_path))
        if mask_array.ndim != 2:
            die(f"Mask is not single-channel: {mask_path} shape={mask_array.shape}")
        if mask_array.shape != (height, width):
            die(f"Image/mask size mismatch: {image_path.name} vs {mask_path.name}")

        foreground_values = np.unique(mask_array[mask_array != 0])
        if len(foreground_values) != 1:
            die(
                f"Expected exactly one non-zero semantic label in {mask_path.name}, "
                f"found {foreground_values.tolist()[:10]}"
            )
        foreground_value = int(foreground_values[0])
        binary_mask = mask_array != 0
        rle, bbox, area = rle_for(binary_mask)
        if area <= 0:
            die(f"Empty foreground mask: {mask_path}")

        file_name = f"{image_id:06d}__{image_path.name}"
        destination = args.output / split / "images" / file_name
        materialize(image_path, destination, args.storage_mode)

        payloads[split]["images"].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        payloads[split]["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_ids[class_name],
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        manifest_rows.append(
            f"{split}\t{image_id}\t{annotation_id}\t{image_path.name}\t{mask_path.name}\t"
            f"{class_name}\t{class_ids[class_name]}\t{foreground_value}\t{area:.0f}\n"
        )
        annotation_id += 1

    for split, payload in payloads.items():
        annotation_path = args.output / split / "annotations" / f"instances_{split}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    (args.output / "split_manifest.tsv").write_text("".join(manifest_rows), encoding="utf-8")
    summary = {
        "source_images": str(args.images.resolve()),
        "source_masks": str(args.masks.resolve()),
        "storage_mode": args.storage_mode,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "images": len(records),
        "classes": len(classes),
        "train_images": len(payloads["train"]["images"]),
        "val_images": len(payloads["val"]["images"]),
        "train_instances": len(payloads["train"]["annotations"]),
        "val_instances": len(payloads["val"]["annotations"]),
        "mask_interpretation": "Each verified single-label, non-zero semantic mask becomes one instance.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print("RESULT=PASS_AQUAOV255_INSTANCE_COCO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
