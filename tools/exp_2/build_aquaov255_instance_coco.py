#!/usr/bin/env python3
"""Build deterministic instance-COCO splits from official AquaOV255 masks.

Official AquaOV255 stores semantic labels in 16-bit masks. Pixel value 65535
is background; foreground values 0..253 map to category.txt lines 1..254.
Filenames are retained solely for provenance because their embedded numbers do
not form a globally reliable category mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUND_VALUE = 65535


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True, help="Official AquaOV255 category.txt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--expected-images", type=int, default=20722)
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Put every paired image in train; create an empty val split.",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--storage-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_categories(path: Path) -> list[str]:
    if not path.is_file():
        die(f"Official category.txt is missing: {path}")
    categories = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(categories) != 254:
        die(f"Expected 254 foreground entries in category.txt, found {len(categories)}")
    if len(set(categories)) != len(categories):
        die("Official category.txt contains duplicate class names")
    return categories


def stable_key(stem: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()[:8], "big")


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def encode(binary_mask: np.ndarray) -> tuple[dict[str, Any], list[float], float]:
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    bbox = [float(value) for value in mask_utils.toBbox(encoded).tolist()]
    area = float(mask_utils.area(encoded))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded, bbox, area


def read_record(image_path: Path, mask_path: Path, categories: list[str], seed: int) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim != 2:
        die(f"Mask is not single-channel: {mask_path} shape={mask.shape}")
    if mask.shape != (height, width):
        die(f"Image/mask size mismatch: {image_path.name} vs {mask_path.name}")

    label_values = sorted(int(value) for value in np.unique(mask) if int(value) != BACKGROUND_VALUE)
    invalid_values = [value for value in label_values if value < 0 or value >= len(categories)]
    if invalid_values:
        die(f"Mask has invalid foreground labels {invalid_values[:10]}: {mask_path}")
    return {
        "image": image_path,
        "mask": mask_path,
        "width": width,
        "height": height,
        "labels": tuple(label_values),
        "key": stable_key(image_path.stem, seed),
    }


def assign_splits(records: list[dict[str, Any]], val_ratio: float) -> None:
    threshold = int(val_ratio * (2**64))
    for record in records:
        record["split"] = "val" if record["key"] < threshold else "train"

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for label in record["labels"]:
            by_label[label].append(record)

    # Repair the deterministic hash split only when a category would otherwise
    # be absent from one side. The selected record is deterministic by hash.
    for label, members in sorted(by_label.items()):
        if not any(record["split"] == "val" for record in members):
            min(members, key=lambda record: record["key"])["split"] = "val"
        if not any(record["split"] == "train" for record in members):
            max(members, key=lambda record: record["key"])["split"] = "train"

    for label, members in sorted(by_label.items()):
        if not any(record["split"] == "train" for record in members) or not any(record["split"] == "val" for record in members):
            die(f"Could not create both train and val coverage for label {label}")


def components_for_label(mask: np.ndarray, label: int) -> list[np.ndarray]:
    binary = (mask == label).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    return [(labels == component_id) for component_id in range(1, count)]


def main() -> int:
    args = parse_args()
    if args.no_validation:
        if args.val_ratio != 0.1:
            die("Do not combine --no-validation with a custom --val-ratio")
    elif not 0 < args.val_ratio < 1:
        die("--val-ratio must be strictly between 0 and 1")
    if not args.images.is_dir() or not args.masks.is_dir():
        die("--images and --masks must both be existing directories")
    if not args.check_only and args.output.exists():
        die(f"Output already exists: {args.output}")

    categories = load_categories(args.categories)
    masks_by_stem: dict[str, Path] = {}
    for path in image_files(args.masks):
        if path.stem in masks_by_stem:
            die(f"Duplicate mask stem: {path.stem}")
        masks_by_stem[path.stem] = path

    images = image_files(args.images)
    if not images:
        die(f"No images found in {args.images}")
    if len(images) != args.expected_images:
        die(f"Expected {args.expected_images} images, found {len(images)}")
    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            die(f"No mask with matching stem for image: {image_path.name}")
        records.append(read_record(image_path, mask_path, categories, args.seed))
        if index % 5000 == 0:
            print(f"validated_images={index}/{len(images)}", file=sys.stderr)

    image_stems = {item["image"].stem for item in records}
    unmatched_masks = sorted(set(masks_by_stem) - image_stems)
    if unmatched_masks:
        die(f"Found {len(unmatched_masks)} masks without an image; first: {unmatched_masks[0]}")

    if args.no_validation:
        for record in records:
            record["split"] = "train"
    else:
        assign_splits(records, args.val_ratio)
    split_images = Counter(record["split"] for record in records)
    split_labels = Counter((record["split"], label) for record in records for label in record["labels"])
    if len({label for _, label in split_labels}) != len(categories):
        die("Not every official category is represented by an AquaOV255 foreground mask")

    if args.check_only:
        print(f"images={len(records)}")
        print(f"foreground_categories={len(categories)}")
        print(f"train_images={split_images['train']}")
        print(f"val_images={split_images['val']}")
        print(f"train_categories={len({label for split, label in split_labels if split == 'train'})}")
        print(f"val_categories={len({label for split, label in split_labels if split == 'val'})}")
        print(f"empty_mask_images={sum(not record['labels'] for record in records)}")
        print("mask_background_value=65535")
        print("RESULT=PASS_AQUAOV255_OFFICIAL_MASK_MAPPING_AND_SPLIT_PREFLIGHT")
        return 0

    coco_categories = [
        {"id": label + 1, "name": name, "supercategory": "aquaov255"}
        for label, name in enumerate(categories)
    ]
    payloads: dict[str, dict[str, Any]] = {
        split: {"images": [], "annotations": [], "categories": coco_categories}
        for split in ("train", "val")
    }
    annotation_id = 1
    manifest_rows = [
        "split\timage_id\tannotation_id\timage_name\tmask_name\tmask_label\tclass_id\tclass_name\tarea\n"
    ]

    for image_id, record in enumerate(records, start=1):
        image_path = Path(record["image"])
        mask_path = Path(record["mask"])
        split = str(record["split"])
        mask = np.asarray(Image.open(mask_path))
        file_name = f"{image_id:06d}__{image_path.name}"
        materialize(image_path, args.output / split / "images" / file_name, args.storage_mode)
        payloads[split]["images"].append(
            {"id": image_id, "file_name": file_name, "width": record["width"], "height": record["height"]}
        )

        for label in record["labels"]:
            for component in components_for_label(mask, label):
                rle, bbox, area = encode(component)
                if area <= 0:
                    continue
                payloads[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label + 1,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )
                manifest_rows.append(
                    f"{split}\t{image_id}\t{annotation_id}\t{image_path.name}\t{mask_path.name}\t"
                    f"{label}\t{label + 1}\t{categories[label]}\t{area:.0f}\n"
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
        "official_category_file": str(args.categories.resolve()),
        "storage_mode": args.storage_mode,
        "seed": args.seed,
        "val_ratio": None if args.no_validation else args.val_ratio,
        "source_validation": not args.no_validation,
        "images": len(records),
        "foreground_categories": len(categories),
        "mask_background_value": BACKGROUND_VALUE,
        "train_images": len(payloads["train"]["images"]),
        "val_images": len(payloads["val"]["images"]),
        "train_instances": len(payloads["train"]["annotations"]),
        "val_instances": len(payloads["val"]["annotations"]),
        "empty_mask_images": sum(not record["labels"] for record in records),
        "train_empty_mask_images": sum(
            record["split"] == "train" and not record["labels"] for record in records
        ),
        "val_empty_mask_images": sum(
            record["split"] == "val" and not record["labels"] for record in records
        ),
        "mask_interpretation": "Each connected component of each foreground semantic label becomes one instance.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print("RESULT=PASS_AQUAOV255_OFFICIAL_MASK_TO_INSTANCE_COCO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
