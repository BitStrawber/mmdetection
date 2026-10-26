#!/usr/bin/env python3
"""Normalize generated Cocosyn images and annotations to detector geometry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("uwnr", "cut", "syreanet", "uwdf", "watergan"))
    parser.add_argument("--annotation", required=True,
                        help="Original COCO annotation containing all method image IDs.")
    parser.add_argument("--generated-manifest", required=True,
                        help="METHOD_generated_manifest.jsonl with image_id/generated_file.")
    parser.add_argument("--generated-root", required=True,
                        help="Directory containing the actual generated images.")
    parser.add_argument("--source-image-root", required=True,
                        help="Original COCO source image root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--materialize", choices=("copy", "hardlink", "symlink"),
                        default="hardlink")
    parser.add_argument("--target-size", default="640x480",
                        help="WaterGAN generated size as WIDTHxHEIGHT.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images for a smoke test.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[int, Path]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = int(row["image_id"])
        if image_id in result:
            raise SystemExit(f"duplicate manifest image_id: {image_id}")
        result[image_id] = Path(row["generated_file"])
    return result


def resolve_generated(root: Path, relative: Path) -> Path:
    candidates = (root / relative, root / relative.name)
    for path in candidates:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    raise FileNotFoundError(f"generated image not found under {root}: {relative}")


def resolve_source(root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (root / relative, root / relative.name)
    for path in candidates:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    raise FileNotFoundError(f"source image not found under {root}: {file_name}")


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source.resolve())


def fit_affine(source_size: tuple[int, int], target_size: tuple[int, int]):
    """Return cv2 affine mapping and x/y scale for PIL ImageOps.fit centering."""
    source_w, source_h = source_size
    target_w, target_h = target_size
    source_aspect = source_w / float(source_h)
    target_aspect = target_w / float(target_h)
    if source_aspect > target_aspect:
        scaled_w = int(target_h * source_aspect + 0.5)
        scale = target_h / float(source_h)
        crop_x = int((scaled_w - target_w) * 0.5 + 0.5)
        matrix = np.array(
            [[scale, 0.0, -crop_x], [0.0, scale, 0.0]],
            dtype=np.float32,
        )
    else:
        scaled_h = int(target_w / source_aspect + 0.5)
        scale = target_w / float(source_w)
        crop_y = int((scaled_h - target_h) * 0.5 + 0.5)
        matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, -crop_y]],
            dtype=np.float32,
        )
    return matrix, scale


def transform_bbox(bbox, matrix, target_size):
    target_w, target_h = target_size
    x, y, width, height = (float(value) for value in bbox)
    sx, _, tx = (float(value) for value in matrix[0])
    _, sy, ty = (float(value) for value in matrix[1])
    x1 = max(0.0, min(float(target_w), x * sx + tx))
    y1 = max(0.0, min(float(target_h), y * sy + ty))
    x2 = max(0.0, min(float(target_w), (x + width) * sx + tx))
    y2 = max(0.0, min(float(target_h), (y + height) * sy + ty))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def annotation_mask(annotation: dict, height: int, width: int):
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), str):
            rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("ascii"),
            }
        return mask_utils.decode(rle)
    if isinstance(segmentation, list) and segmentation:
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(mask_utils.merge(rles))
    return None


def transform_segmentation(annotation: dict, source_size, target_size, matrix):
    source_w, source_h = source_size
    target_w, target_h = target_size
    source_mask = annotation_mask(annotation, source_h, source_w)
    if source_mask is None:
        return None, None
    output_mask = cv2.warpAffine(
        source_mask.astype(np.uint8),
        matrix,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    area = int(output_mask.sum())
    if area <= 0:
        return None, 0
    encoded = mask_utils.encode(
        np.asfortranarray(output_mask.astype(np.uint8))
    )
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded, area


def main() -> None:
    args = parse_args()
    generated_root = Path(args.generated_root).resolve()
    source_root = Path(args.source_image_root).resolve()
    manifest = load_manifest(Path(args.generated_manifest))
    coco = load_json(Path(args.annotation))
    output_root = Path(args.output_root).resolve()
    image_output = output_root / "images"
    annotation_output = output_root / "annotations"

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    target_w, target_h = (int(value) for value in args.target_size.lower().split("x", 1))
    target_size = (target_w, target_h)
    annotations_by_image = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    output_images = []
    output_annotations = []
    rows = []
    skipped_degenerate = 0
    images = list(coco.get("images", []))
    if args.limit > 0:
        images = images[:args.limit]

    for position, image in enumerate(images, 1):
        image_id = int(image["id"])
        generated_relative = manifest.get(image_id)
        if generated_relative is None:
            raise SystemExit(f"missing generated manifest entry for image_id={image_id}")
        generated_path = resolve_generated(generated_root, generated_relative)
        source_path = resolve_source(source_root, str(image["file_name"]))
        with Image.open(source_path) as source_image, Image.open(generated_path) as generated_image:
            source_image = source_image.convert("RGB")
            generated_image = generated_image.convert("RGB")
            source_size = source_image.size

            output_name = Path(str(image["file_name"])).name
            output_path = image_output / output_name
            if output_path.exists() or output_path.is_symlink():
                if not args.overwrite:
                    raise SystemExit(f"destination exists: {output_path}")
                output_path.unlink()

            output_size = source_size
            matrix = None
            scale = 1.0
            if args.method in {"uwnr", "cut"}:
                normalized = generated_image.resize(source_size, RESAMPLE_BICUBIC)
                normalized.save(output_path)
            elif args.method in {"syreanet", "uwdf"}:
                if generated_image.size != source_size:
                    raise SystemExit(
                        f"{args.method} image {image_id} size={generated_image.size}, "
                        f"source={source_size}"
                    )
                materialize(generated_path, output_path, args.materialize)
            else:
                if generated_image.size != target_size:
                    raise SystemExit(
                        f"watergan image {image_id} size={generated_image.size}, "
                        f"expected={target_size}"
                    )
                matrix, scale = fit_affine(source_size, target_size)
                materialize(generated_path, output_path, args.materialize)
                output_size = target_size

            output_image = dict(image)
            output_image["file_name"] = output_name
            output_image["width"] = output_size[0]
            output_image["height"] = output_size[1]
            output_images.append(output_image)

            for annotation in annotations_by_image.get(image_id, []):
                output_annotation = dict(annotation)
                if args.method == "watergan":
                    bbox = transform_bbox(annotation.get("bbox", []), matrix, target_size)
                    if bbox is None:
                        skipped_degenerate += 1
                        continue
                    output_annotation["bbox"] = bbox
                    segmentation, area = transform_segmentation(
                        annotation, source_size, target_size, matrix
                    )
                    if segmentation is not None:
                        output_annotation["segmentation"] = segmentation
                        output_annotation["area"] = area
                    else:
                        output_annotation.pop("segmentation", None)
                        output_annotation["area"] = bbox[2] * bbox[3]
                output_annotations.append(output_annotation)

            rows.append({
                "method": args.method,
                "image_id": image_id,
                "source": str(source_path),
                "generated": str(generated_path),
                "output": str(output_path),
                "source_size": list(source_size),
                "output_size": list(output_size),
                "scale": scale,
            })
        if position % 500 == 0 or position == len(images):
            print(f"normalized={position}/{len(images)}", flush=True)

    result = dict(coco)
    result["images"] = output_images
    result["annotations"] = output_annotations
    write_json(annotation_output / "instances.json", result)
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_root / "summary.json", {
        "method": args.method,
        "annotation": str(Path(args.annotation).resolve()),
        "generated_manifest": str(Path(args.generated_manifest).resolve()),
        "generated_root": str(generated_root),
        "source_image_root": str(source_root),
        "images": len(output_images),
        "annotations": len(output_annotations),
        "skipped_degenerate_annotations": skipped_degenerate,
        "watergan_target_size": list(target_size),
    })
    print(
        f"RESULT=PASS method={args.method} images={len(output_images)} "
        f"annotations={len(output_annotations)} output={output_root}"
    )


if __name__ == "__main__":
    main()
