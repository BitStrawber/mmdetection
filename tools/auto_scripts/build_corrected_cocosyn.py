#!/usr/bin/env python3
"""Build a corrected Cocosyn pool using method-specific geometry rules."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


METHODS = ("uwnr", "syreanet", "cut", "uwdf", "watergan")
MODES = {
    "uwnr": "restore_original",
    "syreanet": "identity",
    "cut": "restore_original",
    "uwdf": "identity",
    "watergan": "center_crop_640x480",
}
RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syn-root", required=True,
                        help="Current generated Cocosyn root.")
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--uwnr-root", default="")
    parser.add_argument("--syreanet-root", default="")
    parser.add_argument("--cut-root", default="")
    parser.add_argument("--uwdf-root", default="")
    parser.add_argument("--watergan-root", default="")
    parser.add_argument("--limit-per-method", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_one_rle(rle):
    counts = rle["counts"]
    if isinstance(counts, (list, tuple, np.ndarray)):
        height, width = map(int, rle["size"])
        flat = np.zeros(height * width, dtype=np.uint8)
        position = 0
        value = 0
        for count in counts:
            count = int(count)
            if count < 0 or position + count > flat.size:
                raise ValueError(f"invalid RLE counts: {counts[:10]}")
            if value:
                flat[position:position + count] = 1
            position += count
            value = 1 - value
        if position != flat.size:
            raise ValueError("invalid uncompressed RLE length")
        return flat.reshape((height, width), order="F")

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded


def decode_mask(segmentation, height: int, width: int):
    if isinstance(segmentation, dict):
        rles = [segmentation]
    elif isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], dict):
        rles = segmentation
    else:
        rles = mask_utils.frPyObjects(segmentation, height, width)
        if isinstance(rles, dict):
            rles = [rles]

    masks = [decode_one_rle(rle) > 0 for rle in rles]
    if not masks:
        return None
    return np.logical_or.reduce(masks).astype(np.uint8)


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def center_crop_annotation(annotation, source_w: int, source_h: int):
    mask = decode_mask(annotation["segmentation"], source_h, source_w)
    if mask is None or not mask.any():
        return None

    target_w, target_h = 640, 480
    scale = max(target_w / source_w, target_h / source_h)
    resized_w = max(target_w, int(source_w * scale + 0.5))
    resized_h = max(target_h, int(source_h * scale + 0.5))
    left = int((resized_w - target_w) * 0.5)
    top = int((resized_h - target_h) * 0.5)

    image = Image.fromarray(mask)
    image = image.resize((resized_w, resized_h), RESAMPLE_NEAREST)
    image = image.crop((left, top, left + target_w, top + target_h))
    transformed = np.asarray(image) > 0
    if not transformed.any():
        return None

    rle = encode_mask(transformed)
    result = dict(annotation)
    result["segmentation"] = rle
    result["bbox"] = [float(value) for value in mask_utils.toBbox(rle)]
    result["area"] = float(transformed.sum())
    return result


def load_manifest(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 10000:
        raise SystemExit(f"{path}: expected 10000 records, got {len(rows)}")
    return rows


def find_generated(roots: list[Path], name: str) -> Path:
    candidates = [root / name for root in roots if (root / name).is_file()]
    if len(candidates) != 1:
        raise SystemExit(
            f"generated mapping error for {name}: candidates={[str(x) for x in candidates]}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    syn_root = Path(args.syn_root).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    output_root = Path(args.output_root).resolve()

    roots = {
        "uwnr": [Path(args.uwnr_root).resolve() if args.uwnr_root
                 else syn_root / "uwnr/train_10k"],
        "syreanet": [Path(args.syreanet_root).resolve() if args.syreanet_root
                     else syn_root / "syreanet/train"],
        "cut": [Path(args.cut_root).resolve() if args.cut_root
                else syn_root / "cut/imagenet_ruod_cut_full_bs2_5epoch_gpu5/test_latest/images/fake_B"],
        "uwdf": [Path(args.uwdf_root).resolve() if args.uwdf_root
                 else syn_root / "uwdf/train"],
        "watergan": [Path(args.watergan_root).resolve() if args.watergan_root
                     else syn_root / "watergan/train_10k"],
    }

    for method, method_roots in roots.items():
        for root in method_roots:
            if not root.is_dir():
                raise SystemExit(f"missing {method} generated root: {root}")
        annotation_path = metadata_root / f"instances_{method}_train.json"
        manifest_path = metadata_root / f"{method}_generated_manifest.jsonl"
        if not annotation_path.is_file():
            raise SystemExit(f"missing annotation: {annotation_path}")
        if not manifest_path.is_file():
            raise SystemExit(f"missing manifest: {manifest_path}")

    if output_root.exists() and args.overwrite:
        import shutil
        shutil.rmtree(output_root)
    image_out = output_root / "images"
    ann_out = output_root / "annotations"
    image_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_annotations = []
    all_records = []
    categories_ref = None
    next_image_id = 1
    next_ann_id = 1
    dropped = defaultdict(int)
    reused = defaultdict(int)
    created = defaultdict(int)

    for method in METHODS:
        mode = MODES[method]
        annotation_path = metadata_root / f"instances_{method}_train.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda item: int(item["id"]))
        if categories_ref is None:
            categories_ref = categories
        elif categories != categories_ref:
            raise SystemExit(f"{method}: category mismatch")

        images_by_id = {int(image["id"]): image for image in data["images"]}
        annotations_by_image = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        records = load_manifest(metadata_root / f"{method}_generated_manifest.jsonl")
        if args.limit_per_method > 0:
            records = records[:args.limit_per_method]

        for index, record in enumerate(records, 1):
            source_id = int(record["image_id"])
            source_image = images_by_id[source_id]
            source_w = int(source_image["width"])
            source_h = int(source_image["height"])
            generated_name = Path(record["generated_file"]).name
            generated_path = find_generated(roots[method], generated_name)

            with Image.open(generated_path) as opened:
                generated_size = opened.size
                restored_image = (
                    opened.convert("RGB") if mode == "restore_original" else None
                )

            if mode == "restore_original":
                if generated_size != (256, 256):
                    raise SystemExit(
                        f"{method}: expected 256x256, got {generated_size}"
                    )
                final_name = f"{method}__{Path(record['source_file']).stem}.jpg"
                final_path = image_out / final_name
                if not final_path.is_file():
                    restored_image.resize(
                        (source_w, source_h), RESAMPLE_BICUBIC
                    ).save(final_path, quality=95, subsampling=0)
                    created[method] += 1
                else:
                    reused[method] += 1
                transformed = [
                    dict(annotation)
                    for annotation in annotations_by_image[source_id]
                ]
                final_w, final_h = source_w, source_h

            elif mode == "identity":
                if generated_size != (source_w, source_h):
                    raise SystemExit(
                        f"{method}: identity size mismatch "
                        f"generated={generated_size}, source={(source_w, source_h)}"
                    )
                final_name = (
                    f"{method}__{Path(record['source_file']).stem}"
                    f"{generated_path.suffix.lower()}"
                )
                final_path = image_out / final_name
                if not final_path.is_file():
                    os.symlink(generated_path, final_path)
                    created[method] += 1
                else:
                    reused[method] += 1
                transformed = [
                    dict(annotation)
                    for annotation in annotations_by_image[source_id]
                ]
                final_w, final_h = source_w, source_h

            else:
                if generated_size != (640, 480):
                    raise SystemExit(
                        f"watergan: expected 640x480, got {generated_size}"
                    )
                transformed = []
                image_annotations = annotations_by_image[source_id]
                for annotation in image_annotations:
                    item = center_crop_annotation(annotation, source_w, source_h)
                    if item is not None:
                        transformed.append(item)
                if not transformed:
                    dropped[method] += 1
                    continue

                final_name = (
                    f"{method}__{Path(record['source_file']).stem}"
                    f"{generated_path.suffix.lower()}"
                )
                final_path = image_out / final_name
                if not final_path.is_file():
                    os.symlink(generated_path, final_path)
                    created[method] += 1
                else:
                    reused[method] += 1
                final_w, final_h = 640, 480

            image_id = next_image_id
            next_image_id += 1
            image = dict(source_image)
            image.update({
                "id": image_id,
                "file_name": final_name,
                "width": final_w,
                "height": final_h,
            })
            all_images.append(image)

            for annotation in transformed:
                annotation["id"] = next_ann_id
                annotation["image_id"] = image_id
                next_ann_id += 1
                all_annotations.append(annotation)

            all_records.append({
                "method": method,
                "mode": mode,
                "new_image_id": image_id,
                "source_file": record["source_file"],
                "final_file": final_name,
                "generated_source": str(generated_path),
            })
            if index % 1000 == 0:
                print(f"{method}: processed={index}/{len(records)}", flush=True)

    result = {
        "info": {"description": "COCOSyn50k corrected geometry"},
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories_ref,
    }
    (ann_out / "instances.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_records),
        encoding="utf-8",
    )
    summary = {
        "images": len(all_images),
        "annotations": len(all_annotations),
        "categories": len(categories_ref),
        "modes": MODES,
        "dropped_empty_images": dict(dropped),
        "created_images": dict(created),
        "reused_images": dict(reused),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"RESULT=PASS_COCOSYN_POOL output={output_root}")


if __name__ == "__main__":
    main()
