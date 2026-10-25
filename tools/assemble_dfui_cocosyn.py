#!/usr/bin/env python3
"""Validate synthesis outputs and assemble DFUI_COCOsyn."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


METHODS = ("uwnr", "syreanet", "cut", "uwdf", "watergan")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-root", required=True)
    p.add_argument("--staging-root", required=True,
                   help="Output root from prepare_dfui_cocosyn.py.")
    p.add_argument("--generated-root", required=True,
                   help="Directory containing METHOD/SPLIT generated images.")
    p.add_argument("--out-root", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def image_index(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            result.setdefault(path.name, path)
    return result


def main() -> None:
    args = parse_args()
    annotation_root = Path(args.annotation_root).resolve()
    staging_root = Path(args.staging_root).resolve()
    generated_root = Path(args.generated_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for split in ("train", "val"):
        coco = load_json(annotation_root / f"instances_{split}.json")
        final_images = out_root / "images" / split
        final_images.mkdir(parents=True, exist_ok=True)
        used = set()
        merged_records = []
        for method in METHODS:
            manifest_path = staging_root / method / split / "manifest.jsonl"
            records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            outputs = image_index(generated_root / method / split)
            method_rows = []
            for record in records:
                image_id = record["image_id"]
                if image_id in used:
                    raise SystemExit(f"duplicate image id in output assignments: {image_id}")
                source = Path(record["source"])
                generated = outputs.get(record["original_name"])
                if generated is None:
                    raise SystemExit(f"missing generated image: {method}/{split}/{record['original_name']}")
                from PIL import Image
                with Image.open(source) as src_im, Image.open(generated) as gen_im:
                    if src_im.size != gen_im.size:
                        raise SystemExit(
                            f"size mismatch {method}/{split}/{record['original_name']}: "
                            f"source={src_im.size}, generated={gen_im.size}")
                dst = final_images / record["original_name"]
                if dst.exists() and not args.overwrite:
                    raise SystemExit(f"destination exists: {dst}; use --overwrite")
                shutil.copy2(generated, dst)
                used.add(image_id)
                row = dict(record)
                row.update({"method": method, "split": split, "generated": str(generated), "final": str(dst)})
                method_rows.append(row)
                merged_records.append(row)
            method_dir = out_root / "method_annotations" / method
            method_dir.mkdir(parents=True, exist_ok=True)
            ids = {r["image_id"] for r in method_rows}
            method_coco = dict(coco)
            method_coco["images"] = [im for im in coco["images"] if im["id"] in ids]
            method_coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
            (method_dir / f"instances_{split}.json").write_text(
                json.dumps(method_coco, ensure_ascii=False), encoding="utf-8")
            result_rows.extend(method_rows)

        expected_ids = {im["id"] for im in coco["images"]}
        if used != expected_ids:
            missing = sorted(expected_ids - used)
            extra = sorted(used - expected_ids)
            raise SystemExit(f"{split}: assignment coverage mismatch missing={missing[:10]} extra={extra[:10]}")
        merged = dict(coco)
        merged["images"] = [dict(im) for im in coco["images"]]
        (out_root / "annotations").mkdir(parents=True, exist_ok=True)
        (out_root / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(merged, ensure_ascii=False), encoding="utf-8")

    with (out_root / "generation_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_root / "generation_summary.json").write_text(
        json.dumps({"methods": METHODS, "rows": len(result_rows)}, indent=2), encoding="utf-8")
    print(f"DFUI_COCOsyn assembled: {out_root}")
    print(f"generated images: {len(result_rows)}")


if __name__ == "__main__":
    main()
