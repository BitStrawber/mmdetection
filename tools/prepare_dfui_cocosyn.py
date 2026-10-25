#!/usr/bin/env python3
"""Prepare DFUI_COCOEasy assignments for the five synthesis models.

The existing synthesis launchers use ImageNet-style folders.  This utility
creates flat, per-method input folders and JSONL manifests while preserving
the original COCO file names and image ids.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


METHODS = ("uwnr", "syreanet", "cut", "uwdf", "watergan")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-root", required=True,
                   help="Directory containing instances_train.json and instances_val.json.")
    p.add_argument("--image-root", required=True,
                   help="COCO train2017 image root.")
    p.add_argument("--assignments", required=True,
                   help="Directory containing METHOD_SPLIT.txt files.")
    p.add_argument("--out-root", required=True,
                   help="Staging output directory.")
    p.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"),
                   default="hardlink")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_coco(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    required = {"images", "annotations", "categories"}
    missing = required - data.keys()
    if missing:
        raise SystemExit(f"{path}: missing COCO keys: {sorted(missing)}")
    return data


def read_assignment(path: Path, images: list[dict]) -> list[dict]:
    by_id = {str(item["id"]): item for item in images}
    by_name = {}
    for item in images:
        by_name[item["file_name"]] = item
        by_name[Path(item["file_name"]).name] = item

    selected = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip().split("\t", 1)[0].strip()
        if not token or token.startswith("#") or token.lower() in {"image_id", "file_name"}:
            continue
        item = by_id.get(token) or by_name.get(token)
        if item is None:
            item = by_name.get(Path(token).name)
        if item is None:
            raise SystemExit(f"{path}: cannot resolve assignment entry: {token}")
        image_id = item["id"]
        if image_id in seen:
            raise SystemExit(f"{path}: duplicate image id: {image_id}")
        seen.add(image_id)
        selected.append(item)
    return selected


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    annotation_root = Path(args.annotation_root).resolve()
    image_root = Path(args.image_root).resolve()
    assignment_root = Path(args.assignments).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_selected: dict[str, set[object]] = {"train": set(), "val": set()}
    summaries = []
    for split in ("train", "val"):
        coco_path = annotation_root / f"instances_{split}.json"
        coco = load_coco(coco_path)
        image_by_id = {item["id"]: item for item in coco["images"]}
        for method in METHODS:
            assignment_path = assignment_root / f"{method}_{split}.txt"
            if not assignment_path.is_file():
                raise SystemExit(f"missing assignment file: {assignment_path}")
            selected = read_assignment(assignment_path, coco["images"])
            overlap = all_selected[split].intersection(item["id"] for item in selected)
            if overlap:
                raise SystemExit(f"cross-method duplicate ids in {split}: {sorted(overlap)[:10]}")
            all_selected[split].update(item["id"] for item in selected)

            method_root = out_root / method / split
            source_root = method_root / "source"
            manifest_path = method_root / "manifest.jsonl"
            method_root.mkdir(parents=True, exist_ok=True)
            records = []
            for index, item in enumerate(selected):
                rel = Path(item["file_name"])
                src = image_root / rel.name
                if not src.is_file():
                    src = image_root / rel
                if not src.is_file():
                    raise SystemExit(f"missing source image for {item['file_name']}: {src}")
                dst = source_root / rel.name
                materialize(src, dst, args.copy_mode, args.overwrite)
                records.append({
                    "index": index,
                    "image_id": item["id"],
                    "file_name": item["file_name"],
                    "original_name": rel.name,
                    "source": str(src),
                    "staged_source": str(dst),
                })
            with manifest_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            selected_ids = {item["id"] for item in selected}
            selected_annotations = [a for a in coco["annotations"] if a["image_id"] in selected_ids]
            method_coco = dict(coco)
            method_coco["images"] = selected
            method_coco["annotations"] = selected_annotations
            (method_root / "instances.json").write_text(
                json.dumps(method_coco, ensure_ascii=False), encoding="utf-8")
            summaries.append({"method": method, "split": split,
                              "images": len(selected),
                              "annotations": len(selected_annotations),
                              "manifest": str(manifest_path)})

        expected = 9770 if split == "train" else 1724
        if len(all_selected[split]) != expected:
            raise SystemExit(f"{split}: assignments cover {len(all_selected[split])}, expected {expected}")

    (out_root / "prepare_summary.json").write_text(
        json.dumps({"methods": METHODS, "summaries": summaries}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    for summary in summaries:
        print(f"{summary['split']} {summary['method']} images={summary['images']} "
              f"annotations={summary['annotations']}")


if __name__ == "__main__":
    main()
