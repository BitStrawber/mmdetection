#!/usr/bin/env python3
"""Count per-category image and annotation statistics for COCO JSON files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann",
        nargs="+",
        required=True,
        help="One or more COCO annotation JSON files.")
    parser.add_argument(
        "--dataset",
        default="COCO",
        help="Dataset name written to the output JSON.")
    parser.add_argument(
        "--kind",
        default="custom",
        help="Annotation kind written to the output JSON.")
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output summary JSON path.")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output category CSV path.")
    return parser.parse_args()


def infer_split(path):
    text = str(path).lower()
    name = Path(path).name.lower()
    if "train" in name or "/train/" in text or "\\train\\" in text:
        return "train"
    if "val" in name or "/val/" in text or "\\val\\" in text:
        return "val"
    if "test" in name or "/test/" in text or "\\test\\" in text:
        return "test"
    return "unknown"


def load_coco(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("{} is not a JSON object".format(path))
    if not isinstance(data.get("images"), list):
        raise ValueError("{} missing COCO images list".format(path))
    if not isinstance(data.get("annotations"), list):
        raise ValueError("{} missing COCO annotations list".format(path))
    return data


def category_map(coco):
    return {
        cat.get("id"): str(cat.get("name", cat.get("id")))
        for cat in coco.get("categories", [])
    }


def count_file(path, global_stats):
    path = Path(path)
    split = infer_split(path)
    coco = load_coco(path)
    cats = category_map(coco)
    image_ids = {img.get("id") for img in coco.get("images", [])}

    file_stats = defaultdict(lambda: {
        "images": set(),
        "annotations": 0,
        "splits": set(),
        "files": set(),
    })
    invalid_annotations = 0

    for ann in tqdm(coco.get("annotations", []), desc=path.name, unit="ann"):
        image_id = ann.get("image_id")
        if image_id not in image_ids:
            invalid_annotations += 1
            continue
        category = cats.get(ann.get("category_id"), str(ann.get("category_id")))
        image_key = "{}::{}".format(path, image_id)
        for target in (file_stats, global_stats):
            item = target[category]
            item["images"].add(image_key)
            item["annotations"] += 1
            item["splits"].add(split)
            item["files"].add(str(path))

    return {
        "path": str(path),
        "split": split,
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": len(coco.get("categories", [])),
        "invalid_annotations": invalid_annotations,
        "category_count": len(file_stats),
    }


def serialize_category_stats(stats):
    rows = []
    for category, item in stats.items():
        rows.append({
            "category": category,
            "images": len(item["images"]),
            "annotations": item["annotations"],
            "splits": sorted(item["splits"]),
            "files": sorted(item["files"]),
        })
    return sorted(rows, key=lambda row: (-row["images"], row["category"]))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "images", "annotations", "splits", "files"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "category": row["category"],
                "images": row["images"],
                "annotations": row["annotations"],
                "splits": ";".join(row["splits"]),
                "files": ";".join(row["files"]),
            })


def main():
    args = parse_args()
    global_stats = defaultdict(lambda: {
        "images": set(),
        "annotations": 0,
        "splits": set(),
        "files": set(),
    })
    files = []
    total_images = 0
    total_annotations = 0
    total_invalid_annotations = 0

    for ann_path in args.ann:
        info = count_file(ann_path, global_stats)
        files.append(info)
        total_images += info["images"]
        total_annotations += info["annotations"]
        total_invalid_annotations += info["invalid_annotations"]

    categories = serialize_category_stats(global_stats)
    payload = {
        "dataset": args.dataset,
        "kind": args.kind,
        "files": files,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "total_invalid_annotations": total_invalid_annotations,
        "category_count": len(categories),
        "categories": categories,
    }
    write_json(args.out_json, payload)
    if args.out_csv:
        write_csv(args.out_csv, categories)

    print("dataset:", args.dataset)
    print("kind:", args.kind)
    print("files:", len(files))
    print("total_images:", total_images)
    print("total_annotations:", total_annotations)
    print("category_count:", len(categories))
    print("out_json:", args.out_json)
    if args.out_csv:
        print("out_csv:", args.out_csv)
    print("top categories:")
    for row in categories[:20]:
        print(
            "  {category}: images={images}, anns={annotations}, splits={splits}".format(
                category=row["category"],
                images=row["images"],
                annotations=row["annotations"],
                splits=";".join(row["splits"])))


if __name__ == "__main__":
    main()
