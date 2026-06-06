#!/usr/bin/env python3
"""Count COCO annotation statistics for exp_2 underwater datasets.

The script intentionally scans only known annotation directories for most
datasets. This avoids walking large per-image JSON folders such as
CoralSCOP/train/jsons.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_GROUPS = {
    "CoralSCOP": [Path("/media/HDD1/XCX/exp_2/CoralSCOP/annotations")],
    "DUO": [Path("/media/HDD1/XCX/exp_2/DUO/annotations")],
    "FathomNet": [Path("/media/HDD1/XCX/exp_2/FathomNet")],
    "MARIS": [Path("/media/HDD1/XCX/exp_2/MARIS/annotations")],
    "MUOT_3M": [Path("/media/HDD1/XCX/exp_2/MUOT_3M/annotations")],
    "UOT100": [Path("/media/HDD1/XCX/exp_2/UOT100/annotations")],
    "USIS16K": [Path("/media/HDD1/XCX/exp_2/USIS16K/USIS16K/annotations")],
    "UVEB": [Path("/media/HDD1/XCX/exp_2/UVEB")],
    "UVOT400": [
        Path("/media/HDD1/XCX/exp_2/UVOT400/train"),
        Path("/media/HDD1/XCX/exp_2/UVOT400/test"),
    ],
    "UW-COT220": [Path("/media/HDD1/XCX/exp_2/UW-COT220/annotations")],
    "WebUOT-1M": [
        Path("/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/annotations")
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final tables, without per-file progress.")
    parser.add_argument(
        "--category-csv",
        default=None,
        help="Optional CSV path for per-category image/annotation counts.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON path for all computed statistics.")
    return parser.parse_args()


def annotation_kind(path):
    name = path.name.lower()
    if "bbox20pct" in name:
        return "bbox20pct"
    if "bbox10pct" in name:
        return "bbox10pct"
    return "raw"


def json_candidates(path):
    if not path.exists():
        return []
    if path.name in {"annotations", "train", "test"}:
        return sorted(path.glob("*.json"))
    return sorted(path.rglob("*.json"))


def load_coco(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            data = json.load(file)
    except Exception as exc:
        return None, str(exc)

    if (
        isinstance(data, dict)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
    ):
        return data, None
    return None, "not_coco"


def image_category_counts(coco, file_key):
    categories = {
        cat.get("id"): str(cat.get("name", cat.get("id")))
        for cat in coco.get("categories", [])
    }
    image_ids = {img.get("id") for img in coco.get("images", [])}
    by_category = {}

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in image_ids:
            continue
        cat_id = ann.get("category_id")
        cat_name = categories.get(cat_id, str(cat_id))
        item = by_category.setdefault(cat_name, {"images": set(), "anns": 0})
        item["images"].add((file_key, image_id))
        item["anns"] += 1

    return by_category


def add_category_counts(category_totals, dataset, kind, file_key, coco):
    for cat_name, counts in image_category_counts(coco, file_key).items():
        item = category_totals.setdefault(
            (dataset, kind, cat_name), {"images": set(), "anns": 0})
        item["images"].update(counts["images"])
        item["anns"] += counts["anns"]


def count_muot3m_source(root):
    root = Path(root)
    split_counts = []
    total = 0
    sequences = 0
    for split in ("train", "test"):
        split_dir = root / split
        split_total = 0
        split_sequences = 0
        if split_dir.is_dir():
            for gt_path in sorted(split_dir.glob("Video_*/groundtruth.txt")):
                with open(gt_path, "r", encoding="utf-8", errors="ignore") as file:
                    count = sum(1 for line in file if line.strip())
                split_total += count
                split_sequences += 1
        split_counts.append({
            "split": split,
            "sequences": split_sequences,
            "images": split_total,
            "annotations": split_total,
        })
        total += split_total
        sequences += split_sequences

    return {
        "dataset": "MUOT_3M",
        "kind": "raw",
        "images": total,
        "annotations": total,
        "categories": 1,
        "jsons": 0,
        "path": str(root / "{train,test}/Video_*/groundtruth.txt"),
        "sequences": sequences,
        "splits": split_counts,
    }


def maybe_add_muot3m_source(rows, totals, category_totals, quiet=False):
    dataset = "MUOT_3M"
    if (dataset, "raw") in totals:
        return

    source = count_muot3m_source("/media/HDD1/XCX/exp_2/MUOT_3M")
    if source["images"] <= 0:
        return

    rows.append((
        dataset,
        "raw",
        source["images"],
        source["annotations"],
        source["categories"],
        source["path"],
    ))
    totals[(dataset, "raw")] = [
        source["images"],
        source["annotations"],
        source["jsons"],
    ]
    category_totals[(dataset, "raw", "object")] = {
        "images": source["images"],
        "anns": source["annotations"],
    }
    if not quiet:
        progress(
            "MUOT_3M raw source: images={}, anns={}, sequences={}".format(
                source["images"], source["annotations"], source["sequences"]))


def print_file_table(rows):
    print("\nCOCO JSON files")
    print("=" * 150)
    print(
        "{:<14} {:<10} {:>12} {:>14} {:>6}  file".format(
            "dataset", "kind", "images", "anns", "cats"))
    print("-" * 150)
    for dataset, kind, images, anns, cats, path in rows:
        print(
            "{:<14} {:<10} {:>12} {:>14} {:>6}  {}".format(
                dataset, kind, images, anns, cats, path))


def print_total_table(totals):
    print("\nDataset totals by kind")
    print("=" * 96)
    print(
        "{:<14} {:<10} {:>6} {:>12} {:>14}".format(
            "dataset", "kind", "jsons", "images", "anns"))
    print("-" * 96)

    grand = {}
    for (dataset, kind), (images, anns, count) in sorted(totals.items()):
        print(
            "{:<14} {:<10} {:>6} {:>12} {:>14}".format(
                dataset, kind, count, images, anns))
        grand.setdefault(kind, [0, 0, 0])
        grand[kind][0] += images
        grand[kind][1] += anns
        grand[kind][2] += count

    print("-" * 96)
    for kind, (images, anns, count) in sorted(grand.items()):
        print(
            "{:<14} {:<10} {:>6} {:>12} {:>14}".format(
                "TOTAL", kind, count, images, anns))


def print_dataset_comparison_table(totals):
    datasets = sorted({dataset for dataset, _ in totals.keys()})
    print("\nDataset raw vs bbox20pct")
    print("=" * 120)
    print(
        "{:<14} {:>12} {:>14} {:>12} {:>14} {:>8}".format(
            "dataset",
            "raw_images",
            "raw_anns",
            "bbox20_imgs",
            "bbox20_anns",
            "keep%"))
    print("-" * 120)

    raw_total = 0
    raw_anns_total = 0
    bbox_total = 0
    bbox_anns_total = 0
    for dataset in datasets:
        raw_images, raw_anns, _ = totals.get((dataset, "raw"), [0, 0, 0])
        bbox_images, bbox_anns, _ = totals.get((dataset, "bbox20pct"), [0, 0, 0])
        raw_total += raw_images
        raw_anns_total += raw_anns
        bbox_total += bbox_images
        bbox_anns_total += bbox_anns
        keep = bbox_images / raw_images * 100.0 if raw_images else 0.0
        print(
            "{:<14} {:>12} {:>14} {:>12} {:>14} {:>7.2f}%".format(
                dataset,
                raw_images,
                raw_anns,
                bbox_images,
                bbox_anns,
                keep))

    print("-" * 120)
    keep = bbox_total / raw_total * 100.0 if raw_total else 0.0
    print(
        "{:<14} {:>12} {:>14} {:>12} {:>14} {:>7.2f}%".format(
            "TOTAL",
            raw_total,
            raw_anns_total,
            bbox_total,
            bbox_anns_total,
            keep))


def category_rows(category_totals):
    rows = []
    for (dataset, kind, category), counts in sorted(category_totals.items()):
        image_count = counts["images"]
        if isinstance(image_count, set):
            image_count = len(image_count)
        rows.append({
            "dataset": dataset,
            "kind": kind,
            "category": category,
            "images": image_count,
            "annotations": counts["anns"],
        })
    return rows


def print_category_table(rows):
    print("\nPer-category image counts")
    print("=" * 150)
    print(
        "{:<14} {:<10} {:<58} {:>12} {:>14}".format(
            "dataset", "kind", "category", "images", "anns"))
    print("-" * 150)
    for row in rows:
        category = row["category"]
        if len(category) > 58:
            category = category[:55] + "..."
        print(
            "{:<14} {:<10} {:<58} {:>12} {:>14}".format(
                row["dataset"],
                row["kind"],
                category,
                row["images"],
                row["annotations"]))


def write_category_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "kind", "category", "images", "annotations"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path, rows, totals, category_rows_data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": [
            {
                "dataset": dataset,
                "kind": kind,
                "images": images,
                "annotations": anns,
                "categories": cats,
                "path": path_text,
            }
            for dataset, kind, images, anns, cats, path_text in rows
        ],
        "totals": [
            {
                "dataset": dataset,
                "kind": kind,
                "images": values[0],
                "annotations": values[1],
                "jsons": values[2],
            }
            for (dataset, kind), values in sorted(totals.items())
        ],
        "categories": category_rows_data,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def progress(message):
    print(message)
    sys.stdout.flush()


def main():
    args = parse_args()
    rows = []
    totals = {}
    category_totals = {}

    if not args.quiet:
        progress("Scanning exp_2 datasets...")

    for dataset, paths in DEFAULT_GROUPS.items():
        if not args.quiet:
            progress("\n===== {} =====".format(dataset))

        found = 0
        for root in paths:
            if not args.quiet:
                progress("root: {}".format(root))

            if not root.exists():
                if not args.quiet:
                    progress("  MISSING")
                rows.append((dataset, "MISSING", 0, 0, 0, str(root)))
                continue

            for path in json_candidates(root):
                if not args.quiet:
                    progress("  checking: {}".format(path))

                data, error = load_coco(path)
                if data is None:
                    if not args.quiet:
                        progress("    skip: {}".format(error))
                    continue

                kind = annotation_kind(path)
                images = len(data["images"])
                anns = len(data["annotations"])
                cats = len(data.get("categories", []))

                rows.append((dataset, kind, images, anns, cats, str(path)))
                totals.setdefault((dataset, kind), [0, 0, 0])
                totals[(dataset, kind)][0] += images
                totals[(dataset, kind)][1] += anns
                totals[(dataset, kind)][2] += 1
                add_category_counts(
                    category_totals=category_totals,
                    dataset=dataset,
                    kind=kind,
                    file_key=str(path),
                    coco=data,
                )
                found += 1

                if not args.quiet:
                    progress(
                        "    COCO {}: images={}, anns={}, cats={}".format(
                            kind, images, anns, cats))

        if found == 0:
            rows.append((dataset, "NO_COCO_JSON", 0, 0, 0, "-"))

    maybe_add_muot3m_source(
        rows=rows,
        totals=totals,
        category_totals=category_totals,
        quiet=args.quiet,
    )

    cat_rows = category_rows(category_totals)
    print_file_table(rows)
    print_total_table(totals)
    print_dataset_comparison_table(totals)
    print_category_table(cat_rows)

    if args.category_csv:
        write_category_csv(args.category_csv, cat_rows)
        print("\nCategory CSV:", args.category_csv)
    if args.summary_json:
        write_summary_json(args.summary_json, rows, totals, cat_rows)
        print("Summary JSON:", args.summary_json)


if __name__ == "__main__":
    main()
