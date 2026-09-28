#!/usr/bin/env python3
"""Audit RealUW image annotations against the Excel 175-class taxonomy.

The RealUW SSL manifest records the source COCO annotation file and image ID
for every ImageFolder image.  This utility resolves those records, maps every
source category through ``original_name`` and ``aliases`` in the Excel
``Three-level mapping`` sheet, and writes source-wise statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = ("yes", "review", "no")


def normalize(value: object) -> str:
    """Normalize source labels while preserving camel-case aliases as strings."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9.() ]+", " ", text)
    return " ".join(text.split())


def split_aliases(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def load_mapping(path: Path) -> tuple[dict[str, dict], Counter, dict[str, set[str]], list[dict]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required. Activate the detector environment before running this audit."
        ) from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if "Three-level mapping" not in workbook.sheetnames:
        raise RuntimeError("Excel workbook does not contain the 'Three-level mapping' sheet.")

    sheet = workbook["Three-level mapping"]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    columns = {name: index for index, name in enumerate(header)}
    required = {"original_name", "detection_name", "training_eligible", "aliases"}
    missing = required - columns.keys()
    if missing:
        raise RuntimeError(f"Excel sheet is missing columns: {sorted(missing)}")

    lookup: dict[str, dict] = {}
    row_counts: Counter = Counter()
    classes_by_status: dict[str, set[str]] = defaultdict(set)
    conflicts: list[dict] = []

    for row_number, row in enumerate(rows, start=2):
        original = str(row[columns["original_name"]] or "").strip()
        detection = str(row[columns["detection_name"]] or "").strip()
        status = str(row[columns["training_eligible"]] or "").strip().casefold()
        if not original or not detection or not status:
            continue

        row_counts[status] += 1
        classes_by_status[status].add(detection)
        value = {
            "detection_name": detection,
            "training_status": status,
            "canonical_original_name": original,
            "excel_row": row_number,
        }

        for candidate in [original, *split_aliases(row[columns["aliases"]])]:
            key = normalize(candidate)
            previous = lookup.get(key)
            if previous and (
                previous["detection_name"] != detection
                or previous["training_status"] != status
            ):
                conflicts.append(
                    {
                        "label": candidate,
                        "normalized_label": key,
                        "first_target": previous["detection_name"],
                        "second_target": detection,
                        "first_row": previous["excel_row"],
                        "second_row": row_number,
                    }
                )
            else:
                lookup[key] = value

    return lookup, row_counts, classes_by_status, conflicts


def load_manifest(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                result[record["dataset"]].append(record)
    return result


def audit_source(source: str, records: list[dict], lookup: dict[str, dict]) -> tuple[dict, Counter, dict[str, set[str]]]:
    annotation_files = {record["annotation_file"] for record in records}
    if len(annotation_files) != 1:
        raise RuntimeError(f"{source}: expected one annotation file, found {sorted(annotation_files)}")

    annotation_path = Path(annotation_files.pop())
    if not annotation_path.is_file():
        raise RuntimeError(f"{source}: missing annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    selected_ids = {record["image_id"] for record in records}
    source_ids = {image["id"] for image in coco.get("images", [])}
    categories = {category["id"]: category["name"] for category in coco.get("categories", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if annotation["image_id"] in selected_ids:
            annotations_by_image[annotation["image_id"]].append(annotation)

    counts: Counter = Counter()
    raw_classes: set[str] = set()
    mapped_classes: set[str] = set()
    classes_by_status: dict[str, set[str]] = defaultdict(set)
    mapped_images: set[int] = set()
    unmapped_images: set[int] = set()
    unmapped_categories: Counter = Counter()

    for image_id, annotations in annotations_by_image.items():
        for annotation in annotations:
            raw_name = str(categories.get(annotation["category_id"], f"UNKNOWN_CATEGORY_ID_{annotation['category_id']}"))
            raw_classes.add(raw_name)
            mapping = lookup.get(normalize(raw_name))
            if mapping is None:
                counts["unmapped_instances"] += 1
                unmapped_categories[raw_name] += 1
                unmapped_images.add(image_id)
                continue

            status = mapping["training_status"]
            detection = mapping["detection_name"]
            counts["mapped_instances_all_statuses"] += 1
            counts[f"{status}_instances"] += 1
            mapped_images.add(image_id)
            mapped_classes.add(detection)
            classes_by_status[status].add(detection)

    row = {
        "source": source,
        "manifest_image_records": len(records),
        "unique_source_image_ids": len(selected_ids),
        "missing_source_image_ids": len(selected_ids - source_ids),
        "bbox_images": len(annotations_by_image),
        "bbox_instances": sum(len(items) for items in annotations_by_image.values()),
        "raw_observed_categories": len(raw_classes),
        "mapped_instances_all_statuses": counts["mapped_instances_all_statuses"],
        "mapped_images_all_statuses": len(mapped_images),
        "mapped_detection_classes_all_statuses": len(mapped_classes),
        "yes_instances": counts["yes_instances"],
        "yes_detection_classes": len(classes_by_status["yes"]),
        "review_instances": counts["review_instances"],
        "review_detection_classes": len(classes_by_status["review"]),
        "no_instances": counts["no_instances"],
        "no_detection_classes": len(classes_by_status["no"]),
        "unmapped_instances": counts["unmapped_instances"],
        "unmapped_images": len(unmapped_images),
    }
    coverage = {
        "raw": raw_classes,
        "mapped": mapped_classes,
        "yes": classes_by_status["yes"],
        "review": classes_by_status["review"],
        "no": classes_by_status["no"],
    }
    return row, unmapped_categories, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.excel.is_file():
        raise RuntimeError(f"Excel file not found: {args.excel}")
    if not args.manifest.is_file():
        raise RuntimeError(f"Manifest file not found: {args.manifest}")

    args.output.mkdir(parents=True, exist_ok=True)
    lookup, definition_rows, definition_classes, conflicts = load_mapping(args.excel)
    records_by_source = load_manifest(args.manifest)

    columns = [
        "source", "manifest_image_records", "unique_source_image_ids", "missing_source_image_ids",
        "bbox_images", "bbox_instances", "raw_observed_categories",
        "mapped_instances_all_statuses", "mapped_images_all_statuses", "mapped_detection_classes_all_statuses",
        "yes_instances", "yes_detection_classes", "review_instances", "review_detection_classes",
        "no_instances", "no_detection_classes", "unmapped_instances", "unmapped_images",
    ]
    rows = []
    all_unmapped: Counter = Counter()
    all_raw_classes: set[str] = set()
    all_mapped_classes: set[str] = set()
    all_classes_by_status: dict[str, set[str]] = defaultdict(set)

    for source in sorted(records_by_source):
        row, unmapped, coverage = audit_source(source, records_by_source[source], lookup)
        rows.append(row)
        all_unmapped.update(unmapped)
        all_raw_classes.update(coverage["raw"])
        all_mapped_classes.update(coverage["mapped"])
        for status in STATUSES:
            all_classes_by_status[status].update(coverage[status])

    total = {column: 0 for column in columns if column != "source"}
    for row in rows:
        for column in total:
            if column not in {
                "raw_observed_categories", "mapped_detection_classes_all_statuses",
                "yes_detection_classes", "review_detection_classes", "no_detection_classes",
            }:
                total[column] += row[column]

    with (args.output / "per_source.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with (args.output / "unmapped_categories.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["original_category", "bbox_instances"])
        writer.writerows(all_unmapped.most_common())

    with (args.output / "alias_conflicts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "normalized_label", "first_target", "second_target", "first_row", "second_row"], delimiter="\t")
        writer.writeheader()
        writer.writerows(conflicts)

    summary = {
        "excel": str(args.excel),
        "manifest": str(args.manifest),
        "excel_mapping_rows_by_status": dict(definition_rows),
        "excel_detection_classes_by_status": {key: len(value) for key, value in definition_classes.items()},
        "excel_total_detection_classes": len(set().union(*definition_classes.values())),
        "lookup_keys_from_originals_and_aliases": len(lookup),
        "alias_conflicts": len(conflicts),
        "sources": len(rows),
        "total_manifest_image_records": total["manifest_image_records"],
        "total_unique_source_image_ids_across_sources": total["unique_source_image_ids"],
        "total_missing_source_image_ids": total["missing_source_image_ids"],
        "total_bbox_images": total["bbox_images"],
        "total_bbox_instances": total["bbox_instances"],
        "total_raw_observed_categories": len(all_raw_classes),
        "total_mapped_instances_all_statuses": total["mapped_instances_all_statuses"],
        "total_mapped_detection_classes_all_statuses": len(all_mapped_classes),
        "total_yes_instances": total["yes_instances"],
        "total_yes_detection_classes": len(all_classes_by_status["yes"]),
        "total_review_instances": total["review_instances"],
        "total_review_detection_classes": len(all_classes_by_status["review"]),
        "total_no_instances": total["no_instances"],
        "total_no_detection_classes": len(all_classes_by_status["no"]),
        "total_unmapped_instances": total["unmapped_instances"],
        "total_unmapped_images": total["unmapped_images"],
        "unmapped_original_categories": len(all_unmapped),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"PER_SOURCE_REPORT={args.output / 'per_source.tsv'}")
    print(f"UNMAPPED_REPORT={args.output / 'unmapped_categories.tsv'}")

    if conflicts:
        print("RESULT=FAILED_ALIAS_CONFLICT")
        return 2
    if total["missing_source_image_ids"]:
        print("RESULT=FAILED_IMAGE_LINK")
        return 2
    if total["unmapped_instances"]:
        print("RESULT=PARTIAL_MAPPING")
        return 2
    print("RESULT=PASS_FULL_175CLASS_IMAGE_BBOX_MAPPING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
