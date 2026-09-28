#!/usr/bin/env python3
"""Build a traceable RealUW COCO annotation set in the Excel 175-class taxonomy.

The RealUW SSL manifest is the authority for the ImageFolder samples.  Each
manifest source normally uses its original COCO annotation file, but sources
whose original annotations use tracking placeholders can be replaced with a
resolved COCO file through ``--annotation-override SOURCE=PATH``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_realuw_excel_175class_mapping import load_mapping, normalize


def parse_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override must be SOURCE=PATH, got: {value}")
        source, path = value.split("=", 1)
        source = source.strip()
        if not source:
            raise ValueError(f"Override source is empty: {value}")
        overrides[source] = Path(path).expanduser()
    return overrides


def load_manifest(path: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "dataset" not in record or "image_id" not in record:
                raise RuntimeError(f"{path}:{line_number}: dataset/image_id missing")
            records[str(record["dataset"])].append(record)
    return records


def manifest_file_name(record: dict) -> str:
    # Different RealUW manifest revisions used different names for this field.
    for key in ("image_name", "file_name", "relative_path", "image_path", "path"):
        value = record.get(key)
        if value:
            return str(value)
    raise RuntimeError(
        "Manifest record has no usable ImageFolder filename field. "
        f"Available keys: {sorted(record)}"
    )


def load_coco(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_annotation_path(source: str, records: list[dict], overrides: dict[str, Path]) -> Path:
    if source in overrides:
        return overrides[source]
    paths = {str(record["annotation_file"]) for record in records}
    if len(paths) != 1:
        raise RuntimeError(f"{source}: expected one raw annotation file, found {sorted(paths)}")
    return Path(paths.pop())


def build(args: argparse.Namespace) -> dict:
    lookup, _, definition_classes, conflicts = load_mapping(args.excel)
    if conflicts:
        raise RuntimeError(f"Excel aliases contain {len(conflicts)} conflicting targets")

    category_names = sorted(set().union(*definition_classes.values()))
    category_id = {name: index for index, name in enumerate(category_names, start=1)}
    category_status: dict[str, str] = {}
    for mapping in lookup.values():
        name, status = mapping["detection_name"], mapping["training_status"]
        previous = category_status.setdefault(name, status)
        if previous != status:
            raise RuntimeError(f"Excel category {name!r} has inconsistent statuses")

    records_by_source = load_manifest(args.manifest)
    overrides = parse_overrides(args.annotation_override)
    unknown_overrides = sorted(set(overrides) - set(records_by_source))
    if unknown_overrides:
        raise RuntimeError(f"Overrides name unknown manifest sources: {unknown_overrides}")

    images: list[dict] = []
    annotations: list[dict] = []
    image_lines: list[dict] = []
    source_rows: list[dict] = []
    image_key_to_id: dict[tuple[str, int], int] = {}
    annotation_id = 1
    unmapped = Counter()
    all_source_categories: set[str] = set()

    for source in sorted(records_by_source):
        records = records_by_source[source]
        annotation_path = source_annotation_path(source, records, overrides)
        coco = load_coco(annotation_path)
        source_images = {int(image["id"]): image for image in coco.get("images", [])}
        source_categories = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
        selected_ids = {int(record["image_id"]) for record in records}
        missing_ids = selected_ids - set(source_images)
        if missing_ids:
            raise RuntimeError(f"{source}: {len(missing_ids)} manifest image IDs absent from {annotation_path}")

        manifest_duplicates = 0
        for record in records:
            source_id = int(record["image_id"])
            key = (source, source_id)
            if key in image_key_to_id:
                manifest_duplicates += 1
                continue
            source_image = source_images[source_id]
            output_id = len(images) + 1
            image_key_to_id[key] = output_id
            file_name = manifest_file_name(record)
            image = {
                "id": output_id,
                "file_name": file_name,
                "width": source_image.get("width"),
                "height": source_image.get("height"),
                "source_dataset": source,
                "source_image_id": source_id,
                "source_annotation_file": str(annotation_path),
            }
            images.append(image)
            image_lines.append({**image, "manifest_record": record})

        source_annotations = [
            item for item in coco.get("annotations", []) if int(item["image_id"]) in selected_ids
        ]
        mapped_instances = Counter()
        status_instances = Counter()
        source_unmapped = Counter()
        source_detection_classes: set[str] = set()
        for item in source_annotations:
            source_id = int(item["image_id"])
            raw_name = source_categories.get(int(item["category_id"]), f"UNKNOWN_CATEGORY_ID_{item['category_id']}")
            all_source_categories.add(raw_name)
            mapping = lookup.get(normalize(raw_name))
            output = {
                "id": annotation_id,
                "image_id": image_key_to_id[(source, source_id)],
                "bbox": item["bbox"],
                "area": item.get("area", float(item["bbox"][2]) * float(item["bbox"][3])),
                "iscrowd": int(item.get("iscrowd", 0)),
                "source_dataset": source,
                "source_annotation_id": item.get("id"),
                "source_category_id": item["category_id"],
                "source_category_name": raw_name,
            }
            if mapping is None:
                output.update({
                    "category_id": 0,
                    "mapped_category_name": "__unmapped__",
                    "training_status": "unmapped",
                })
                unmapped[raw_name] += 1
                source_unmapped[raw_name] += 1
            else:
                name, status = mapping["detection_name"], mapping["training_status"]
                output.update({
                    "category_id": category_id[name],
                    "mapped_category_name": name,
                    "training_status": status,
                    "canonical_excel_name": mapping["canonical_original_name"],
                    "excel_row": mapping["excel_row"],
                })
                mapped_instances[name] += 1
                status_instances[status] += 1
                source_detection_classes.add(name)
            annotations.append(output)
            annotation_id += 1

        source_rows.append({
            "source": source,
            "annotation_file": str(annotation_path),
            "annotation_override": int(source in overrides),
            "manifest_records": len(records),
            "unique_images": len(selected_ids),
            "manifest_duplicate_records": manifest_duplicates,
            "bbox_images": len({int(item["image_id"]) for item in source_annotations}),
            "bbox_instances": len(source_annotations),
            "mapped_instances": sum(mapped_instances.values()),
            "yes_instances": status_instances["yes"],
            "review_instances": status_instances["review"],
            "no_instances": status_instances["no"],
            "unmapped_instances": sum(source_unmapped.values()),
            "mapped_detection_classes": len(source_detection_classes),
        })

    categories = [
        {"id": category_id[name], "name": name, "training_status": category_status[name]}
        for name in category_names
    ]
    if unmapped:
        categories.insert(0, {"id": 0, "name": "__unmapped__", "training_status": "unmapped"})

    summary = {
        "excel": str(args.excel),
        "manifest": str(args.manifest),
        "annotation_overrides": {source: str(path) for source, path in overrides.items()},
        "sources": len(source_rows),
        "manifest_records": sum(row["manifest_records"] for row in source_rows),
        "unique_images": len(images),
        "manifest_duplicate_records": sum(row["manifest_duplicate_records"] for row in source_rows),
        "bbox_instances": len(annotations),
        "excel_categories": len(category_names),
        "mapped_bbox_instances": sum(1 for item in annotations if item["training_status"] != "unmapped"),
        "yes_bbox_instances": sum(1 for item in annotations if item["training_status"] == "yes"),
        "review_bbox_instances": sum(1 for item in annotations if item["training_status"] == "review"),
        "no_bbox_instances": sum(1 for item in annotations if item["training_status"] == "no"),
        "unmapped_bbox_instances": sum(unmapped.values()),
        "unmapped_source_categories": len(unmapped),
        "source_categories_observed": len(all_source_categories),
    }
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "image_lines": image_lines,
        "source_rows": source_rows,
        "unmapped": unmapped,
        "summary": summary,
    }


def write_outputs(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {key: result[key] for key in ("images", "annotations", "categories")}
    (output / "instances_realuw_175class_all.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    for name, allowed in (("yes", {"yes"}), ("yes_review", {"yes", "review"})):
        filtered = {
            "images": result["images"],
            "annotations": [item for item in result["annotations"] if item["training_status"] in allowed],
            "categories": [item for item in result["categories"] if item["training_status"] in allowed],
        }
        (output / f"instances_realuw_175class_{name}.json").write_text(
            json.dumps(filtered, ensure_ascii=False), encoding="utf-8"
        )
    with (output / "image_to_annotation.jsonl").open("w", encoding="utf-8") as handle:
        for line in result["image_lines"]:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    with (output / "per_source.tsv").open("w", encoding="utf-8", newline="") as handle:
        rows = result["source_rows"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "unmapped_source_categories.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source_category_name", "bbox_instances"])
        writer.writerows(result["unmapped"].most_common())
    (output / "summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotation-override", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.dry_run:
        print("RESULT=DRY_RUN_OK" if not result["unmapped"] else "RESULT=DRY_RUN_WITH_UNMAPPED")
        return 0 if not result["unmapped"] else 2
    write_outputs(result, args.output)
    print(f"OUTPUT={args.output}")
    print("RESULT=PASS_FULL_175CLASS_COCO" if not result["unmapped"] else "RESULT=COMPLETE_WITH_UNMAPPED")
    return 0 if not result["unmapped"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
