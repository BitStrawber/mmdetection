#!/usr/bin/env python3
"""Audit RealUW image/annotation labels against a detection-category mapping TSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_KEYS = {"file_name", "filename", "image_path", "image_file", "filepath", "path"}
LABEL_KEYS = {"category_name", "class_name", "label_name", "object_name", "class", "label"}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9.() ]+", " ", text)
    return " ".join(text.split())


def split_aliases(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def load_mapping(path: Path):
    exact: dict[str, dict] = {}
    aliases: dict[str, list[dict]] = defaultdict(list)
    duplicate_originals: dict[str, list[str]] = defaultdict(list)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row)
            key = normalize(row["original_name"])
            if key in exact:
                duplicate_originals[key].extend([exact[key]["original_name"], row["original_name"]])
            else:
                exact[key] = row
            for alias in split_aliases(row.get("aliases", "")):
                aliases[normalize(alias)].append(row)
    alias_conflicts = {
        key: sorted({item["detection_name"] for item in values})
        for key, values in aliases.items()
        if len({item["detection_name"] for item in values}) > 1
    }
    return rows, exact, aliases, duplicate_originals, alias_conflicts


def map_label(label: str, exact: dict, aliases: dict, alias_conflicts: dict):
    key = normalize(label)
    if key in exact:
        return "mapped_exact", exact[key]
    if key in alias_conflicts:
        return "ambiguous_alias", None
    candidates = aliases.get(key, [])
    if candidates:
        return "mapped_alias", candidates[0]
    return "unmapped", None


def image_exists(root: Path, annotation_file: Path, value: str) -> bool:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.is_file()
    roots = [
        annotation_file.parent,
        annotation_file.parent.parent,
        root,
        root / "images",
        root / "imagefolder" / "train",
        root / "imagefolder" / "val",
    ]
    return any((base / candidate).is_file() for base in roots)


def audit_labels(labels, exact, aliases, alias_conflicts, status, unmapped, ambiguous, mapped_targets):
    for label in labels:
        result, target = map_label(label, exact, aliases, alias_conflicts)
        status[result] += 1
        if result == "unmapped":
            unmapped[label] += 1
        elif result == "ambiguous_alias":
            ambiguous[label] += 1
        elif target:
            mapped_targets[(target["detection_name"], target["training_eligible"])] += 1


def audit_coco(path, root, exact, aliases, alias_conflicts, totals, status, unmapped, ambiguous, mapped_targets, missing_images, invalid_boxes):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict) or not all(key in data for key in ("images", "annotations", "categories")):
        return False
    categories = {item.get("id"): str(item.get("name", "")) for item in data["categories"]}
    images = {item.get("id"): str(item.get("file_name", "")) for item in data["images"]}
    totals["coco_files"] += 1
    totals["coco_images"] += len(images)
    totals["coco_annotations"] += len(data["annotations"])
    audit_labels(categories.values(), exact, aliases, alias_conflicts, status, unmapped, ambiguous, mapped_targets)
    for image_id, file_name in images.items():
        if file_name and not image_exists(root, path, file_name):
            missing_images[(str(path), str(image_id), file_name)] += 1
    for ann in data["annotations"]:
        category = categories.get(ann.get("category_id"), f"<missing category_id={ann.get('category_id')}>")
        result, target = map_label(category, exact, aliases, alias_conflicts)
        status[f"annotation_{result}"] += 1
        if result == "unmapped":
            unmapped[category] += 1
        elif result == "ambiguous_alias":
            ambiguous[category] += 1
        elif target:
            mapped_targets[(target["detection_name"], target["training_eligible"])] += 1
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_boxes[(str(path), str(ann.get("id")), repr(bbox))] += 1
    return True


def walk_values(value, image_refs, labels, parent_key=""):
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = key.casefold()
            if key_norm in IMAGE_KEYS and isinstance(item, str):
                image_refs.append(item)
            if key_norm in LABEL_KEYS and isinstance(item, str):
                labels.append(item)
            if key_norm == "category" and isinstance(item, dict) and isinstance(item.get("name"), str):
                labels.append(item["name"])
            walk_values(item, image_refs, labels, key_norm)
    elif isinstance(value, list):
        for item in value:
            walk_values(item, image_refs, labels, parent_key)


def audit_manifest(path, root, exact, aliases, alias_conflicts, totals, status, unmapped, ambiguous, mapped_targets, missing_images):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            totals["manifest_records"] += 1
            try:
                record = json.loads(line)
            except Exception:
                totals["manifest_json_errors"] += 1
                continue
            image_refs, labels = [], []
            walk_values(record, image_refs, labels)
            totals["manifest_image_references"] += len(image_refs)
            totals["manifest_label_references"] += len(labels)
            audit_labels(labels, exact, aliases, alias_conflicts, status, unmapped, ambiguous, mapped_targets)
            for image_ref in image_refs:
                if not image_exists(root, path, image_ref):
                    missing_images[(str(path), str(line_number), image_ref)] += 1


def write_counter(path: Path, header: list[str], rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mapping-tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, exact, aliases, duplicate_originals, alias_conflicts = load_mapping(args.mapping_tsv)
    totals, status = Counter(), Counter()
    unmapped, ambiguous, mapped_targets = Counter(), Counter(), Counter()
    missing_images, invalid_boxes = Counter(), Counter()

    json_files = sorted(args.root.rglob("*.json"))
    for path in json_files:
        audit_coco(path, args.root, exact, aliases, alias_conflicts, totals, status, unmapped, ambiguous, mapped_targets, missing_images, invalid_boxes)
    manifest = args.root / "annotations" / "manifest.jsonl"
    if manifest.is_file():
        audit_manifest(manifest, args.root, exact, aliases, alias_conflicts, totals, status, unmapped, ambiguous, mapped_targets, missing_images)

    write_counter(args.output / "unmapped_labels.tsv", ["label", "occurrences"], unmapped.most_common())
    write_counter(args.output / "ambiguous_labels.tsv", ["label", "occurrences"], ambiguous.most_common())
    write_counter(args.output / "alias_conflicts.tsv", ["normalized_alias", "targets"], sorted((k, ";".join(v)) for k, v in alias_conflicts.items()))
    write_counter(args.output / "missing_images.tsv", ["source", "record", "image_reference", "occurrences"], [(*key, count) for key, count in missing_images.items()])
    write_counter(args.output / "invalid_bboxes.tsv", ["source", "annotation_id", "bbox", "occurrences"], [(*key, count) for key, count in invalid_boxes.items()])
    write_counter(args.output / "mapped_target_counts.tsv", ["detection_name", "training_status", "occurrences"], [(*key, count) for key, count in mapped_targets.most_common()])

    summary = {
        "root": str(args.root), "mapping_rows": len(rows), "normalized_originals": len(exact),
        "duplicate_original_names": len(duplicate_originals), "alias_conflicts": len(alias_conflicts),
        **totals, **status, "unmapped_unique_labels": len(unmapped), "ambiguous_unique_labels": len(ambiguous),
        "missing_image_references": sum(missing_images.values()), "invalid_bboxes": sum(invalid_boxes.values()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status.get("annotation_unmapped", 0) or status.get("annotation_ambiguous_alias", 0) or invalid_boxes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
