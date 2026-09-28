#!/usr/bin/env python3
"""Summarize nested DINO sample sizes, target instances, and class coverage.

ImageNet-1K and Synthetic5 inherit one class-level target per selected image.
RealUW is joined against a unified, mapped COCO annotation file and therefore
reports its actual bounding-box instances and mapped detection classes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SCALES = (("100k", 100_000), ("300k", 300_000), ("500k", 500_000), ("800k", 800_000), ("1m", 1_000_000))
SOURCES = ("imagenet", "realuw", "synthetic5")


def read_index(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"Index file not found: {path}")
    values = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Index file contains duplicate paths: {path}")
    return values


def selected_paths(index_dir: Path, target_count: int) -> list[str]:
    base = read_index(index_dir / "base_100k.txt")
    remaining = read_index(index_dir / "remaining_permutation.txt")
    if len(base) != 100_000:
        raise RuntimeError(f"Expected 100,000 base paths in {index_dir}, found {len(base)}")
    extra_count = target_count - len(base)
    if extra_count < 0 or len(remaining) < extra_count:
        raise RuntimeError(f"{index_dir}: insufficient remaining paths for {target_count} samples")
    values = [*base, *remaining[:extra_count]]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Base and remaining indexes overlap in {index_dir}")
    return values


def load_realuw_annotations(path: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_path: dict[str, list[dict]] = defaultdict(list)
    annotations_by_basename: dict[str, list[dict]] = defaultdict(list)

    for annotation in coco.get("annotations", []):
        image = images_by_id[int(annotation["image_id"])]
        file_name = str(image["file_name"]).replace("\\", "/")
        annotations_by_path[file_name].append(annotation)
        annotations_by_basename[Path(file_name).name].append(annotation)

    return annotations_by_path, annotations_by_basename


def resolve_realuw_annotations(
    selected: list[str],
    annotations_by_path: dict[str, list[dict]],
    annotations_by_basename: dict[str, list[dict]],
) -> tuple[list[dict], list[str]]:
    annotations: list[dict] = []
    unmatched: list[str] = []
    for path in selected:
        path = path.replace("\\", "/")
        direct = annotations_by_path.get(path)
        if direct is not None:
            annotations.extend(direct)
            continue

        candidates = annotations_by_basename.get(Path(path).name, [])
        image_ids = {item["image_id"] for item in candidates}
        if len(image_ids) == 1:
            annotations.extend(candidates)
        else:
            unmatched.append(path)
    return annotations, unmatched


def class_from_index_path(path: str) -> str:
    pieces = Path(path).parts
    if len(pieces) < 2:
        raise RuntimeError(f"Expected class-relative image path, got: {path}")
    return pieces[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-root", required=True, type=Path)
    parser.add_argument("--realuw-coco", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.realuw_coco.is_file():
        raise RuntimeError(f"RealUW unified COCO file not found: {args.realuw_coco}")

    args.output.mkdir(parents=True, exist_ok=True)
    realuw_by_path, realuw_by_basename = load_realuw_annotations(args.realuw_coco)
    rows: list[dict] = []
    selected_by_source_scale: dict[tuple[str, str], set[str]] = {}
    unmatched_by_scale: dict[str, list[str]] = {}

    for source in SOURCES:
        index_dir = args.nested_root / source / "indexes"
        previous: set[str] | None = None
        for scale, target_count in SCALES:
            selected = selected_paths(index_dir, target_count)
            selected_set = set(selected)
            selected_by_source_scale[(source, scale)] = selected_set
            nested = previous is None or previous <= selected_set
            previous = selected_set

            if source in {"imagenet", "synthetic5"}:
                classes = {class_from_index_path(path) for path in selected}
                row = {
                    "source": source,
                    "scale": scale,
                    "selected_images": len(selected),
                    "images_with_target_instances": len(selected),
                    "target_instances": len(selected),
                    "covered_classes": len(classes),
                    "yes_instances": len(selected),
                    "review_instances": 0,
                    "no_instances": 0,
                    "unmapped_instances": 0,
                    "unmatched_selected_images": 0,
                    "nested_vs_previous": "PASS" if nested else "FAIL",
                    "instance_definition": "one inherited ImageNet class target per selected image",
                }
            else:
                annotations, unmatched = resolve_realuw_annotations(selected, realuw_by_path, realuw_by_basename)
                unmatched_by_scale[scale] = unmatched
                statuses = Counter(str(item.get("training_status", "unmapped")) for item in annotations)
                classes = {
                    int(item["category_id"])
                    for item in annotations
                    if int(item["category_id"]) > 0
                }
                image_ids = {int(item["image_id"]) for item in annotations}
                row = {
                    "source": source,
                    "scale": scale,
                    "selected_images": len(selected),
                    "images_with_target_instances": len(image_ids),
                    "target_instances": len(annotations),
                    "covered_classes": len(classes),
                    "yes_instances": statuses["yes"],
                    "review_instances": statuses["review"],
                    "no_instances": statuses["no"],
                    "unmapped_instances": statuses["unmapped"],
                    "unmatched_selected_images": len(unmatched),
                    "nested_vs_previous": "PASS" if nested else "FAIL",
                    "instance_definition": "mapped RealUW detection bounding boxes",
                }
            rows.append(row)

    fields = list(rows[0])
    with (args.output / "nested_pretraining_instance_class_statistics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    for scale, unmatched in unmatched_by_scale.items():
        (args.output / f"realuw_{scale}_unmatched_selected_paths.txt").write_text(
            "\n".join(unmatched) + ("\n" if unmatched else ""), encoding="utf-8"
        )

    print("source\tscale\tselected_images\timages_with_target_instances\ttarget_instances\tcovered_classes\tyes_instances\treview_instances\tno_instances\tunmapped_instances\tunmatched_selected_images\tnested_vs_previous")
    for row in rows:
        print("\t".join(str(row[field]) for field in fields[:-1]))

    failures = [
        row for row in rows
        if row["nested_vs_previous"] != "PASS"
        or row["unmatched_selected_images"]
        or row["unmapped_instances"]
    ]
    print(f"REPORT={args.output / 'nested_pretraining_instance_class_statistics.tsv'}")
    if failures:
        print("RESULT=PARTIAL_OR_FAILED")
        return 2
    print("RESULT=PASS_NESTED_PRETRAINING_INSTANCE_CLASS_STATISTICS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
