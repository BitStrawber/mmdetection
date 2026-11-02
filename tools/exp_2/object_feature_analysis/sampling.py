from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .common import (
    ensure_empty_or_create,
    existing_file,
    file_sha256,
    read_json,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a class-balanced, scale-stratified RUOD instance manifest."
    )
    parser.add_argument("--config", required=True, help="Analysis JSON config.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _loader_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    sources = config.get("data_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must contain a non-empty data_sources list")
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Each data source must be a JSON object")
        for field in ("split", "annotation", "image_root"):
            if not source.get(field):
                raise ValueError(f"Data source is missing {field}: {source}")
    return dict(config)


def _candidate_from_annotation(
    source: Mapping[str, Any],
    image: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    image_width = float(image["width"])
    image_height = float(image["height"])
    if width <= 0 or height <= 0:
        return {}
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    if x2 <= x1 or y2 <= y1:
        return {}
    scale = math.sqrt(((x2 - x1) * (y2 - y1)) / (image_width * image_height))
    image_root = source.get("_resolved_image_root")
    if image_root is None:
        image_root = Path(source["image_root"]).expanduser().resolve()
    image_path = Path(image_root) / image["file_name"]
    return {
        "instance_id": f"{source['split']}:{image['id']}:{annotation['id']}",
        "annotation_id": int(annotation["id"]),
        "image_id": int(image["id"]),
        "category_id": int(annotation["category_id"]),
        "split": str(source["split"]),
        "file_name": str(image["file_name"]),
        "image_path": str(image_path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "bbox_xyxy": [x1, y1, x2, y2],
        "normalized_scale": scale,
    }


def load_candidates(config: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[int, str]]:
    candidates: List[Dict[str, Any]] = []
    category_names: Dict[int, str] = {}
    category_fingerprint = None
    for source in config["data_sources"]:
        annotation_path = existing_file(source["annotation"])
        resolved_source = {
            **source,
            "_resolved_image_root": str(
                Path(source["image_root"]).expanduser().resolve()
            ),
        }
        coco = read_json(annotation_path)
        images = {int(image["id"]): image for image in coco.get("images", [])}
        categories = {
            int(category["id"]): str(category["name"])
            for category in coco.get("categories", [])
        }
        if not categories:
            raise ValueError(f"No categories found: {annotation_path}")
        fingerprint = tuple(sorted(categories.items()))
        if category_fingerprint is None:
            category_fingerprint = fingerprint
            category_names = categories
        elif fingerprint != category_fingerprint:
            raise ValueError("RUOD category definitions differ between sources")
        for annotation in coco.get("annotations", []):
            image = images.get(int(annotation["image_id"]))
            if image is None:
                continue
            candidate = _candidate_from_annotation(
                resolved_source, image, annotation
            )
            if candidate:
                candidate["annotation_file"] = str(annotation_path)
                candidates.append(candidate)
    if not candidates:
        raise ValueError("No valid annotated instances were found")
    return candidates, category_names


def add_global_size_bins(
    candidates: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[float]]:
    scales = np.asarray(
        [float(candidate["normalized_scale"]) for candidate in candidates],
        dtype=np.float64,
    )
    thresholds = np.quantile(scales, [0.25, 0.5, 0.75]).tolist()
    output = []
    for candidate in candidates:
        item = dict(candidate)
        item["size_bin"] = int(np.digitize(item["normalized_scale"], thresholds))
        output.append(item)
    return output, [float(value) for value in thresholds]


def allocate_quota(
    available: Mapping[Any, int],
    total: int,
) -> Dict[Any, int]:
    """Allocate an exact quota proportionally while respecting capacities."""
    keys = sorted(available, key=str)
    if sum(max(int(available[key]), 0) for key in keys) < total:
        raise ValueError("Not enough candidates to satisfy quota")
    counts = {key: 0 for key in keys}
    remaining = int(total)
    while remaining:
        capacities = {
            key: max(int(available[key]) - counts[key], 0) for key in keys
        }
        capacity_total = sum(capacities.values())
        if capacity_total <= 0:
            raise ValueError("Quota allocator exhausted all candidates")
        raw = {
            key: remaining * capacities[key] / float(capacity_total)
            for key in keys
        }
        additions = {
            key: min(int(math.floor(raw[key])), capacities[key]) for key in keys
        }
        allocated = sum(additions.values())
        if allocated == 0:
            key = max(keys, key=lambda value: (raw[value], str(value)))
            additions[key] = 1
            allocated = 1
        for key in keys:
            counts[key] += additions[key]
        remaining -= allocated
    return counts


def select_balanced_instances(
    candidates: Sequence[Dict[str, Any]],
    samples_per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped: Dict[int, Dict[tuple[str, int], List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        category = int(candidate["category_id"])
        stratum = (str(candidate["split"]), int(candidate["size_bin"]))
        grouped[category][stratum].append(candidate)

    selected: List[Dict[str, Any]] = []
    for category in sorted(grouped):
        strata = grouped[category]
        for values in strata.values():
            values.sort(key=lambda row: str(row["instance_id"]))
        available = {key: len(values) for key, values in strata.items()}
        quota = allocate_quota(available, samples_per_class)
        for key in sorted(strata, key=str):
            values = strata[key]
            chosen = rng.sample(values, quota[key])
            selected.extend(chosen)
    selected.sort(
        key=lambda row: (
            int(row["image_id"]),
            int(row["category_id"]),
            str(row["instance_id"]),
        )
    )
    return selected


def summarize_manifest(
    rows: Sequence[Dict[str, Any]],
    category_names: Mapping[int, str],
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    category_counts = Counter(int(row["category_id"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    size_counts = Counter(int(row["size_bin"]) for row in rows)
    image_id_values = len({int(row["image_id"]) for row in rows})
    split_image_groups = len(
        {(str(row["split"]), int(row["image_id"])) for row in rows}
    )
    return {
        "instances": len(rows),
        "images": split_image_groups,
        "image_id_values": image_id_values,
        "split_image_groups": split_image_groups,
        "category_counts": {
            str(category): category_counts[category] for category in sorted(category_counts)
        },
        "category_names": {
            str(category): category_names[category]
            for category in sorted(category_names)
        },
        "split_counts": dict(sorted(split_counts.items())),
        "size_bin_counts": {
            str(size): size_counts[size] for size in sorted(size_counts)
        },
        "size_bin_thresholds": list(thresholds),
    }


def main() -> None:
    args = parse_args()
    config = _loader_config(read_json(args.config))
    candidates, category_names = load_candidates(config)
    candidates, thresholds = add_global_size_bins(candidates)
    selected = select_balanced_instances(
        candidates,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    manifest_path = out_dir / "instances.jsonl"
    write_jsonl(manifest_path, selected)
    summary = summarize_manifest(selected, category_names, thresholds)
    summary.update(
        {
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "manifest_sha256": file_sha256(manifest_path),
            "candidate_instances": len(candidates),
        }
    )
    write_json(out_dir / "sampling_summary.json", summary)
    write_json(
        out_dir / "categories.json",
        [
            {"id": category, "name": category_names[category]}
            for category in sorted(category_names)
        ],
    )
    print(f"Instances: {len(selected)}")
    print(f"Images: {summary['images']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
