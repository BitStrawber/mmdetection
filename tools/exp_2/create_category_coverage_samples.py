#!/usr/bin/env python3
"""Create reproducible category-coverage image samples for four data sources.

ImageFolder sources select one random image from each of as many distinct
classes as requested. COCO sources first select images that cover every
observable category, then fill the remaining quota randomly without
replacement. Images are symlinked by default so that making visual samples
does not duplicate the underlying datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"Image root not found: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def read_coco(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"COCO annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def imagefolder_samples(root: Path, count: int, rng: random.Random) -> tuple[list[dict], int]:
    class_to_paths: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ]
        if paths:
            class_to_paths[class_dir.name] = sorted(paths)

    available_classes = sorted(class_to_paths)
    if len(available_classes) < count:
        raise RuntimeError(
            f"{root}: only {len(available_classes)} non-empty classes; "
            f"cannot take {count} category-distinct images"
        )

    selected_classes = rng.sample(available_classes, count)
    return [
        {
            "source_path": str(rng.choice(class_to_paths[class_name])),
            "categories": [class_name],
            "selection_reason": "one_random_image_from_one_distinct_class",
        }
        for class_name in selected_classes
    ], len(available_classes)


def unique_basename_index(root: Path) -> dict[str, Path]:
    paths_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in image_files(root):
        paths_by_name[path.name].append(path)

    ambiguous = {
        name: paths for name, paths in paths_by_name.items() if len(paths) > 1
    }
    if ambiguous:
        name, paths = next(iter(ambiguous.items()))
        raise RuntimeError(
            f"{root}: ambiguous basename {name!r}: "
            f"{[str(path) for path in paths[:5]]}"
        )
    return {name: paths[0] for name, paths in paths_by_name.items()}


def coco_samples(
    coco_path: Path,
    image_root: Path,
    count: int,
    rng: random.Random,
) -> tuple[list[dict], int]:
    coco = read_coco(coco_path)
    paths_by_name = unique_basename_index(image_root)
    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }

    image_path_by_id: dict[int, Path] = {}
    for image in coco.get("images", []):
        image_id = int(image["id"])
        path = paths_by_name.get(Path(str(image["file_name"])).name)
        if path is not None:
            image_path_by_id[image_id] = path

    labels_by_image: dict[int, set[int]] = defaultdict(set)
    image_ids_by_category: dict[int, list[int]] = defaultdict(list)

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id in image_path_by_id:
            labels_by_image[image_id].add(category_id)

    for image_id, labels in labels_by_image.items():
        for category_id in labels:
            image_ids_by_category[category_id].append(image_id)

    observed_categories = sorted(image_ids_by_category)
    if not observed_categories:
        raise RuntimeError(f"{coco_path}: no annotated images matched {image_root}")
    if len(labels_by_image) < count:
        raise RuntimeError(
            f"{coco_path}: only {len(labels_by_image)} matched annotated images; "
            f"cannot sample {count}"
        )

    # Cover every category when the requested quota permits it. Each selected
    # image may cover multiple categories, which leaves more room for random fill.
    category_order = observed_categories[:]
    rng.shuffle(category_order)
    coverage_categories = category_order[:count]
    selected_ids: set[int] = set()
    coverage_ids: set[int] = set()

    for category_id in coverage_categories:
        image_id = rng.choice(image_ids_by_category[category_id])
        selected_ids.add(image_id)
        coverage_ids.add(image_id)

    candidates = sorted(labels_by_image)
    remaining = [image_id for image_id in candidates if image_id not in selected_ids]
    required = count - len(selected_ids)
    if required > 0:
        selected_ids.update(rng.sample(remaining, required))

    if len(selected_ids) != count:
        raise RuntimeError(f"Expected {count} selected images, found {len(selected_ids)}")

    samples = []
    for image_id in sorted(selected_ids):
        samples.append({
            "source_path": str(image_path_by_id[image_id]),
            "categories": [
                category_names[category_id]
                for category_id in sorted(labels_by_image[image_id])
            ],
            "selection_reason": (
                "category_coverage" if image_id in coverage_ids else "random_fill"
            ),
        })
    return samples, len(observed_categories)


def materialize(
    source: str,
    samples: list[dict],
    output_root: Path,
    storage_mode: str,
) -> dict:
    source_root = output_root / source
    image_root = source_root / "images"
    image_root.mkdir(parents=True, exist_ok=False)
    category_counts: Counter[str] = Counter()

    with (source_root / "samples.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=(
                "source", "sample_index", "sample_path", "source_path",
                "categories", "selection_reason",
            ),
        )
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            source_path = Path(sample["source_path"])
            target_path = image_root / f"{index:04d}__{source_path.name}"
            if storage_mode == "symlink":
                target_path.symlink_to(source_path)
            else:
                shutil.copy2(source_path, target_path)
            for category in sample["categories"]:
                category_counts[category] += 1
            writer.writerow({
                "source": source,
                "sample_index": index,
                "sample_path": str(target_path),
                "source_path": str(source_path),
                "categories": "; ".join(sample["categories"]),
                "selection_reason": sample["selection_reason"],
            })

    with (source_root / "category_coverage.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("category", "sampled_images_containing_category"))
        writer.writerows(sorted(category_counts.items(), key=lambda item: (-item[1], item[0])))

    return {
        "source": source,
        "sampled_images": len(samples),
        "covered_categories": len(category_counts),
        "samples": str(source_root / "samples.tsv"),
        "coverage": str(source_root / "category_coverage.tsv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", required=True, type=Path)
    parser.add_argument("--synthetic5-root", required=True, type=Path)
    parser.add_argument("--realuw-root", required=True, type=Path)
    parser.add_argument("--realuw-coco", required=True, type=Path)
    parser.add_argument("--ruod-root", required=True, type=Path)
    parser.add_argument("--ruod-coco", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count-per-source", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--storage-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    if args.count_per_source <= 0:
        raise RuntimeError("--count-per-source must be positive")
    if args.output.exists():
        raise RuntimeError(f"Output directory already exists: {args.output}")

    args.output.mkdir(parents=True)
    count = args.count_per_source
    sample_specs = (
        ("imagenet", imagefolder_samples, (args.imagenet_root, count, random.Random(args.seed + 101))),
        ("synthetic5", imagefolder_samples, (args.synthetic5_root, count, random.Random(args.seed + 202))),
        ("realuw", coco_samples, (args.realuw_coco, args.realuw_root, count, random.Random(args.seed + 303))),
        ("ruod", coco_samples, (args.ruod_coco, args.ruod_root, count, random.Random(args.seed + 404))),
    )

    results = []
    for source, sampler, sampler_args in sample_specs:
        samples, available_categories = sampler(*sampler_args)
        result = materialize(source, samples, args.output, args.storage_mode)
        result["available_categories"] = available_categories
        results.append(result)
        print(
            f"{source}: images={result['sampled_images']} "
            f"covered_categories={result['covered_categories']}/"
            f"{available_categories}"
        )

    with (args.output / "sampling_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    metadata = {
        "seed": args.seed,
        "count_per_source": count,
        "total_images": count * len(results),
        "storage_mode": args.storage_mode,
        "imagefolder_strategy": "one random image from each of randomly selected distinct classes",
        "coco_strategy": "category coverage first, followed by random fill without replacement",
        "results": results,
    }
    (args.output / "sampling_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output={args.output}")
    print("RESULT=PASS_CATEGORY_COVERAGE_SAMPLES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
