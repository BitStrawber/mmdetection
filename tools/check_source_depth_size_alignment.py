#!/usr/bin/env python
"""Check whether source images and depth maps have matching sizes.

This is mainly used before SyreaNet physical synthesis, where source RGB and
depth maps must share the same pixel grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check source/depth size alignment for synthetic datasets."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory of source images, e.g. .../syreanet/source/train.",
    )
    parser.add_argument(
        "--depth-root",
        required=True,
        help="Root directory of depth maps, e.g. .../depthanything_v2_maps/syreanet/train.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix. The script writes .json, .csv and .txt files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick checks. 0 means all images.",
    )
    parser.add_argument(
        "--no-exif-transpose",
        action="store_true",
        help="Disable ImageOps.exif_transpose before reading size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum mismatch records embedded in the summary json/txt.",
    )
    return parser.parse_args()


def image_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_size(path: Path, apply_exif: bool) -> Tuple[int, int]:
    with Image.open(path) as image:
        if apply_exif:
            image = ImageOps.exif_transpose(image)
        return image.size


def classify_size(source_size: Tuple[int, int], depth_size: Tuple[int, int]) -> str:
    if source_size == depth_size:
        return "match"
    if source_size == (depth_size[1], depth_size[0]):
        return "swapped_hw"
    return "other_mismatch"


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    depth_root = Path(args.depth_root)
    out_prefix = Path(args.out_prefix)
    apply_exif = not args.no_exif_transpose

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"depth root not found: {depth_root}")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    sources = image_files(source_root)
    total_before_limit = len(sources)
    if args.limit > 0:
        sources = sources[:args.limit]

    counters: Counter[str] = Counter()
    mismatch_records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for source_path in tqdm(sources, desc="check source/depth sizes", unit="image"):
        rel = source_path.relative_to(source_root)
        depth_path = (depth_root / rel).with_suffix(".png")

        row: Dict[str, Any] = {
            "relative": str(rel).replace("\\", "/"),
            "source": str(source_path),
            "depth": str(depth_path),
            "source_width": "",
            "source_height": "",
            "depth_width": "",
            "depth_height": "",
            "status": "",
            "error": "",
        }

        if not depth_path.exists():
            row["status"] = "missing_depth"
            counters["missing_depth"] += 1
            mismatch_records.append(dict(row))
            csv_rows.append(row)
            continue

        try:
            source_size = read_size(source_path, apply_exif)
            depth_size = read_size(depth_path, apply_exif)
            status = classify_size(source_size, depth_size)
            row.update({
                "source_width": source_size[0],
                "source_height": source_size[1],
                "depth_width": depth_size[0],
                "depth_height": depth_size[1],
                "status": status,
            })
            counters[status] += 1
            if status != "match":
                mismatch_records.append(dict(row))
        except Exception as exc:  # noqa: BLE001
            row["status"] = "read_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            counters["read_error"] += 1
            mismatch_records.append(dict(row))

        csv_rows.append(row)

    total_checked = len(sources)
    total_problem = total_checked - counters["match"]
    summary = {
        "source_root": str(source_root),
        "depth_root": str(depth_root),
        "out_prefix": str(out_prefix),
        "apply_exif_transpose": apply_exif,
        "limit": args.limit,
        "total_before_limit": total_before_limit,
        "total_checked": total_checked,
        "match": counters["match"],
        "problem_total": total_problem,
        "missing_depth": counters["missing_depth"],
        "swapped_hw": counters["swapped_hw"],
        "other_mismatch": counters["other_mismatch"],
        "read_error": counters["read_error"],
        "status_counts": dict(counters),
        "sample_records": mismatch_records[:args.max_samples],
    }

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    txt_path = out_prefix.with_suffix(".txt")

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "relative",
            "status",
            "source_width",
            "source_height",
            "depth_width",
            "depth_height",
            "source",
            "depth",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            if row["status"] != "match":
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "Source/depth size alignment summary",
        "=" * 80,
        f"source_root:          {source_root}",
        f"depth_root:           {depth_root}",
        f"apply_exif_transpose: {apply_exif}",
        f"total_before_limit:   {total_before_limit}",
        f"total_checked:        {total_checked}",
        f"match:                {counters['match']}",
        f"problem_total:        {total_problem}",
        f"missing_depth:        {counters['missing_depth']}",
        f"swapped_hw:           {counters['swapped_hw']}",
        f"other_mismatch:       {counters['other_mismatch']}",
        f"read_error:           {counters['read_error']}",
        "",
        "Sample problem records:",
    ]
    for record in mismatch_records[:args.max_samples]:
        lines.append(json.dumps(record, ensure_ascii=False))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print(f"txt:  {txt_path}")


if __name__ == "__main__":
    main()
