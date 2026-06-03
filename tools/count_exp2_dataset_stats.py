#!/usr/bin/env python3
"""Count COCO annotation statistics for exp_2 underwater datasets.

The script intentionally scans only known annotation directories for most
datasets. This avoids walking large per-image JSON folders such as
CoralSCOP/train/jsons.
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_GROUPS = {
    "CoralSCOP": [Path("/media/HDD1/XCX/exp_2/CoralSCOP/annotations")],
    "DUO": [Path("/media/HDD1/XCX/exp_2/DUO/annotations")],
    "FathomNet": [Path("/media/HDD1/XCX/exp_2/FathomNet")],
    "MARIS": [Path("/media/HDD1/XCX/exp_2/MARIS/annotations")],
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


def progress(message):
    print(message)
    sys.stdout.flush()


def main():
    args = parse_args()
    rows = []
    totals = {}

    if not args.quiet:
        progress("Scanning exp_2 datasets except MUOT_3M...")

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
                found += 1

                if not args.quiet:
                    progress(
                        "    COCO {}: images={}, anns={}, cats={}".format(
                            kind, images, anns, cats))

        if found == 0:
            rows.append((dataset, "NO_COCO_JSON", 0, 0, 0, "-"))

    print_file_table(rows)
    print_total_table(totals)


if __name__ == "__main__":
    main()
