#!/usr/bin/env python3
"""Build a compact category-name index from exp_2 dataset statistics.

The input is the JSON written by ``tools/count_exp2_dataset_stats.py``. The
script scans ``categories`` records, creates one entry for each new category
name, and increments a counter when the same name appears again. The text output
is formatted like:

    1 fish (16)    2 turtle (5)    3 crab

The counter is the number of category records sharing the same name, not the
number of images. Image and annotation totals are also kept in CSV/JSON outputs.
"""

from __future__ import print_function, unicode_literals

import argparse
import io
import json
import os
import re


try:
    text_type = unicode
except NameError:  # pragma: no cover
    text_type = str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="logs/exp2_dataset_stats_summary.json",
        help="Input summary JSON from count_exp2_dataset_stats.py.")
    parser.add_argument(
        "--kind",
        default="bbox20pct",
        help="Category kind to scan. Use 'all' to include every kind.")
    parser.add_argument(
        "--out-prefix",
        default="logs/exp2_category_name_index",
        help="Output prefix for .txt/.csv/.json files.")
    parser.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of entries per row in TXT output.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Merge category names after lowercasing and removing spaces/punctuation.")
    parser.add_argument(
        "--sort-by",
        choices=["first", "name", "count", "images"],
        default="first",
        help="Ordering for output entries.")
    return parser.parse_args()


def normalize_name(name):
    text = text_type(name).strip().lower()
    text = re.sub(r"[\s_\-]+", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text


def load_categories(path, kind):
    with io.open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    categories = data.get("categories", [])
    if kind != "all":
        categories = [item for item in categories if item.get("kind") == kind]
    return categories


def build_index(categories, normalize=False):
    index = {}
    order = []

    for item in categories:
        raw_name = text_type(item.get("category", "")).strip()
        if not raw_name:
            continue
        key = normalize_name(raw_name) if normalize else raw_name
        if key not in index:
            index[key] = {
                "name": raw_name,
                "key": key,
                "count": 0,
                "images": 0,
                "annotations": 0,
                "datasets": set(),
                "kinds": set(),
                "aliases": set(),
            }
            order.append(key)

        record = index[key]
        record["count"] += 1
        record["images"] += int(item.get("images") or 0)
        record["annotations"] += int(item.get("annotations") or 0)
        record["datasets"].add(text_type(item.get("dataset", "")))
        record["kinds"].add(text_type(item.get("kind", "")))
        record["aliases"].add(raw_name)

    rows = []
    for position, key in enumerate(order, start=1):
        record = index[key]
        rows.append({
            "first_index": position,
            "name": record["name"],
            "key": record["key"],
            "count": record["count"],
            "images": record["images"],
            "annotations": record["annotations"],
            "datasets": sorted(x for x in record["datasets"] if x),
            "kinds": sorted(x for x in record["kinds"] if x),
            "aliases": sorted(record["aliases"]),
        })
    return rows


def sort_rows(rows, sort_by):
    if sort_by == "first":
        return sorted(rows, key=lambda item: item["first_index"])
    if sort_by == "name":
        return sorted(rows, key=lambda item: item["name"].lower())
    if sort_by == "count":
        return sorted(rows, key=lambda item: (-item["count"], item["name"].lower()))
    if sort_by == "images":
        return sorted(rows, key=lambda item: (-item["images"], item["name"].lower()))
    return rows


def format_label(index, row):
    if row["count"] > 1:
        return "{} {} ({})".format(index, row["name"], row["count"])
    return "{} {}".format(index, row["name"])


def ensure_parent(path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)


def write_txt(path, rows, columns):
    ensure_parent(path)
    labels = [format_label(index, row) for index, row in enumerate(rows, start=1)]
    width = max([len(label) for label in labels] + [1]) + 4
    with io.open(path, "w", encoding="utf-8") as file:
        for start in range(0, len(labels), columns):
            chunk = labels[start:start + columns]
            file.write("".join(label.ljust(width) for label in chunk).rstrip())
            file.write("\n\n")


def csv_escape(value):
    text = text_type(value)
    if any(mark in text for mark in [",", "\"", "\n", "\r"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "index",
        "name",
        "count",
        "images",
        "annotations",
        "datasets",
        "kinds",
        "aliases",
    ]
    with io.open(path, "w", encoding="utf-8-sig", newline="") as file:
        file.write(",".join(fields) + "\n")
        for index, row in enumerate(rows, start=1):
            values = {
                "index": index,
                "name": row["name"],
                "count": row["count"],
                "images": row["images"],
                "annotations": row["annotations"],
                "datasets": ";".join(row["datasets"]),
                "kinds": ";".join(row["kinds"]),
                "aliases": ";".join(row["aliases"]),
            }
            file.write(",".join(csv_escape(values[field]) for field in fields) + "\n")


def write_json(path, rows):
    ensure_parent(path)
    serializable = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["index"] = index
        serializable.append(item)
    text = json.dumps(serializable, indent=2, ensure_ascii=False)
    if not isinstance(text, text_type):
        text = text.decode("utf-8")
    with io.open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    args = parse_args()
    categories = load_categories(args.input, args.kind)
    rows = build_index(categories, normalize=args.normalize)
    rows = sort_rows(rows, args.sort_by)

    txt_path = args.out_prefix + ".txt"
    csv_path = args.out_prefix + ".csv"
    json_path = args.out_prefix + ".json"

    write_txt(txt_path, rows, args.columns)
    write_csv(csv_path, rows)
    write_json(json_path, rows)

    print("input:", args.input)
    print("kind:", args.kind)
    print("category records:", len(categories))
    print("unique names:", len(rows))
    print("txt:", txt_path)
    print("csv:", csv_path)
    print("json:", json_path)


if __name__ == "__main__":
    main()
