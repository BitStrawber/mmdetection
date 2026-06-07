#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact category-name index from exp_2 dataset statistics.

The input is the JSON written by ``tools/count_exp2_dataset_stats.py``. The
script scans ``categories`` records, creates one entry for each new category
name, and increments a counter when the same name appears again. The text output
is formatted like:

    1 fish (16)    2 turtle (5)    3 crab

The counter is the number of category records sharing the same name, not the
number of images. Image and annotation totals are also kept in CSV/JSON outputs.

The optional synonym output keeps a second, merged version. It uses conservative
normalization rules by default and can also read user-defined synonym groups.
Every alias-to-canonical merge is written to a separate merge record file.
"""

from __future__ import print_function, unicode_literals

import argparse
import csv
import io
import json
import os
import re
import sys


try:
    text_type = unicode
except NameError:  # pragma: no cover
    text_type = str

PY2 = sys.version_info[0] == 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="logs/exp2_dataset_stats_summary.json",
        help=(
            "Input summary JSON from count_exp2_dataset_stats.py, or an "
            "existing category index CSV with name/images/annotations fields."))
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Input format. Auto uses file extension.")
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
    parser.add_argument(
        "--synonym-merge",
        action="store_true",
        help="Also write a synonym/normalized merged category index.")
    parser.add_argument(
        "--synonym-file",
        default=None,
        help=(
            "Optional JSON synonym rules. Supported formats: "
            "{\"canonical\": [\"alias1\", \"alias2\"]} or "
            "[{\"canonical\": \"name\", \"aliases\": [...]}]."))
    parser.add_argument(
        "--synonym-out-prefix",
        default=None,
        help=(
            "Output prefix for synonym-merged .txt/.csv/.json files. "
            "Defaults to '<out-prefix>_synonym_merged'."))
    return parser.parse_args()


MOJIBAKE_REPLACEMENTS = (
    ("鈥檚", "'s"),
    ("鈥�", "'"),
    ("鈥", "'"),
    ("茅", "e"),
)

TRAILING_STOP_WORDS = (
    "swims", "swim", "swimming", "moves", "move", "moving",
    "rests", "resting", "crawls", "crawling", "glides", "gliding",
    "floats", "floating", "hovers", "hovering", "sits", "sitting",
    "stands", "standing", "walks", "walking", "runs", "running",
    "plays", "playing", "displays", "displaying", "drifts", "drifting",
    "feeds", "feeding", "shelters", "sheltering", "hides", "hiding",
    "nestles", "nestled", "lies", "lying", "blends", "blending",
    "peeks", "peeking", "peers", "peering", "emerges", "emerging",
    "soars", "soaring", "undulates", "undulating", "propels",
    "propelling", "explores", "exploring", "descends", "descending",
    "camouflaged", "partially", "gracefully", "upright", "under",
    "against", "among", "near", "on", "over", "above", "below",
    "through", "with", "in", "inside", "outside", "at", "along",
    "around", "beside", "between", "just", "from",
)

DEFAULT_SYNONYM_GROUPS = {
    "great white shark": [
        "Great White Shark",
        "Great white shark",
        "white shark",
        "WhiteShark",
    ],
    "cuttlefish": [
        "Cuttlefish",
        "cuttlefish",
        "giant cuttle",
        "sepia",
    ],
    "octopus": [
        "Octopus",
        "octopuses",
    ],
    "turtle": [
        "turtle",
        "turtles",
    ],
    "sea lion": [
        "sea lion",
        "sea lions",
    ],
    "jellyfish": [
        "jellyfish",
        "medusa",
    ],
}


def normalize_name(name):
    text = text_type(name).strip().lower()
    text = re.sub(r"[\s_\-]+", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text


def clean_text(name):
    text = text_type(name).strip()
    for old, new in MOJIBAKE_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n,.;:")


def strip_trailing_descriptors(name):
    text = text_type(name).strip()
    text = re.sub(r"(?i)^well[-\s]+camouflaged\s+", "", text)
    text = re.sub(r"(?i)^well[-\s]+hidden\s+", "", text)
    words = clean_text(text).split()
    if not words:
        return ""
    low_words = [word.lower().strip(".,;:") for word in words]
    stop_index = None
    for index, word in enumerate(low_words):
        if word in TRAILING_STOP_WORDS:
            stop_index = index
            break
    if stop_index is not None and stop_index > 0:
        words = words[:stop_index]
    return " ".join(words).strip()


def singularize_last_word(name):
    words = clean_text(name).split()
    if not words:
        return ""
    last = words[-1]
    low = last.lower()

    plural_map = {
        "crabs": "crab",
        "dolphins": "dolphin",
        "eels": "eel",
        "fishes": "fish",
        "lobsters": "lobster",
        "octopuses": "octopus",
        "rays": "ray",
        "seals": "seal",
        "jellyfishes": "jellyfish",
        "seahorses": "seahorse",
        "sharks": "shark",
        "squids": "squid",
        "turtles": "turtle",
        "clams": "clam",
        "submarines": "submarine",
        "cuttlefishes": "cuttlefish",
    }
    if low in plural_map:
        last = plural_map[low]
    words[-1] = last
    return " ".join(words)


def canonical_synonym_name(name):
    text = strip_trailing_descriptors(name)
    text = singularize_last_word(text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = clean_text(text)
    return text.lower()


def canonical_synonym_key(name):
    return normalize_name(canonical_synonym_name(name))


def load_synonym_groups(path):
    groups = dict(DEFAULT_SYNONYM_GROUPS)
    if not path:
        return groups
    with io.open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        for canonical, aliases in data.items():
            groups[text_type(canonical)] = [text_type(alias) for alias in aliases]
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            canonical = item.get("canonical")
            if not canonical:
                continue
            groups[text_type(canonical)] = [
                text_type(alias) for alias in item.get("aliases", [])
            ]
    return groups


def build_synonym_alias_map(path=None):
    alias_to_canonical = {}
    groups = load_synonym_groups(path)
    default_keys = set(DEFAULT_SYNONYM_GROUPS.keys())
    for canonical, aliases in groups.items():
        canonical_name = canonical_synonym_name(canonical)
        names = [canonical] + list(aliases)
        source = "builtin_synonym"
        if canonical not in default_keys:
            source = "user_synonym"
        for alias in names:
            alias_to_canonical[canonical_synonym_key(alias)] = {
                "canonical": canonical_name,
                "source": source,
            }
    return alias_to_canonical


def load_categories(path, kind):
    input_format = "json"
    if path.lower().endswith(".csv"):
        input_format = "csv"
    return load_categories_by_format(path, kind, input_format)


def split_field(value):
    text = text_type(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def load_categories_by_format(path, kind, input_format):
    if input_format == "auto":
        input_format = "csv" if path.lower().endswith(".csv") else "json"
    if input_format == "csv":
        return load_categories_from_csv(path, kind)
    return load_categories_from_json(path, kind)


def load_categories_from_json(path, kind):
    with io.open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    categories = data.get("categories", [])
    if kind != "all":
        categories = [item for item in categories if item.get("kind") == kind]
    return categories


def load_categories_from_csv(path, kind):
    categories = []
    if PY2:
        with open(path, "rb") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row = {
                    text_type(key, "utf-8-sig") if isinstance(key, str) else key:
                    text_type(value, "utf-8") if isinstance(value, str) else value
                    for key, value in row.items()
                }
                append_category_from_csv_row(categories, row, kind)
    else:
        with io.open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                append_category_from_csv_row(categories, row, kind)
    return categories


def append_category_from_csv_row(categories, row, kind):
    row_kind = row.get("kind") or row.get("kinds") or ""
    if kind != "all" and kind not in split_field(row_kind) and row_kind != kind:
        return
    categories.append({
        "dataset": row.get("dataset") or row.get("datasets") or "",
        "kind": row_kind,
        "category": row.get("category") or row.get("name") or "",
        "images": row.get("images") or 0,
        "annotations": row.get("annotations") or 0,
        "count": row.get("count") or 1,
        "aliases": row.get("aliases") or "",
    })


def add_values(target_set, value):
    values = split_field(value)
    if not values:
        values = [text_type(value or "").strip()]
    for item in values:
        if item:
            target_set.add(item)


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
        record_count = int(item.get("count") or 1)
        record["count"] += record_count
        record["images"] += int(item.get("images") or 0)
        record["annotations"] += int(item.get("annotations") or 0)
        add_values(record["datasets"], item.get("dataset", ""))
        add_values(record["kinds"], item.get("kind", ""))
        aliases = split_field(item.get("aliases", ""))
        if aliases:
            for alias in aliases:
                record["aliases"].add(alias)
        else:
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


def build_synonym_index(categories, synonym_file=None):
    alias_to_canonical = build_synonym_alias_map(synonym_file)
    index = {}
    order = []
    alias_records = {}

    for item in categories:
        raw_name = text_type(item.get("category", "")).strip()
        if not raw_name:
            continue

        normalized_name = canonical_synonym_name(raw_name)
        normalized_key = canonical_synonym_key(raw_name)
        mapped = alias_to_canonical.get(normalized_key)
        if mapped:
            canonical_name = mapped["canonical"]
            reason = mapped["source"]
        else:
            canonical_name = normalized_name
            reason = "normalized"

        key = normalize_name(canonical_name)
        if key not in index:
            index[key] = {
                "name": canonical_name,
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
        record_count = int(item.get("count") or 1)
        record["count"] += record_count
        record["images"] += int(item.get("images") or 0)
        record["annotations"] += int(item.get("annotations") or 0)
        add_values(record["datasets"], item.get("dataset", ""))
        add_values(record["kinds"], item.get("kind", ""))
        aliases = split_field(item.get("aliases", ""))
        if aliases:
            for alias in aliases:
                record["aliases"].add(alias)
        else:
            record["aliases"].add(raw_name)

        alias_key = (raw_name, canonical_name, reason)
        alias_item = alias_records.setdefault(alias_key, {
            "alias": raw_name,
            "canonical": canonical_name,
            "reason": reason,
            "count": 0,
            "images": 0,
            "annotations": 0,
            "datasets": set(),
            "kinds": set(),
        })
        alias_item["count"] += record_count
        alias_item["images"] += int(item.get("images") or 0)
        alias_item["annotations"] += int(item.get("annotations") or 0)
        add_values(alias_item["datasets"], item.get("dataset", ""))
        add_values(alias_item["kinds"], item.get("kind", ""))

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

    merge_rows = []
    for (_, _, _), item in sorted(
            alias_records.items(),
            key=lambda pair: (pair[1]["canonical"], pair[1]["alias"])):
        alias = item["alias"]
        canonical = item["canonical"]
        if normalize_name(alias) == normalize_name(canonical):
            continue
        merge_rows.append({
            "alias": alias,
            "canonical": canonical,
            "reason": item["reason"],
            "count": item["count"],
            "images": item["images"],
            "annotations": item["annotations"],
            "datasets": sorted(x for x in item["datasets"] if x),
            "kinds": sorted(x for x in item["kinds"] if x),
        })

    return rows, merge_rows


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


def write_merge_csv(path, rows):
    ensure_parent(path)
    fields = [
        "alias",
        "canonical",
        "reason",
        "count",
        "images",
        "annotations",
        "datasets",
        "kinds",
    ]
    with io.open(path, "w", encoding="utf-8-sig", newline="") as file:
        file.write(",".join(fields) + "\n")
        for row in rows:
            values = {
                "alias": row["alias"],
                "canonical": row["canonical"],
                "reason": row["reason"],
                "count": row["count"],
                "images": row["images"],
                "annotations": row["annotations"],
                "datasets": ";".join(row["datasets"]),
                "kinds": ";".join(row["kinds"]),
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


def write_merge_json(path, rows):
    ensure_parent(path)
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    if not isinstance(text, text_type):
        text = text.decode("utf-8")
    with io.open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    args = parse_args()
    categories = load_categories_by_format(
        args.input, args.kind, args.input_format)
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

    if args.synonym_merge:
        synonym_prefix = args.synonym_out_prefix
        if not synonym_prefix:
            synonym_prefix = args.out_prefix + "_synonym_merged"
        synonym_rows, merge_rows = build_synonym_index(
            categories, synonym_file=args.synonym_file)
        synonym_rows = sort_rows(synonym_rows, args.sort_by)

        synonym_txt = synonym_prefix + ".txt"
        synonym_csv = synonym_prefix + ".csv"
        synonym_json = synonym_prefix + ".json"
        merge_csv = synonym_prefix + "_merges.csv"
        merge_json = synonym_prefix + "_merges.json"

        write_txt(synonym_txt, synonym_rows, args.columns)
        write_csv(synonym_csv, synonym_rows)
        write_json(synonym_json, synonym_rows)
        write_merge_csv(merge_csv, merge_rows)
        write_merge_json(merge_json, merge_rows)

        print("synonym unique names:", len(synonym_rows))
        print("synonym merge records:", len(merge_rows))
        print("synonym txt:", synonym_txt)
        print("synonym csv:", synonym_csv)
        print("synonym json:", synonym_json)
        print("merge csv:", merge_csv)
        print("merge json:", merge_json)


if __name__ == "__main__":
    main()
