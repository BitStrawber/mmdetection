#!/usr/bin/env python3
"""Apply manually curated UVOT400 sequence labels to COCO annotations.

Input mapping table:
  column A / video_sequence: Video_0001
  column B / extracted_description: fish

The script rewrites category ids according to each image's Video_xxxx sequence
and writes new COCO JSON files. Original annotation files are not overwritten
unless the output path is explicitly set to the same file.
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_ANN_FILES = [
    "/media/HDD1/XCX/exp_2/UVOT400/train/instances_train.json",
    "/media/HDD1/XCX/exp_2/UVOT400/test/instances_test.json",
    "/media/HDD1/XCX/exp_2/UVOT400/train/instances_train_bbox20pct.json",
    "/media/HDD1/XCX/exp_2/UVOT400/test/instances_test_bbox20pct.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        required=True,
        help=(
            "Manual mapping table in .xlsx or .csv format. Expected columns: "
            "video_sequence and extracted_description, or first two columns."))
    parser.add_argument(
        "--ann",
        nargs="+",
        default=DEFAULT_ANN_FILES,
        help="Input COCO annotation JSON files.")
    parser.add_argument(
        "--out",
        nargs="+",
        default=None,
        help=(
            "Output COCO JSON files. Must match --ann length. If omitted, "
            "'_resolved_categories' is appended before .json."))
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON path.")
    return parser.parse_args()


def normalize_label(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_sequence(text):
    text = str(text or "").strip()
    match = re.search(r"Video[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return "Video_{:04d}".format(int(match.group(1)))
    match = re.search(r"(\d+)$", text)
    if match:
        return "Video_{:04d}".format(int(match.group(1)))
    return text


def read_csv_mapping(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if "video_sequence" in row:
                sequence = row.get("video_sequence")
            elif "sequence" in row:
                sequence = row.get("sequence")
            else:
                sequence = row.get(fieldnames[0]) if fieldnames else ""

            if "extracted_description" in row:
                label = row.get("extracted_description")
            elif "label" in row:
                label = row.get("label")
            elif "category" in row:
                label = row.get("category")
            else:
                label = row.get(fieldnames[1]) if len(fieldnames) > 1 else ""

            rows.append((sequence, label))
    return rows


def read_xlsx_shared_strings(zf):
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(path))
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values = []
    for si in root.findall("a:si", namespace):
        texts = [node.text or "" for node in si.findall(".//a:t", namespace)]
        values.append("".join(texts))
    return values


def first_sheet_path(zf):
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheet = workbook.find("{%s}sheets/{%s}sheet" % (ns_main, ns_main))
    if sheet is None:
        return "xl/worksheets/sheet1.xml"
    rel_id = sheet.attrib.get("{%s}id" % ns_rel)
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    for rel in rels.findall("{%s}Relationship" % ns_pkg):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            if target.startswith("/"):
                return target.lstrip("/")
            return "xl/" + target
    return "xl/worksheets/sheet1.xml"


def column_index(cell_ref):
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def cell_value(cell, shared_strings, namespace):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//a:t", namespace)]
        return "".join(texts)
    value = cell.find("a:v", namespace)
    if value is None:
        return ""
    text = value.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (ValueError, IndexError):
            return ""
    return text


def read_xlsx_mapping(path):
    rows = []
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings = read_xlsx_shared_strings(zf)
        sheet_xml = zf.read(first_sheet_path(zf))
    root = ET.fromstring(sheet_xml)
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    table = []
    for row in root.findall(".//a:sheetData/a:row", namespace):
        values = []
        for cell in row.findall("a:c", namespace):
            idx = column_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_value(cell, shared_strings, namespace)
        table.append(values)

    if not table:
        return rows
    header = [str(item).strip() for item in table[0]]
    lower_header = [item.lower() for item in header]
    seq_idx = lower_header.index("video_sequence") if "video_sequence" in lower_header else 0
    if "extracted_description" in lower_header:
        label_idx = lower_header.index("extracted_description")
    elif "label" in lower_header:
        label_idx = lower_header.index("label")
    elif "category" in lower_header:
        label_idx = lower_header.index("category")
    else:
        label_idx = 1

    for row in table[1:]:
        sequence = row[seq_idx] if len(row) > seq_idx else ""
        label = row[label_idx] if len(row) > label_idx else ""
        rows.append((sequence, label))
    return rows


def load_mapping(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = read_csv_mapping(path)
    elif path.suffix.lower() == ".xlsx":
        rows = read_xlsx_mapping(path)
    else:
        raise ValueError("Unsupported mapping file: {}".format(path))

    mapping = {}
    duplicates = []
    skipped = []
    for sequence, label in rows:
        sequence = normalize_sequence(sequence)
        label = normalize_label(label)
        if not sequence or not label:
            skipped.append({"sequence": sequence, "label": label})
            continue
        old = mapping.get(sequence)
        if old and old != label:
            duplicates.append({"sequence": sequence, "old": old, "new": label})
        mapping[sequence] = label
    return mapping, duplicates, skipped


def image_sequence(image):
    sequence = image.get("sequence")
    if sequence:
        return normalize_sequence(sequence)
    file_name = str(image.get("file_name", ""))
    parts = re.split(r"[\\/]+", file_name)
    for part in parts:
        if re.match(r"^Video[_-]?\d+$", part, flags=re.IGNORECASE):
            return normalize_sequence(part)
    return ""


def default_output_path(path):
    path = Path(path)
    return str(path.with_name(path.stem + "_resolved_categories" + path.suffix))


def apply_mapping_to_coco(ann_path, out_path, mapping):
    with open(ann_path, "r", encoding="utf-8") as file:
        coco = json.load(file)

    category_names = []
    category_id_by_name = {}

    def get_category_id(name):
        if name not in category_id_by_name:
            category_id_by_name[name] = len(category_id_by_name) + 1
            category_names.append(name)
        return category_id_by_name[name]

    image_category = {}
    missing_sequences = {}
    matched_images = 0

    for image in coco.get("images", []):
        sequence = image_sequence(image)
        label = mapping.get(sequence)
        image["sequence"] = sequence
        if label:
            category_id = get_category_id(label)
            image["resolved_category"] = label
            image_category[image.get("id")] = category_id
            matched_images += 1
        else:
            missing_sequences[sequence or ""] = missing_sequences.get(sequence or "", 0) + 1

    updated_annotations = 0
    for ann in tqdm(coco.get("annotations", []), desc=Path(ann_path).name, unit="ann"):
        image_id = ann.get("image_id")
        category_id = image_category.get(image_id)
        if category_id:
            ann["category_id"] = category_id
            updated_annotations += 1

    coco["categories"] = [
        {"id": category_id_by_name[name], "name": name}
        for name in category_names
    ]
    coco.setdefault("info", {})
    coco["info"]["uvot400_manual_labels"] = {
        "source": "manual sequence label mapping",
        "mapping_sequences": len(mapping),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    return {
        "input": str(ann_path),
        "output": str(out_path),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": len(coco.get("categories", [])),
        "matched_images": matched_images,
        "updated_annotations": updated_annotations,
        "missing_sequence_images": sum(missing_sequences.values()),
        "missing_sequences": missing_sequences,
    }


def write_summary(path, summary):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    mapping, duplicates, skipped = load_mapping(args.mapping)
    if args.out and len(args.out) != len(args.ann):
        raise ValueError("--out length must match --ann length")
    outputs = args.out or [default_output_path(path) for path in args.ann]

    results = []
    for ann_path, out_path in zip(args.ann, outputs):
        ann_path = Path(ann_path)
        if not ann_path.is_file():
            print("[skip missing]", ann_path)
            continue
        result = apply_mapping_to_coco(ann_path, out_path, mapping)
        results.append(result)
        print(
            "{input} -> {output}: images={images}, anns={annotations}, "
            "cats={categories}, matched_images={matched_images}, "
            "updated_anns={updated_annotations}, missing_images={missing_sequence_images}".format(
                **result))

    summary = {
        "mapping": args.mapping,
        "mapping_sequences": len(mapping),
        "duplicate_mapping_rows": duplicates,
        "skipped_mapping_rows": skipped,
        "results": results,
    }
    write_summary(args.summary, summary)
    if args.summary:
        print("summary:", args.summary)


if __name__ == "__main__":
    main()
