#!/usr/bin/env python3
"""Build a UVOT400 sequence-to-text JSON mapping.

The script only matches text file contents to UVOT400 video sequence names. It
does not extract, normalize, or infer labels from the text.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uvot-root",
        default="/media/HDD1/XCX/exp_2/UVOT400",
        help="UVOT400 root containing train/ and test/ sequence folders.")
    parser.add_argument(
        "--text-root",
        default="/media/HDD1/XCX/exp_2/UVOT400/label_source",
        help="Root directory containing downloaded text files.")
    parser.add_argument(
        "--out",
        default="/media/HDD1/XCX/exp_2/UVOT400/annotations/uvot400_text_mapping.json",
        help="Output JSON path.")
    parser.add_argument(
        "--upload-remote",
        default=None,
        help="Optional rclone destination, e.g. syn:datasets/exp2_stats/.")
    parser.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Print the rclone command without running it.")
    return parser.parse_args()


def normalize_key(text):
    text = str(text).lower()
    text = re.sub(r"\.[^.]+$", "", text)
    text = re.sub(r"[^0-9a-z]+", "", text)
    return text


def sequence_aliases(sequence):
    aliases = {sequence}
    aliases.add(sequence.lower())
    aliases.add(sequence.replace("_", ""))
    aliases.add(sequence.replace("-", ""))

    match = re.search(r"(\d+)$", sequence)
    if match:
        number = int(match.group(1))
        aliases.add(str(number))
        aliases.add("{:03d}".format(number))
        aliases.add("{:04d}".format(number))
        aliases.add("video{}".format(number))
        aliases.add("video{:03d}".format(number))
        aliases.add("video{:04d}".format(number))
    return {normalize_key(item) for item in aliases if item}


def collect_sequences(uvot_root):
    uvot_root = Path(uvot_root)
    sequences = []
    alias_to_sequence = {}
    ambiguous_aliases = set()

    for split in ("train", "test"):
        split_dir = uvot_root / split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            sequence = seq_dir.name
            item = {
                "split": split,
                "sequence": sequence,
                "sequence_dir": str(seq_dir),
            }
            sequences.append(item)
            for alias in sequence_aliases(sequence):
                old = alias_to_sequence.get(alias)
                if old and old["sequence"] != sequence:
                    ambiguous_aliases.add(alias)
                    continue
                alias_to_sequence[alias] = item

    for alias in ambiguous_aliases:
        alias_to_sequence.pop(alias, None)
    return sequences, alias_to_sequence, sorted(ambiguous_aliases)


def text_file_aliases(path):
    path = Path(path)
    aliases = set()
    for value in (path.stem, path.name, path.parent.name):
        aliases.add(normalize_key(value))

    parts = list(path.parts)
    for idx, part in enumerate(parts):
        norm = normalize_key(part)
        if re.match(r"video\d+$", norm) or re.match(r"\d+$", norm):
            aliases.add(norm)
        if idx + 1 < len(parts):
            aliases.add(normalize_key(part + "_" + parts[idx + 1]))
    return {item for item in aliases if item}


def read_text(path):
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(errors="ignore").strip()


def collect_texts(text_root, alias_to_sequence):
    text_root = Path(text_root)
    matched = {}
    unmatched = []

    text_files = sorted(text_root.rglob("*.txt")) if text_root.is_dir() else []
    for path in tqdm(text_files, desc="match text files", unit="file"):
        aliases = text_file_aliases(path)
        candidates = []
        seen = set()
        for alias in aliases:
            seq = alias_to_sequence.get(alias)
            if seq and seq["sequence"] not in seen:
                candidates.append(seq)
                seen.add(seq["sequence"])

        text = read_text(path)
        rel_path = str(path.relative_to(text_root))
        if len(candidates) == 1:
            sequence = candidates[0]["sequence"]
            old = matched.get(sequence)
            item = {
                "text_file": str(path),
                "relative_text_file": rel_path,
                "text": text,
                "match_aliases": sorted(aliases),
            }
            if old is None:
                matched[sequence] = item
            else:
                old.setdefault("duplicate_text_files", []).append(item)
        else:
            unmatched.append({
                "text_file": str(path),
                "relative_text_file": rel_path,
                "text": text,
                "candidate_sequences": [item["sequence"] for item in candidates],
                "match_aliases": sorted(aliases),
            })
    return matched, unmatched, len(text_files)


def build_payload(uvot_root, text_root):
    sequences, alias_to_sequence, ambiguous_aliases = collect_sequences(uvot_root)
    matched_texts, unmatched_texts, text_file_count = collect_texts(
        text_root, alias_to_sequence)

    records = []
    missing = []
    for seq in sequences:
        text_item = matched_texts.get(seq["sequence"])
        record = dict(seq)
        if text_item:
            record.update(text_item)
            record["has_text"] = True
            records.append(record)
        else:
            record["has_text"] = False
            record["text"] = ""
            missing.append(record)
            records.append(record)

    return {
        "info": {
            "uvot_root": str(uvot_root),
            "text_root": str(text_root),
            "note": (
                "Raw text-to-sequence mapping only. No label extraction, "
                "normalization, or semantic processing is applied."),
        },
        "summary": {
            "sequences": len(sequences),
            "text_files": text_file_count,
            "matched_sequences": sum(1 for item in records if item["has_text"]),
            "missing_text_sequences": len(missing),
            "unmatched_text_files": len(unmatched_texts),
            "ambiguous_aliases": len(ambiguous_aliases),
        },
        "records": records,
        "missing_text_sequences": missing,
        "unmatched_text_files": unmatched_texts,
        "ambiguous_aliases": ambiguous_aliases,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def upload_with_rclone(path, remote, dry_run=False):
    cmd = ["rclone", "copy", "-P", str(path), remote]
    print("upload command:", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def main():
    args = parse_args()
    payload = build_payload(args.uvot_root, args.text_root)
    write_json(args.out, payload)

    print("output:", args.out)
    print("summary:")
    for key, value in payload["summary"].items():
        print("  {}: {}".format(key, value))

    if args.upload_remote:
        upload_with_rclone(args.out, args.upload_remote, args.dry_run_upload)


if __name__ == "__main__":
    main()
