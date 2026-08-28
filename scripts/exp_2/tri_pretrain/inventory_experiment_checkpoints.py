#!/usr/bin/env python3
"""Inventory downstream checkpoints before publishing an experiment bundle.

This tool deliberately does not copy, delete, or upload files.  It records the
checkpoint location and the nearby configuration/log files so that a publication
manifest can be reviewed before any artifacts are staged to Hugging Face.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CHECKPOINT_GLOB = "best*.pth"
METRIC_RE = re.compile(r"best_(?P<metric>.+?)_epoch_(?P<epoch>\d+)\.pth$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_task(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("segm", "mask", "instance-seg")):
        return "segmentation"
    if any(token in lowered for token in ("bbox", "det", "cascade", "rcnn")):
        return "detection"
    return "unknown"


def infer_experiment(parent: Path) -> str:
    text = str(parent).lower()
    tokens: list[str] = []
    tokens.append("imagenet100k" if any(x in text for x in ("100k", "control100k")) else "imagenet1k")
    if "dfui_ruod_uiis" in text:
        tokens.append("dfui_ruod_uiis_easy")
    elif "dfui_ruod" in text:
        tokens.append("dfui_ruod_easy")
    elif "dfui" in text:
        tokens.append("dfui")
    elif "realuw" in text:
        tokens.append("realuw")
    elif "synthetic" in text or "syn" in text:
        tokens.append("synthetic")
    else:
        tokens.append("imagenet")
    tokens.append("vits" if any(x in text for x in ("vits", "vit-small", "vit_small")) else "resnet50")
    return "_".join(tokens)


def latest_file(paths: Iterable[Path]) -> Path | None:
    values = list(paths)
    return max(values, key=lambda path: path.stat().st_mtime) if values else None


def nearest_config(directory: Path) -> Path | None:
    direct = sorted(directory.glob("*.py"))
    if direct:
        return direct[0]
    return latest_file(directory.rglob("*.py"))


def nearest_log(directory: Path) -> Path | None:
    return latest_file(directory.rglob("*.log"))


def config_digest(path: Path | None) -> str:
    return sha256(path) if path else ""


def checkpoint_metadata(path: Path) -> dict[str, str]:
    """Return the provenance fields embedded by MMEngine/MMDetection checkpoints."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
        cfg = meta.get("cfg", "") if isinstance(meta, dict) else ""
        cfg_text = str(cfg)
        load_from = re.search(r"(?:^|\n)load_from\s*=\s*['\"]([^'\"]+)", cfg_text)
        work_dir = re.search(r"(?:^|\n)work_dir\s*=\s*['\"]([^'\"]+)", cfg_text)
        return {
            "checkpoint_meta_load_from": load_from.group(1) if load_from else "",
            "checkpoint_meta_work_dir": work_dir.group(1) if work_dir else "",
        }
    except Exception as error:  # Inventory must still complete for legacy checkpoints.
        return {
            "checkpoint_meta_load_from": "",
            "checkpoint_meta_work_dir": "",
            "checkpoint_meta_error": f"{type(error).__name__}: {error}",
        }


def collect(
    root: Path, host: str, with_sha256: bool, with_checkpoint_metadata: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for checkpoint in sorted(root.rglob(CHECKPOINT_GLOB)):
        if not checkpoint.is_file():
            continue
        parent = checkpoint.parent
        config = nearest_config(parent)
        log = nearest_log(parent)
        match = METRIC_RE.search(checkpoint.name)
        stat = checkpoint.stat()
        row: dict[str, object] = {
                "host": host,
                "task": infer_task(checkpoint.name + " " + str(parent)),
                "experiment_key_candidate": infer_experiment(parent),
                "run_directory": str(parent),
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                "metric": match.group("metric") if match else "",
                "epoch": int(match.group("epoch")) if match else "",
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(checkpoint) if with_sha256 else "",
                "config": str(config) if config else "",
                "config_sha256": config_digest(config) if with_sha256 else "",
            "latest_log": str(log) if log else "",
        }
        if with_checkpoint_metadata:
            row.update(checkpoint_metadata(checkpoint))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Provenance label, e.g. fcp or fuping.")
    parser.add_argument("--root", action="append", required=True, type=Path, help="A work-dir root to scan. Repeatable.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--with-sha256", action="store_true", help="Hash checkpoint/config files; slower but required before upload.")
    parser.add_argument(
        "--with-checkpoint-metadata",
        action="store_true",
        help="Read embedded MMEngine metadata to verify the actual load_from provenance.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for root in args.root:
        if not root.is_dir():
            print(f"WARNING: skipped missing root: {root}")
            continue
        rows.extend(
            collect(
                root.resolve(),
                args.host,
                args.with_sha256,
                args.with_checkpoint_metadata,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "host", "task", "experiment_key_candidate", "run_directory", "checkpoint",
        "checkpoint_name", "metric", "epoch", "size_bytes", "modified_utc", "sha256",
        "config", "config_sha256", "latest_log",
    ]
    if args.with_checkpoint_metadata:
        fields.extend(
            ["checkpoint_meta_load_from", "checkpoint_meta_work_dir", "checkpoint_meta_error"]
        )
    tsv_path = args.out_dir / f"checkpoint_inventory_{args.host}.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / f"checkpoint_inventory_{args.host}.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"Checkpoint inventory: {tsv_path}")
    print(f"JSON inventory:       {json_path}")
    print(f"Records:              {len(rows)}")
    for task in ("detection", "segmentation", "unknown"):
        print(f"{task}: {sum(row['task'] == task for row in rows)}")


if __name__ == "__main__":
    main()
