#!/usr/bin/env python3
"""Build nested ImageFolder subsets for controlled DINO scale experiments.

The initial 100K split is treated as immutable.  Each larger split contains
every item in that split plus a deterministic sample from the remaining source
pool.  Outputs are symlink ImageFolders by default, so no image bytes are
duplicated on the pretraining host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SIZES = (("100k", 100_000), ("300k", 300_000), ("500k", 500_000),
         ("800k", 800_000), ("1m", 1_000_000))


def parse_assignment(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path).expanduser().resolve()


def image_paths(root: Path) -> list[str]:
    if not root.is_dir():
        raise RuntimeError(f"imagefolder root does not exist: {root}")
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_base(source_root: Path, base_root: Path) -> list[str]:
    base = image_paths(base_root)
    if len(base) != 100_000:
        raise RuntimeError(
            f"the immutable 100K root must contain exactly 100000 images, "
            f"found {len(base)}: {base_root}")
    missing = [relative for relative in base if not (source_root / relative).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            "the existing 100K split is not path-compatible with its full "
            f"source root; examples: {preview}")
    return base


def manifest_digest(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def materialize(source_root: Path, destination: Path, paths: list[str], mode: str) -> None:
    for relative in paths:
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if mode == "symlink" and target.is_symlink() and target.resolve() == source.resolve():
                continue
            if mode == "hardlink" and target.samefile(source):
                continue
            raise RuntimeError(f"refuse to overwrite existing output: {target}")
        if mode == "symlink":
            target.symlink_to(source)
        elif mode == "hardlink":
            os.link(source, target)
        else:
            shutil.copy2(source, target)


def build_source(name: str, source_root: Path, base_root: Path, output_root: Path,
                 seed: int, mode: str, dry_run: bool) -> None:
    pool = image_paths(source_root)
    pool_set = set(pool)
    base = validate_base(source_root, base_root)
    if len(pool) < SIZES[-1][1]:
        raise RuntimeError(
            f"{name}: source has {len(pool)} images but 1M is requested: {source_root}")
    if not set(base).issubset(pool_set):
        raise RuntimeError(f"{name}: 100K split contains images absent from source pool")

    remaining = [relative for relative in pool if relative not in set(base)]
    rng = random.Random(f"nested-dino-v1:{seed}:{name}")
    rng.shuffle(remaining)
    selected = list(base)
    previous_count = 0

    for label, target_count in SIZES:
        need = target_count - previous_count
        if need < 0:
            raise AssertionError("sizes must increase")
        if target_count == 100_000:
            selected = list(base)
        else:
            selected.extend(remaining[previous_count - 100_000:target_count - 100_000])
        if len(selected) != target_count:
            raise AssertionError(f"{name}/{label}: selection size mismatch")

        subset_root = output_root / name / label
        imagefolder = subset_root / "imagefolder" / "train"
        metadata = subset_root / "subset_manifest.json"
        payload = {
            "schema_version": 1,
            "source": name,
            "source_root": str(source_root),
            "immutable_100k_root": str(base_root),
            "label": label,
            "image_count": target_count,
            "seed": seed,
            "link_mode": mode,
            "nested_parent": None if label == "100k" else SIZES[SIZES.index((label, target_count)) - 1][0],
            "selection_sha256": manifest_digest(selected),
            "relative_paths": selected,
        }
        if metadata.exists():
            current = json.loads(metadata.read_text(encoding="utf-8"))
            if current.get("selection_sha256") != payload["selection_sha256"]:
                raise RuntimeError(f"{name}/{label}: existing manifest differs: {metadata}")
        elif not dry_run:
            subset_root.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(f"{name:12s} {label:4s} images={target_count:7d} root={imagefolder}")
        if not dry_run:
            materialize(source_root, imagefolder, selected, mode)
        previous_count = target_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_assignment,
                        metavar="NAME=IMAGEFOLDER_TRAIN_ROOT")
    parser.add_argument("--base-100k", action="append", required=True, type=parse_assignment,
                        metavar="NAME=EXISTING_100K_IMAGEFOLDER_TRAIN_ROOT")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = dict(args.source)
    bases = dict(args.base_100k)
    if set(sources) != set(bases):
        raise SystemExit("--source and --base-100k must provide exactly the same names")
    for name in sorted(sources):
        build_source(name, sources[name], bases[name], args.out_root.expanduser().resolve(),
                     args.seed, args.link_mode, args.dry_run)


if __name__ == "__main__":
    main()
