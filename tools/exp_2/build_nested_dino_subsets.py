#!/usr/bin/env python3
"""Build compact nested indexes for controlled DINO scale experiments.

The original source ImageFolders remain the only image storage. For each
source, the builder writes one immutable 100K path list and one deterministic
permutation of the remaining paths. The 300K/500K/800K/1M manifests only refer
to prefixes of those two lists, so no images or ImageFolder trees are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
            "the immutable 100K root must contain exactly 100000 images, "
            f"found {len(base)}: {base_root}")
    missing = [relative for relative in base if not (source_root / relative).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            "the existing 100K split is not path-compatible with its full "
            f"source root; examples: {preview}")
    return base


def digest_paths(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_lines(path: Path, paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")


def write_immutable_index(path: Path, paths: list[str], name: str) -> None:
    if path.exists():
        current = path.read_text(encoding="utf-8").splitlines()
        if current != paths:
            raise RuntimeError(f"{name}: existing index differs: {path}")
        return
    write_lines(path, paths)


def build_source(name: str, source_root: Path, base_root: Path, output_root: Path,
                 seed: int, dry_run: bool) -> None:
    pool = image_paths(source_root)
    pool_set = set(pool)
    base = validate_base(source_root, base_root)
    if len(pool) < SIZES[-1][1]:
        raise RuntimeError(
            f"{name}: source has {len(pool)} images but 1M is requested: {source_root}")
    if not set(base).issubset(pool_set):
        raise RuntimeError(f"{name}: 100K split contains images absent from source pool")

    base_set = set(base)
    remaining = [relative for relative in pool if relative not in base_set]
    rng = random.Random(f"nested-dino-v2:{seed}:{name}")
    rng.shuffle(remaining)

    source_output = output_root / name
    index_root = source_output / "indexes"
    base_index = index_root / "base_100k.txt"
    remaining_index = index_root / "remaining_permutation.txt"
    if not dry_run:
        write_immutable_index(base_index, base, name)
        write_immutable_index(remaining_index, remaining, name)

    for position, (label, target_count) in enumerate(SIZES):
        extra_count = target_count - len(base)
        selected = base + remaining[:extra_count]
        subset_root = source_output / label
        metadata = subset_root / "subset_manifest.json"
        payload = {
            "schema_version": 2,
            "storage_mode": "index_only",
            "source": name,
            "source_root": str(source_root),
            "immutable_100k_root": str(base_root),
            "base_index_file": str(base_index),
            "remaining_index_file": str(remaining_index),
            "label": label,
            "image_count": target_count,
            "base_count": len(base),
            "additional_count": extra_count,
            "seed": seed,
            "nested_parent": None if position == 0 else SIZES[position - 1][0],
            "selection_sha256": digest_paths(selected),
            "source_pool_sha256": digest_paths(pool),
        }
        if metadata.exists():
            current = json.loads(metadata.read_text(encoding="utf-8"))
            if current != payload:
                raise RuntimeError(f"{name}/{label}: existing manifest differs: {metadata}")
        elif not dry_run:
            subset_root.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(
            f"{name:12s} {label:4s} images={target_count:7d} "
            f"base={len(base):7d} additional={extra_count:7d} manifest={metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_assignment,
                        metavar="NAME=IMAGEFOLDER_TRAIN_ROOT")
    parser.add_argument("--base-100k", action="append", required=True, type=parse_assignment,
                        metavar="NAME=EXISTING_100K_IMAGEFOLDER_TRAIN_ROOT")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = dict(args.source)
    bases = dict(args.base_100k)
    if set(sources) != set(bases):
        raise SystemExit("--source and --base-100k must provide exactly the same names")
    for name in sorted(sources):
        build_source(name, sources[name], bases[name], args.out_root.expanduser().resolve(),
                     args.seed, args.dry_run)


if __name__ == "__main__":
    main()
