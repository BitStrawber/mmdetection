#!/usr/bin/env python3
"""Launch unmodified Facebook DINO with an index-backed ImageFolder dataset."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from dino_indexed_dataset import IndexedImageFolder


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dino-main", required=True)
    parser.add_argument("--index-manifest", required=True)
    args, dino_args = parser.parse_known_args()

    dino_main = Path(args.dino_main).expanduser().resolve()
    manifest = Path(args.index_manifest).expanduser().resolve()
    if not dino_main.is_file():
        raise SystemExit(f"Facebook DINO entry point does not exist: {dino_main}")
    if not manifest.is_file():
        raise SystemExit(f"DINO index manifest does not exist: {manifest}")

    from torchvision import datasets

    def indexed_imagefolder(root, transform=None, target_transform=None, loader=None,
                            is_valid_file=None):
        del root, is_valid_file
        return IndexedImageFolder(
            manifest_path=manifest,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
        )

    datasets.ImageFolder = indexed_imagefolder
    sys.path.insert(0, str(dino_main.parent))
    sys.argv = [str(dino_main), *dino_args]
    runpy.run_path(str(dino_main), run_name="__main__")


if __name__ == "__main__":
    main()
