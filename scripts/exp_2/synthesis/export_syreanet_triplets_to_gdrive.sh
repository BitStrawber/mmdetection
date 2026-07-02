#!/usr/bin/env bash
set -euo pipefail

# Randomly sample SyreaNet generated outputs and export triplets:
#   source | depth(ref) | generated
#
# Usage:
#   bash scripts/exp_2/synthesis/export_syreanet_triplets_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/syreanet/source}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/syreanet}"
GENERATED_ROOT="${GENERATED_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis_official/generated}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/syreanet_synthesis_triplets_random100}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
SPLITS="${SPLITS:-train val}"
NUM_PER_SPLIT="${NUM_PER_SPLIT:-100}"
SEED="${SEED:-2026}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Export SyreaNet synthesis triplets"
echo "========================================="
echo "SOURCE_ROOT:   ${SOURCE_ROOT}"
echo "DEPTH_ROOT:    ${DEPTH_ROOT}"
echo "GENERATED_ROOT:${GENERATED_ROOT}"
echo "OUT_ROOT:      ${OUT_ROOT}"
echo "ARCHIVE_PATH:  ${ARCHIVE_PATH}"
echo "SPLITS:        ${SPLITS}"
echo "NUM_PER_SPLIT: ${NUM_PER_SPLIT}"
echo "SEED:          ${SEED}"
echo "UPLOAD:        ${UPLOAD}"
echo "RCLONE_DEST:   ${RCLONE_DEST}"
echo "OVERWRITE:     ${OVERWRITE}"
echo "========================================="

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Error: SOURCE_ROOT not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DEPTH_ROOT}" ]]; then
  echo "Error: DEPTH_ROOT not found: ${DEPTH_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${GENERATED_ROOT}" ]]; then
  echo "Error: GENERATED_ROOT not found: ${GENERATED_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
GENERATED_ROOT="${GENERATED_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
SPLITS="${SPLITS}" \
NUM_PER_SPLIT="${NUM_PER_SPLIT}" \
SEED="${SEED}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


source_root = Path(os.environ["SOURCE_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
generated_root = Path(os.environ["GENERATED_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])
splits = os.environ["SPLITS"].split()
num_per_split = int(os.environ["NUM_PER_SPLIT"])
seed = int(os.environ["SEED"])

suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in suffixes


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if is_image(p))


def find_source(split: str, rel: Path) -> Path | None:
    base = source_root / split / rel.with_suffix("")
    for suffix in [".JPEG", ".jpeg", ".jpg", ".png", ".bmp", ".webp"]:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    parent = source_root / split / rel.parent
    if parent.exists():
        hits = sorted(parent.glob(f"{rel.stem}.*"))
        for hit in hits:
            if is_image(hit):
                return hit
    return None


def find_depth(split: str, rel: Path) -> Path | None:
    candidate = (depth_root / split / rel).with_suffix(".png")
    return candidate if candidate.exists() else None


def load_tile(path: Path, size=(360, 360), grayscale=False) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if grayscale:
            img = img.convert("L")
            img = Image.merge("RGB", (img, img, img))
        else:
            img = img.convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, (255, 255, 255))
    tile.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return tile


def label_tile(tile: Image.Image, label: str) -> Image.Image:
    label_h = 34
    out = Image.new("RGB", (tile.width, tile.height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, tile.width, label_h], fill=(245, 245, 245))
    draw.text((8, 9), label, fill=(0, 0, 0))
    out.paste(tile, (0, label_h))
    return out


def make_triplet(source: Path, depth: Path, generated: Path, out: Path) -> bool:
    try:
        tiles = [
            label_tile(load_tile(source), "source"),
            label_tile(load_tile(depth, grayscale=True), "depth/ref"),
            label_tile(load_tile(generated), "syreanet generated"),
        ]
    except Exception:
        return False

    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return True


summary = {
    "source_root": str(source_root),
    "depth_root": str(depth_root),
    "generated_root": str(generated_root),
    "out_root": str(out_root),
    "num_per_split": num_per_split,
    "seed": seed,
    "splits": {},
}

all_records = []
rng = random.Random(seed)

for split in splits:
    generated_split = generated_root / split
    source_split = source_root / split
    depth_split = depth_root / split
    out_split = out_root / split
    triplet_dir = out_split / "triplets"
    source_dir = out_split / "source"
    depth_dir = out_split / "depth_ref"
    generated_dir = out_split / "generated"
    for directory in [triplet_dir, source_dir, depth_dir, generated_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    generated_images = image_files(generated_split)
    rng.shuffle(generated_images)

    records = []
    skipped = []
    for generated in tqdm(generated_images, desc=f"export {split}", unit="image"):
        if len(records) >= num_per_split:
            break

        rel = generated.relative_to(generated_split)
        source = find_source(split, rel)
        depth = find_depth(split, rel)
        if source is None or depth is None:
            skipped.append({
                "generated": str(generated),
                "relative": str(rel).replace("\\", "/"),
                "missing_source": source is None,
                "missing_depth": depth is None,
            })
            continue

        idx = len(records) + 1
        stem = f"{idx:03d}_{rel.parent.name}_{rel.stem}"
        triplet_out = triplet_dir / f"{stem}_triplet.jpg"
        if not make_triplet(source, depth, generated, triplet_out):
            skipped.append({
                "generated": str(generated),
                "relative": str(rel).replace("\\", "/"),
                "reason": "triplet_failed",
            })
            continue

        source_out = source_dir / f"{stem}{source.suffix.lower()}"
        depth_out = depth_dir / f"{stem}{depth.suffix.lower()}"
        generated_out = generated_dir / f"{stem}{generated.suffix.lower()}"
        shutil.copy2(source, source_out)
        shutil.copy2(depth, depth_out)
        shutil.copy2(generated, generated_out)

        record = {
            "split": split,
            "index": idx,
            "relative": str(rel).replace("\\", "/"),
            "source": str(source),
            "depth_ref": str(depth),
            "generated": str(generated),
            "triplet": str(triplet_out),
            "source_export": str(source_out),
            "depth_export": str(depth_out),
            "generated_export": str(generated_out),
        }
        records.append(record)
        all_records.append(record)

    summary["splits"][split] = {
        "generated_candidates": len(generated_images),
        "exported": len(records),
        "skipped": len(skipped),
        "skipped_samples": skipped[:50],
        "triplet_dir": str(triplet_dir),
    }
    (out_split / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

(out_root / "manifest_all.json").write_text(
    json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
(out_root / "export_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "Create archive"
rm -f "${ARCHIVE_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  echo
  echo "Upload archive"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone not found. Set UPLOAD=0 to skip upload." >&2
    exit 1
  fi
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
else
  echo "Skip upload because UPLOAD=${UPLOAD}"
fi

echo
echo "Done."
echo "Export dir: ${OUT_ROOT}"
echo "Archive:    ${ARCHIVE_PATH}"
echo "Remote:     ${RCLONE_DEST}"
