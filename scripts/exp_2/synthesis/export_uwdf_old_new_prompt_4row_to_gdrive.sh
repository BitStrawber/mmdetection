#!/usr/bin/env bash
set -euo pipefail

# Build 4-row comparison grids for UWDF old-prompt vs new reference-linked
# prompt experiments, then package low-strength and high-strength results into
# two archives and upload them with rclone.
#
# Layout per exported grid:
#   row 1: source repeated across strength columns
#   row 2: blurred reference repeated across strength columns
#   row 3: old-prompt generated outputs
#   row 4: new-prompt generated outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
TILE_SIZE="${TILE_SIZE:-512}"
LABEL_H="${LABEL_H:-38}"

OLD_LOW_ROOT="${OLD_LOW_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_blur_ref_strength_sweep}"
OLD_HIGH_ROOT="${OLD_HIGH_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_blur_ref_high_strength_sweep}"
NEW_LOW_ROOT="${NEW_LOW_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_real_underwater_ref_linked_prompt_s020_s040}"
NEW_HIGH_ROOT="${NEW_HIGH_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_real_underwater_ref_linked_prompt_s050_s075}"

OUT_BASE="${OUT_BASE:-/media/HDD1/XCX/exp_2/uwdf_old_new_prompt_4row_compare}"
LOW_OUT="${LOW_OUT:-${OUT_BASE}/low_strength_s020_s040}"
HIGH_OUT="${HIGH_OUT:-${OUT_BASE}/high_strength_s050_s075}"
LOW_ARCHIVE="${LOW_ARCHIVE:-/media/HDD1/XCX/exp_2/uwdf_old_new_prompt_4row_compare_low_strength_s020_s040.tar.gz}"
HIGH_ARCHIVE="${HIGH_ARCHIVE:-/media/HDD1/XCX/exp_2/uwdf_old_new_prompt_4row_compare_high_strength_s050_s075.tar.gz}"

UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Export UWDF old/new prompt 4-row grids"
echo "========================================="
echo "NUM:          ${NUM}"
echo "SEED:         ${SEED}"
echo "TILE_SIZE:    ${TILE_SIZE}"
echo "OLD_LOW:      ${OLD_LOW_ROOT}"
echo "OLD_HIGH:     ${OLD_HIGH_ROOT}"
echo "NEW_LOW:      ${NEW_LOW_ROOT}"
echo "NEW_HIGH:     ${NEW_HIGH_ROOT}"
echo "LOW_OUT:      ${LOW_OUT}"
echo "HIGH_OUT:     ${HIGH_OUT}"
echo "LOW_ARCHIVE:  ${LOW_ARCHIVE}"
echo "HIGH_ARCHIVE: ${HIGH_ARCHIVE}"
echo "UPLOAD:       ${UPLOAD}"
echo "RCLONE_DEST:  ${RCLONE_DEST}"
echo "OVERWRITE:    ${OVERWRITE}"
echo "========================================="

for path in "${OLD_LOW_ROOT}" "${OLD_HIGH_ROOT}" "${NEW_LOW_ROOT}" "${NEW_HIGH_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Error: experiment root not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${LOW_OUT}" "${HIGH_OUT}" "${LOW_ARCHIVE}" "${HIGH_ARCHIVE}"
fi

OLD_LOW_ROOT="${OLD_LOW_ROOT}" \
OLD_HIGH_ROOT="${OLD_HIGH_ROOT}" \
NEW_LOW_ROOT="${NEW_LOW_ROOT}" \
NEW_HIGH_ROOT="${NEW_HIGH_ROOT}" \
LOW_OUT="${LOW_OUT}" \
HIGH_OUT="${HIGH_OUT}" \
LOW_ARCHIVE="${LOW_ARCHIVE}" \
HIGH_ARCHIVE="${HIGH_ARCHIVE}" \
NUM="${NUM}" \
SEED="${SEED}" \
TILE_SIZE="${TILE_SIZE}" \
LABEL_H="${LABEL_H}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

old_low_root = Path(os.environ["OLD_LOW_ROOT"])
old_high_root = Path(os.environ["OLD_HIGH_ROOT"])
new_low_root = Path(os.environ["NEW_LOW_ROOT"])
new_high_root = Path(os.environ["NEW_HIGH_ROOT"])
low_out = Path(os.environ["LOW_OUT"])
high_out = Path(os.environ["HIGH_OUT"])
low_archive = Path(os.environ["LOW_ARCHIVE"])
high_archive = Path(os.environ["HIGH_ARCHIVE"])
num_images = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
tile_size = int(os.environ["TILE_SIZE"])
label_h = int(os.environ["LABEL_H"])

low_exps = [
    ("s020", "e1_blurref_s020"),
    ("s025", "e2_blurref_s025"),
    ("s030", "e3_blurref_s030"),
    ("s035", "e4_blurref_s035"),
    ("s040", "e5_blurref_s040"),
]
high_exps = [
    ("s050", "e1_blurref_s050"),
    ("s055", "e2_blurref_s055"),
    ("s060", "e3_blurref_s060"),
    ("s070", "e4_blurref_s070"),
    ("s075", "e5_blurref_s075"),
]
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_img(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in suffixes


def find_generated_dir(exp_root: Path) -> Path | None:
    candidates = [exp_root / "generated", exp_root / "images", exp_root / "outputs"]
    for candidate in candidates:
        if candidate.exists() and any(is_img(p) for p in candidate.rglob("*")):
            return candidate
    image_dirs = []
    if exp_root.exists():
        for path in exp_root.rglob("*"):
            if not path.is_dir():
                continue
            if not any(is_img(child) for child in path.iterdir()):
                continue
            name = str(path).lower()
            if any(token in name for token in ["source", "reference", "compare", "comparison"]):
                continue
            image_dirs.append(path)
    if image_dirs:
        image_dirs.sort(key=lambda item: len(str(item)))
        return image_dirs[0]
    return None


def collect_images(root: Path) -> dict[str, Path]:
    images = {}
    if not root.exists():
        return images
    for path in root.rglob("*"):
        if is_img(path):
            images[path.stem] = path
    return images


def find_selected_source(root: Path) -> dict[str, Path]:
    images = {}
    for subdir in ["selected/source/train", "selected/source", "source/train", "source"]:
        path = root / subdir
        if path.exists():
            images.update(collect_images(path))
    return images


def find_selected_ref(root: Path) -> dict[str, Path]:
    images = {}
    for subdir in [
        "selected/reference_blur/qingxi",
        "selected/reference_blur",
        "selected/reference/qingxi",
        "selected/reference",
        "reference_blur/qingxi",
        "reference/qingxi",
    ]:
        path = root / subdir
        if path.exists():
            images.update(collect_images(path))
    return images


def resize_cover(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    width, height = image.size
    scale = max(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    image = image.resize((new_width, new_height), Image.BICUBIC)
    left = max(0, (new_width - size) // 2)
    top = max(0, (new_height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def make_labeled_tile(path: Path | None, label: str) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + label_h), "white")
    if path and path.exists():
        try:
            image = resize_cover(Image.open(path), tile_size)
        except Exception:
            image = Image.new("RGB", (tile_size, tile_size), (235, 235, 235))
    else:
        image = Image.new("RGB", (tile_size, tile_size), (235, 235, 235))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile_size, label_h), fill=(25, 25, 25))
    draw.text((10, 10), label, fill=(255, 255, 255))
    return canvas


def build_grid(
    source: Path | None,
    reference: Path | None,
    old_paths: list[Path | None],
    new_paths: list[Path | None],
    labels: list[str],
    out_path: Path,
) -> None:
    cols = len(labels)
    rows = 4
    tile_h = tile_size + label_h
    grid = Image.new("RGB", (cols * tile_size, rows * tile_h), "white")
    for col, label in enumerate(labels):
        x = col * tile_size
        grid.paste(make_labeled_tile(source, f"source / {label}"), (x, 0 * tile_h))
        grid.paste(make_labeled_tile(reference, f"blur ref / {label}"), (x, 1 * tile_h))
        grid.paste(make_labeled_tile(old_paths[col], f"old prompt / {label}"), (x, 2 * tile_h))
        grid.paste(make_labeled_tile(new_paths[col], f"new prompt / {label}"), (x, 3 * tile_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, compress_level=0)


def export_group(
    name: str,
    old_root: Path,
    new_root: Path,
    experiments: list[tuple[str, str]],
    out_root: Path,
    archive: Path,
) -> None:
    print(f"===== Export {name} =====")
    print(f"old_root: {old_root}")
    print(f"new_root: {new_root}")
    print(f"out_root: {out_root}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    old_maps = []
    new_maps = []
    for tag, exp_name in experiments:
        old_exp_root = old_root / "experiments" / exp_name
        if not old_exp_root.exists():
            old_exp_root = old_root / exp_name
        new_exp_root = new_root / "experiments" / exp_name
        if not new_exp_root.exists():
            new_exp_root = new_root / exp_name
        old_generated = find_generated_dir(old_exp_root)
        new_generated = find_generated_dir(new_exp_root)
        if old_generated is None:
            raise RuntimeError(f"old generated dir not found for {exp_name} under {old_root}")
        if new_generated is None:
            raise RuntimeError(f"new generated dir not found for {exp_name} under {new_root}")
        print(f"{tag}: old={old_generated}")
        print(f"{tag}: new={new_generated}")
        old_maps.append(collect_images(old_generated))
        new_maps.append(collect_images(new_generated))

    source_map = find_selected_source(new_root) or find_selected_source(old_root)
    reference_map = find_selected_ref(new_root) or find_selected_ref(old_root)

    common = set(old_maps[0]) & set(new_maps[0])
    for mapping in old_maps[1:] + new_maps[1:]:
        common &= set(mapping)
    common = sorted(common)
    if not common:
        raise RuntimeError(f"No common generated image stems found for {name}")

    rng = random.Random(seed)
    chosen = rng.sample(common, min(num_images, len(common)))
    source_keys = sorted(source_map)
    reference_keys = sorted(reference_map)
    labels = [tag for tag, _ in experiments]
    manifest = []

    for index, stem in enumerate(chosen):
        old_paths = [mapping.get(stem) for mapping in old_maps]
        new_paths = [mapping.get(stem) for mapping in new_maps]
        source = source_map.get(stem)
        reference = reference_map.get(stem)
        if source is None and source_keys:
            source = source_map[source_keys[index % len(source_keys)]]
        if reference is None and reference_keys:
            reference = reference_map[reference_keys[index % len(reference_keys)]]
        out_path = out_root / "grids" / f"{index:03d}_{stem}.png"
        build_grid(source, reference, old_paths, new_paths, labels, out_path)
        manifest.append({
            "index": index,
            "stem": stem,
            "grid": str(out_path),
            "source": str(source) if source else "",
            "reference": str(reference) if reference else "",
            "old_outputs": [str(path) if path else "" for path in old_paths],
            "new_outputs": [str(path) if path else "" for path in new_paths],
            "strength_tags": labels,
        })

    summary = {
        "name": name,
        "old_root": str(old_root),
        "new_root": str(new_root),
        "out_root": str(out_root),
        "archive": str(archive),
        "requested": num_images,
        "available_common": len(common),
        "exported": len(chosen),
        "seed": seed,
        "strength_tags": labels,
        "layout": "4 rows: source, blur reference, old prompt outputs, new prompt outputs",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if archive.exists():
        archive.unlink()
    subprocess.run(["tar", "-czf", str(archive), "-C", str(out_root.parent), out_root.name], check=True)
    subprocess.run(["ls", "-lh", str(archive)], check=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


export_group("low_strength_s020_s040", old_low_root, new_low_root, low_exps, low_out, low_archive)
export_group("high_strength_s050_s075", old_high_root, new_high_root, high_exps, high_out, high_archive)
PY

if [[ "${UPLOAD}" == "1" ]]; then
  echo "========================================="
  echo "Upload archives to Google Drive"
  echo "========================================="
  rclone copy -P "${LOW_ARCHIVE}" "${RCLONE_DEST}"
  rclone copy -P "${HIGH_ARCHIVE}" "${RCLONE_DEST}"
fi

echo
echo "Done."
echo "Low output:    ${LOW_OUT}"
echo "High output:   ${HIGH_OUT}"
echo "Low archive:   ${LOW_ARCHIVE}"
echo "High archive:  ${HIGH_ARCHIVE}"
if [[ "${UPLOAD}" == "1" ]]; then
  echo "Remote:        ${RCLONE_DEST}"
fi
