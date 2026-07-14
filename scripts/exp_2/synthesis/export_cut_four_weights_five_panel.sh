#!/usr/bin/env bash
set -euo pipefail

# Build five-panel comparison figures for the CUT random20 export:
#   source | CUT 1 epoch | CUT 2 epochs | CUT 3 epochs | CUT 5 epochs
#
# Usage:
#   EXPORT_ROOT=/media/HDD1/XCX/exp_2/cut_four_weights_random20_export \
#   UPLOAD=0 bash scripts/exp_2/synthesis/export_cut_four_weights_five_panel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXPORT_ROOT="${EXPORT_ROOT:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export}"
OUT_DIR="${OUT_DIR:-${EXPORT_ROOT}/five_panel}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_DIR}.tar.gz}"
UPLOAD="${UPLOAD:-0}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/cut_four_weights_random20_export}"
TILE_SIZE="${TILE_SIZE:-320}"
LABEL_HEIGHT="${LABEL_HEIGHT:-34}"
OVERWRITE="${OVERWRITE:-1}"
MODEL_NAMES="${MODEL_NAMES:-imagenet_ruod_cut_full_bs2_1epoch_gpu2 imagenet_ruod_cut_full_bs2_2epoch_gpu3 imagenet_ruod_cut_full_bs2_3epoch_gpu4 imagenet_ruod_cut_full_bs2_5epoch_gpu5}"
MODEL_LABELS="${MODEL_LABELS:-CUT 1epoch|CUT 2epoch|CUT 3epoch|CUT 5epoch}"

cat <<EOF
=========================================
CUT four-weight five-panel export
=========================================
EXPORT_ROOT:  ${EXPORT_ROOT}
OUT_DIR:      ${OUT_DIR}
ARCHIVE_PATH: ${ARCHIVE_PATH}
TILE_SIZE:    ${TILE_SIZE}
UPLOAD:       ${UPLOAD}
RCLONE_DEST:  ${RCLONE_DEST}
MODEL_NAMES:  ${MODEL_NAMES}
=========================================
EOF

if [[ ! -d "${EXPORT_ROOT}" ]]; then
  echo "Error: EXPORT_ROOT not found: ${EXPORT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${EXPORT_ROOT}/selection_manifest.json" ]]; then
  echo "Error: selection_manifest.json not found under ${EXPORT_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

EXPORT_ROOT="${EXPORT_ROOT}" \
OUT_DIR="${OUT_DIR}" \
TILE_SIZE="${TILE_SIZE}" \
LABEL_HEIGHT="${LABEL_HEIGHT}" \
MODEL_NAMES="${MODEL_NAMES}" \
MODEL_LABELS="${MODEL_LABELS}" \
python - <<'PY'
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

export_root = Path(os.environ["EXPORT_ROOT"])
out_dir = Path(os.environ["OUT_DIR"])
tile_size = int(os.environ["TILE_SIZE"])
label_height = int(os.environ["LABEL_HEIGHT"])
model_names = os.environ["MODEL_NAMES"].split()
model_labels = os.environ["MODEL_LABELS"].split("|")
if len(model_labels) != len(model_names):
    model_labels = model_names

source_dir = export_root / "source"
generated_dir = export_root / "generated"
manifest_path = export_root / "selection_manifest.json"
panels_dir = out_dir / "panels"
panels_dir.mkdir(parents=True, exist_ok=True)

image_suffixes = [".png", ".jpg", ".jpeg", ".JPEG", ".JPG"]


def find_by_stem(root, stem):
    for suffix in image_suffixes:
        p = root / f"{stem}{suffix}"
        if p.exists():
            return p
    hits = sorted(root.glob(f"{stem}.*"))
    for hit in hits:
        if hit.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return hit
    return None


def load_tile(path, label):
    tile = Image.new("RGB", (tile_size, tile_size), (248, 248, 248))
    if path is None or not path.exists():
        draw = ImageDraw.Draw(tile)
        draw.text((12, tile_size // 2 - 8), "missing", fill=(180, 0, 0))
    else:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile.paste(img, ((tile_size - img.width) // 2, (tile_size - img.height) // 2))

    out = Image.new("RGB", (tile_size, tile_size + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, tile_size, label_height], fill=(242, 242, 242))
    draw.text((8, 9), label, fill=(0, 0, 0))
    out.paste(tile, (0, label_height))
    return out


def hcat(tiles):
    w = sum(t.width for t in tiles)
    h = max(t.height for t in tiles)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    return canvas


def vcat(rows):
    if not rows:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
records = manifest.get("records", [])
if not records:
    raise SystemExit("selection_manifest.json has no records")

rows = []
summary = []
for rec in records:
    idx = int(rec.get("index", len(summary)))
    test_name = rec.get("test_name")
    if not test_name:
        continue
    stem = Path(test_name).stem
    source_path = source_dir / test_name
    if not source_path.exists():
        source_path = find_by_stem(source_dir, stem)

    tiles = [load_tile(source_path, "source")]
    missing = []
    for model_name, label in zip(model_names, model_labels):
        image_path = find_by_stem(generated_dir / model_name, stem)
        if image_path is None:
            missing.append(model_name)
        tiles.append(load_tile(image_path, label))

    panel = hcat(tiles)
    panel_name = f"{idx:03d}_{stem}_five_panel.jpg"
    panel_path = panels_dir / panel_name
    panel.save(panel_path, quality=95)
    rows.append(panel)
    summary.append({
        "index": idx,
        "test_name": test_name,
        "panel": str(panel_path),
        "missing_models": missing,
    })

# A single overview grid is convenient for slide/report inspection.
grid = vcat(rows)
grid_path = out_dir / "cut_four_weights_random20_five_panel_grid.jpg"
grid.save(grid_path, quality=92)

(out_dir / "five_panel_manifest.json").write_text(
    json.dumps({
        "export_root": str(export_root),
        "tile_size": tile_size,
        "columns": ["source"] + model_labels,
        "model_names": model_names,
        "records": summary,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"panels: {len(summary)} -> {panels_dir}")
print(f"grid:   {grid_path}")
missing_total = sum(len(item["missing_models"]) for item in summary)
print(f"missing model outputs: {missing_total}")
if missing_total:
    for item in summary[:5]:
        if item["missing_models"]:
            print(f"missing index {item['index']}: {item['missing_models']}")
PY

find "${OUT_DIR}" -maxdepth 2 -type f | sort | sed 's#^#  #' | head -n 80

tar -C "$(dirname "${OUT_DIR}")" -czf "${ARCHIVE_PATH}" "$(basename "${OUT_DIR}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  command -v rclone >/dev/null 2>&1 || { echo "Error: rclone not found" >&2; exit 1; }
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
  rclone copy -P "${OUT_DIR}" "${RCLONE_DEST}/$(basename "${OUT_DIR}")/"
fi

printf '\nDone. Five-panel output: %s\n' "${OUT_DIR}"