#!/usr/bin/env bash
set -euo pipefail

# Export UWDF seven-condition ablation results as high-resolution per-sample grids.
# First column: source, raw reference, UWNR-style lightfield reference, depth.
# Next columns: seven generated outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_condition_linkage_seven_ablation/experiments}"
EXPERIMENTS="${EXPERIMENTS:-e1_original_stable e2_style_only e3_depth_only e4_style_depth e5_text_style_linked e6_text_depth_linked e7_text_style_depth_linked}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_condition_linkage_seven_ablation/selection_manifest.json}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_condition_linkage_seven_ablation_grid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
MAX_IMAGES="${MAX_IMAGES:-20}"
TILE_SIZE="${TILE_SIZE:-1024}"
CONDITION_TILE_SIZE="${CONDITION_TILE_SIZE:-248}"
LABEL_H="${LABEL_H:-32}"
TILE_MODE="${TILE_MODE:-cover}"
PANEL_FORMAT="${PANEL_FORMAT:-png}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-0}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

cat <<EOF
=========================================
Export UWDF condition-linkage grid
=========================================
EXP_ROOT:           ${EXP_ROOT}
EXPERIMENTS:        ${EXPERIMENTS}
SELECTION_MANIFEST: ${SELECTION_MANIFEST}
OUT_ROOT:           ${OUT_ROOT}
ARCHIVE_PATH:       ${ARCHIVE_PATH}
MAX_IMAGES:         ${MAX_IMAGES}
TILE_SIZE:          ${TILE_SIZE}
CONDITION_TILE:     ${CONDITION_TILE_SIZE}
LABEL_H:            ${LABEL_H}
TILE_MODE:          ${TILE_MODE}
PANEL_FORMAT:       ${PANEL_FORMAT}
PNG_LEVEL:          ${PNG_COMPRESS_LEVEL}
UPLOAD:             ${UPLOAD}
RCLONE_DEST:        ${RCLONE_DEST}
=========================================
EOF

if [[ ! -f "${SELECTION_MANIFEST}" ]]; then
  echo "Error: SELECTION_MANIFEST not found: ${SELECTION_MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${EXP_ROOT}" ]]; then
  echo "Error: EXP_ROOT not found: ${EXP_ROOT}" >&2
  exit 1
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT_ROOT}" "${ARCHIVE_PATH}"
fi
mkdir -p "${OUT_ROOT}/panels" "${OUT_ROOT}/logs" "${OUT_ROOT}/metadata"

EXP_ROOT="${EXP_ROOT}" \
EXPERIMENTS="${EXPERIMENTS}" \
SELECTION_MANIFEST="${SELECTION_MANIFEST}" \
OUT_ROOT="${OUT_ROOT}" \
MAX_IMAGES="${MAX_IMAGES}" \
TILE_SIZE="${TILE_SIZE}" \
CONDITION_TILE_SIZE="${CONDITION_TILE_SIZE}" \
LABEL_H="${LABEL_H}" \
TILE_MODE="${TILE_MODE}" \
PANEL_FORMAT="${PANEL_FORMAT}" \
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import csv
import json
import os

exp_root = Path(os.environ["EXP_ROOT"])
experiments = os.environ["EXPERIMENTS"].split()
selection_manifest = Path(os.environ["SELECTION_MANIFEST"])
out_root = Path(os.environ["OUT_ROOT"])
max_images = int(os.environ["MAX_IMAGES"])
tile_size = int(os.environ["TILE_SIZE"])
condition_tile = int(os.environ["CONDITION_TILE_SIZE"])
label_h = int(os.environ["LABEL_H"])
tile_mode = os.environ["TILE_MODE"]
panel_format = os.environ["PANEL_FORMAT"].lower().lstrip(".")
png_compress_level = int(os.environ["PNG_COMPRESS_LEVEL"])
if tile_mode not in {"cover", "contain"}:
    raise SystemExit(f"TILE_MODE must be cover or contain, got: {tile_mode}")
if panel_format not in {"png", "jpg", "jpeg"}:
    raise SystemExit(f"PANEL_FORMAT must be png, jpg, or jpeg, got: {panel_format}")

panel_dir = out_root / "panels"
metadata_dir = out_root / "metadata"
panel_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

def open_rgb(path):
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None

def make_tile(path, label, body_size, label_h, mode):
    canvas = Image.new("RGB", (body_size, body_size + label_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, body_size - 1, label_h - 1], fill=(28, 28, 28))
    draw.text((8, max(4, (label_h - 14) // 2)), label[:64], fill=(255, 255, 255))
    if path and Path(path).exists():
        im = open_rgb(Path(path))
        if im is None:
            draw.text((10, label_h + 16), "read error", fill=(180, 0, 0))
        else:
            if mode == "cover":
                im = ImageOps.fit(im, (body_size, body_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            else:
                im.thumbnail((body_size, body_size), Image.Resampling.LANCZOS)
            x = (body_size - im.width) // 2
            y = label_h + (body_size - im.height) // 2
            canvas.paste(im, (x, y))
    else:
        draw.text((10, label_h + 16), "missing", fill=(180, 0, 0))
    return canvas

def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
selection_records = selection.get("records", [])[:max_images]
exp_records = {}
for exp in experiments:
    manifest = exp_root / exp / "manifest.jsonl"
    records = read_jsonl(manifest)
    by_index = {}
    for rec in records:
        if "index" in rec:
            by_index[int(rec["index"])] = rec
    exp_records[exp] = {"manifest": str(manifest), "records": records, "by_index": by_index}

exp_labels = {
    "e1_original_stable": "1 original stable",
    "e2_style_only": "2 style only",
    "e3_depth_only": "3 depth only",
    "e4_style_depth": "4 style + depth",
    "e5_text_style_linked": "5 text-style linked",
    "e6_text_depth_linked": "6 text-depth linked",
    "e7_text_style_depth_linked": "7 text-style-depth linked",
}
condition_labels = ["source", "ref raw", "ref lightfield", "depth"]
rows = []
for row_idx, sel in enumerate(selection_records):
    condition_paths = [
        sel.get("selected_source") or sel.get("source") or "",
        sel.get("selected_reference_raw") or sel.get("reference_raw") or "",
        sel.get("selected_reference_lightfield") or "",
        sel.get("selected_depth") or sel.get("depth") or "",
    ]
    condition_tiles = [
        make_tile(path, label, condition_tile, label_h, tile_mode)
        for path, label in zip(condition_paths, condition_labels)
    ]
    condition_col_h = sum(t.height for t in condition_tiles)
    result_h = tile_size + label_h
    panel_h = max(condition_col_h, result_h)
    panel_w = condition_tile + tile_size * len(experiments)
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    y = 0
    for tile in condition_tiles:
        panel.paste(tile, (0, y))
        y += tile.height

    exp_outputs = {}
    for col, exp in enumerate(experiments):
        rec = exp_records[exp]["by_index"].get(row_idx)
        out_path = ""
        if rec:
            out_path = rec.get("output") or rec.get("generated") or rec.get("generated_path") or ""
        exp_outputs[exp] = out_path
        tile = make_tile(out_path, exp_labels.get(exp, exp), tile_size, label_h, tile_mode)
        panel.paste(tile, (condition_tile + col * tile_size, 0))

    suffix = "jpg" if panel_format == "jpeg" else panel_format
    key = Path(condition_paths[0]).stem if condition_paths[0] else f"sample_{row_idx:03d}"
    panel_path = panel_dir / f"{row_idx:03d}_{key}.{suffix}"
    if panel_format in {"jpg", "jpeg"}:
        panel.save(panel_path, quality=100, subsampling=0)
    else:
        panel.save(panel_path, compress_level=png_compress_level)
    rows.append({
        "index": row_idx,
        "key": key,
        "relative": sel.get("relative", ""),
        "source": condition_paths[0],
        "reference_raw": condition_paths[1],
        "reference_lightfield": condition_paths[2],
        "depth": condition_paths[3],
        "experiments": exp_outputs,
        "panel": str(panel_path),
    })

summary = {
    "exp_root": str(exp_root),
    "experiments": experiments,
    "selection_manifest": str(selection_manifest),
    "out_root": str(out_root),
    "max_images": max_images,
    "tile_size": tile_size,
    "condition_tile_size": condition_tile,
    "label_h": label_h,
    "tile_mode": tile_mode,
    "panel_format": panel_format,
    "png_compress_level": png_compress_level,
    "panel_count": len(rows),
    "experiment_manifests": {k: v["manifest"] for k, v in exp_records.items()},
    "rows": rows,
}
(out_root / "condition_linkage_grid_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with (out_root / "condition_linkage_grid_rows.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["index", "key", "relative", "source", "reference_raw", "reference_lightfield", "depth", *experiments, "panel"])
    for row in rows:
        writer.writerow([
            row["index"], row["key"], row["relative"], row["source"], row["reference_raw"],
            row["reference_lightfield"], row["depth"],
            *[row["experiments"].get(exp, "") for exp in experiments], row["panel"],
        ])
print(json.dumps({"panel_count": len(rows), "panel_dir": str(panel_dir), "summary": str(out_root / "condition_linkage_grid_summary.json")}, indent=2))
PY

cp -a "${SELECTION_MANIFEST}" "${OUT_ROOT}/metadata/selection_manifest.json"
for exp in ${EXPERIMENTS}; do
  mkdir -p "${OUT_ROOT}/metadata/${exp}"
  for name in manifest.jsonl summary.json; do
    if [[ -f "${EXP_ROOT}/${exp}/${name}" ]]; then
      cp -a "${EXP_ROOT}/${exp}/${name}" "${OUT_ROOT}/metadata/${exp}/${name}"
    fi
  done
  if [[ -d "${LOG_ROOT}" ]]; then
    shopt -s nullglob
    for log_file in "${LOG_ROOT}"/*"${exp}"*.log; do
      cp -a "${log_file}" "${OUT_ROOT}/logs/"
    done
    shopt -u nullglob
  fi
done

rm -f "${ARCHIVE_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone not found. Set UPLOAD=0 to skip upload." >&2
    exit 1
  fi
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
else
  echo "Skip upload because UPLOAD=${UPLOAD}"
fi

echo "Done."
echo "Export dir: ${OUT_ROOT}"
echo "Archive:    ${ARCHIVE_PATH}"
echo "Remote:     ${RCLONE_DEST}"