#!/usr/bin/env bash
set -euo pipefail

# Export UWDF depth/control ablation outputs as multi-panel grids and upload.
#
# Default experiment layout:
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation/a_text_image
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation/b_text_image_ref
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation/c_text_image_depth
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation/d_text_image_ref_depth
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation/e_ref_depth_stronger
#
# The script builds:
#   source | depth | reference | exp1 | exp2 | exp3 | exp4 | exp5
#
# Usage:
#   bash scripts/exp_2/synthesis/export_uwdf_depth_ablation_multigrid_to_gdrive.sh
#
# Common overrides:
#   EXP_ROOT=/media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation \
#   EXPERIMENTS="a_text_image b_text_image_ref c_text_image_depth d_text_image_ref_depth e_ref_depth_stronger" \
#   MAX_IMAGES=20 \
#   RCLONE_DEST=fcp:datasets/exp2_synthesis_visual/ \
#   bash scripts/exp_2/synthesis/export_uwdf_depth_ablation_multigrid_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_depth_ablation}"
EXPERIMENTS="${EXPERIMENTS:-a_text_image b_text_image_ref c_text_image_depth d_text_image_ref_depth e_ref_depth_stronger}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_depth_ablation_multigrid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf/train}"
MAX_IMAGES="${MAX_IMAGES:-20}"
TILE_SIZE="${TILE_SIZE:-1024}"
GRID_COLUMNS="${GRID_COLUMNS:-2}"
TILE_MODE="${TILE_MODE:-cover}"
PANEL_FORMAT="${PANEL_FORMAT:-png}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-0}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"
AUTO_DETECT_EXP_ROOT="${AUTO_DETECT_EXP_ROOT:-1}"
SYNTHESIS_WORK_ROOT="${SYNTHESIS_WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"

echo "========================================="
echo "Export UWDF depth ablation multi-grid"
echo "========================================="
echo "EXP_ROOT:     ${EXP_ROOT}"
echo "EXPERIMENTS:  ${EXPERIMENTS}"
echo "OUT_ROOT:     ${OUT_ROOT}"
echo "ARCHIVE_PATH: ${ARCHIVE_PATH}"
echo "LOG_ROOT:     ${LOG_ROOT}"
echo "DEPTH_ROOT:   ${DEPTH_ROOT}"
echo "MAX_IMAGES:   ${MAX_IMAGES}"
echo "TILE_SIZE:    ${TILE_SIZE}"
echo "GRID_COLUMNS: ${GRID_COLUMNS}"
echo "TILE_MODE:    ${TILE_MODE}"
echo "PANEL_FORMAT: ${PANEL_FORMAT}"
echo "PNG_LEVEL:    ${PNG_COMPRESS_LEVEL}"
echo "UPLOAD:       ${UPLOAD}"
echo "RCLONE_DEST:  ${RCLONE_DEST}"
echo "OVERWRITE:    ${OVERWRITE}"
echo "AUTO_DETECT:  ${AUTO_DETECT_EXP_ROOT}"
echo "========================================="

if [[ ! -d "${EXP_ROOT}" ]]; then
  if [[ "${AUTO_DETECT_EXP_ROOT}" == "1" && -d "${SYNTHESIS_WORK_ROOT}" ]]; then
    echo "Warning: EXP_ROOT not found: ${EXP_ROOT}" >&2
    echo "Try auto-detecting experiment root under ${SYNTHESIS_WORK_ROOT}" >&2
    detected="$(
      SYNTHESIS_WORK_ROOT="${SYNTHESIS_WORK_ROOT}" \
      EXPERIMENTS="${EXPERIMENTS}" \
      python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["SYNTHESIS_WORK_ROOT"])
experiments = os.environ["EXPERIMENTS"].split()
best = None
best_count = 0
for child in sorted(root.iterdir()):
    if not child.is_dir():
        continue
    count = sum(1 for exp in experiments if (child / exp).is_dir())
    if count > best_count:
        best = child
        best_count = count
if best is not None and best_count > 0:
    print(best)
PY
    )"
    if [[ -n "${detected}" && -d "${detected}" ]]; then
      EXP_ROOT="${detected}"
      echo "Auto-detected EXP_ROOT: ${EXP_ROOT}" >&2
    else
      echo "Error: EXP_ROOT not found and auto-detect found no matching experiment root." >&2
      echo "Candidate directories under ${SYNTHESIS_WORK_ROOT}:" >&2
      find "${SYNTHESIS_WORK_ROOT}" -maxdepth 2 -type d 2>/dev/null | sed -n '1,120p' >&2
      exit 1
    fi
  else
    echo "Error: EXP_ROOT not found: ${EXP_ROOT}" >&2
    echo "Set EXP_ROOT to the parent directory that contains the five experiment folders." >&2
    exit 1
  fi
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}/multi_panel" "${OUT_ROOT}/experiments" "${OUT_ROOT}/logs"

echo
echo "Build multi-panel images"
EXP_ROOT="${EXP_ROOT}" \
EXPERIMENTS="${EXPERIMENTS}" \
OUT_ROOT="${OUT_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
MAX_IMAGES="${MAX_IMAGES}" \
TILE_SIZE="${TILE_SIZE}" \
GRID_COLUMNS="${GRID_COLUMNS}" \
TILE_MODE="${TILE_MODE}" \
PANEL_FORMAT="${PANEL_FORMAT}" \
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import csv
import json
import os
import shutil

exp_root = Path(os.environ["EXP_ROOT"])
experiments = os.environ["EXPERIMENTS"].split()
out_root = Path(os.environ["OUT_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
max_images = int(os.environ["MAX_IMAGES"])
tile_size = int(os.environ["TILE_SIZE"])
grid_columns = max(1, int(os.environ["GRID_COLUMNS"]))
tile_mode = os.environ["TILE_MODE"]
panel_format = os.environ["PANEL_FORMAT"].lower().lstrip(".")
png_compress_level = int(os.environ["PNG_COMPRESS_LEVEL"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
if tile_mode not in {"contain", "cover"}:
    raise SystemExit(f"TILE_MODE must be contain or cover, got: {tile_mode}")
if panel_format not in {"png", "jpg", "jpeg"}:
    raise SystemExit(f"PANEL_FORMAT must be png, jpg, or jpeg, got: {panel_format}")

panel_dir = out_root / "multi_panel"
panel_dir.mkdir(parents=True, exist_ok=True)

def safe_open(path):
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None

def tile_image(path, label, tile=320, label_h=34, mode="cover"):
    canvas = Image.new("RGB", (tile, tile + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, tile - 1, label_h - 1], fill=(32, 32, 32))
    draw.text((8, 9), label[:46], fill=(255, 255, 255))
    if path and Path(path).exists():
        im = safe_open(path)
        if im is not None:
            if mode == "cover":
                im = ImageOps.fit(im, (tile, tile), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            else:
                im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = (tile - im.width) // 2
            y = label_h + (tile - im.height) // 2
            canvas.paste(im, (x, y))
        else:
            draw.text((10, label_h + 16), "read error", fill=(180, 0, 0))
    else:
        draw.text((10, label_h + 16), "missing", fill=(180, 0, 0))
    return canvas

def find_manifest(exp_dir):
    candidates = [
        exp_dir / "manifest.jsonl",
        exp_dir / "generated" / "manifest.jsonl",
        exp_dir / "summary.json",
    ]
    candidates.extend(sorted(exp_dir.rglob("manifest.jsonl")))
    candidates.extend(sorted(exp_dir.rglob("summary.json")))
    seen = set()
    out = []
    for p in candidates:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def normalize_record(record, exp_dir):
    def first_existing(keys):
        for k in keys:
            v = record.get(k)
            if isinstance(v, str) and v:
                p = Path(v)
                if p.exists():
                    return p
                # Some scripts store paths relative to experiment root.
                q = exp_dir / v
                if q.exists():
                    return q
        return None

    source = first_existing(["source", "source_path", "source_image", "input", "image", "init_image"])
    depth = first_existing(["depth", "depth_path", "depth_image", "control_image"])
    reference = first_existing(["reference", "reference_path", "reference_image", "ip_adapter_image"])
    generated = first_existing(["generated", "generated_path", "output", "output_path", "save_path"])
    rel = record.get("relative") or record.get("source_relative") or record.get("stem") or None
    key = None
    if source:
        key = source.stem
    elif rel:
        key = Path(rel).stem
    elif generated:
        key = generated.stem.split("_underwater")[0]
    return {
        "key": key,
        "relative": rel,
        "source": str(source) if source else "",
        "depth": str(depth) if depth else "",
        "reference": str(reference) if reference else "",
        "generated": str(generated) if generated else "",
    }

def infer_depth_from_source(source_path, relative=""):
    if not depth_root.exists():
        return ""
    candidates = []
    if relative:
        candidates.append((depth_root / relative).with_suffix(".png"))
    if source_path:
        p = Path(source_path)
        # Common source layouts:
        #   .../source/train/<synset>/<name>.JPEG
        #   .../selected/source/train/<synset>/<name>.JPEG
        #   .../<synset>/<name>.JPEG
        if p.parent.name and p.parent.name.startswith("n"):
            candidates.append(depth_root / p.parent.name / f"{p.stem}.png")
        candidates.append(depth_root / f"{p.stem}.png")
        parts = p.parts
        for marker in ("train", "val"):
            if marker in parts:
                i = parts.index(marker)
                if i + 2 < len(parts):
                    rel = Path(*parts[i + 1:])
                    candidates.append((depth_root / rel).with_suffix(".png"))
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return str(c)
    return ""

def records_from_jsonl(path, exp_dir):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rec = normalize_record(item, exp_dir)
                if rec["key"]:
                    records.append(rec)
    return records

def records_from_summary(path, exp_dir):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = []
    if isinstance(data, dict):
        for key in ["records", "items", "outputs", "images", "generated"]:
            if isinstance(data.get(key), list):
                items = data[key]
                break
    records = []
    for item in items:
        if isinstance(item, dict):
            rec = normalize_record(item, exp_dir)
            if rec["key"]:
                records.append(rec)
    return records

def scan_generated(exp_dir):
    candidates = []
    for sub in ["generated", "outputs", "images"]:
        root = exp_dir / sub
        if root.exists():
            candidates.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not candidates:
        candidates = [p for p in exp_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts and "compare" not in str(p).lower()]
    records = []
    for p in sorted(candidates):
        stem = p.stem.split("_underwater")[0]
        records.append({"key": stem, "relative": "", "source": "", "depth": "", "reference": "", "generated": str(p)})
    return records

def load_experiment(exp):
    exp_dir = exp_root / exp
    if not exp_dir.exists():
        return {"exp": exp, "root": str(exp_dir), "exists": False, "records": []}
    records = []
    for p in find_manifest(exp_dir):
        if p.suffix == ".jsonl":
            records = records_from_jsonl(p, exp_dir)
        elif p.name == "summary.json":
            records = records_from_summary(p, exp_dir)
        if records:
            break
    if not records:
        records = scan_generated(exp_dir)

    by_key = {}
    for r in records:
        if r["key"] and r["key"] not in by_key:
            by_key[r["key"]] = r
    return {"exp": exp, "root": str(exp_dir), "exists": True, "records": records, "by_key": by_key}

loaded = [load_experiment(e) for e in experiments]
existing = [x for x in loaded if x["exists"]]
if not existing:
    raise SystemExit(f"No experiment directories found under {exp_root}")

base = max(existing, key=lambda x: len(x.get("records", [])))
base_records = base.get("records", [])[:max_images]

rows = []
for idx, base_rec in enumerate(base_records):
    key = base_rec["key"]
    row = {
        "index": idx,
        "key": key,
        "relative": base_rec.get("relative", ""),
        "source": base_rec.get("source", ""),
        "depth": base_rec.get("depth", ""),
        "reference": base_rec.get("reference", ""),
        "experiments": {},
    }
    for exp_data in loaded:
        rec = exp_data.get("by_key", {}).get(key)
        if rec is None and idx < len(exp_data.get("records", [])):
            rec = exp_data["records"][idx]
        if rec:
            row["experiments"][exp_data["exp"]] = rec.get("generated", "")
            row["source"] = row["source"] or rec.get("source", "")
            row["depth"] = row["depth"] or rec.get("depth", "")
            row["reference"] = row["reference"] or rec.get("reference", "")
        else:
            row["experiments"][exp_data["exp"]] = ""
    if not row["depth"]:
        row["depth"] = infer_depth_from_source(row["source"], row["relative"])
    rows.append(row)

label_h = 34
labels = ["source", "depth", "reference"] + experiments
for row in rows:
    tiles = [
        tile_image(row["source"], "source", tile_size, label_h, tile_mode),
        tile_image(row["depth"], "depth", tile_size, label_h, tile_mode),
        tile_image(row["reference"], "reference", tile_size, label_h, tile_mode),
    ]
    for exp in experiments:
        tiles.append(tile_image(row["experiments"].get(exp, ""), exp, tile_size, label_h, tile_mode))
    ncols = min(grid_columns, len(tiles))
    nrows = (len(tiles) + ncols - 1) // ncols
    grid = Image.new("RGB", (tile_size * ncols, (tile_size + label_h) * nrows), (255, 255, 255))
    for i, tile in enumerate(tiles):
        x = (i % ncols) * tile_size
        y = (i // ncols) * (tile_size + label_h)
        grid.paste(tile, (x, y))
    suffix = "jpg" if panel_format == "jpeg" else panel_format
    out_path = panel_dir / f"{row['index']:03d}_{row['key'] or 'sample'}.{suffix}"
    if panel_format in {"jpg", "jpeg"}:
        grid.save(out_path, quality=100, subsampling=0)
    else:
        grid.save(out_path, compress_level=png_compress_level)
    row["panel"] = str(out_path)

summary = {
    "exp_root": str(exp_root),
    "out_root": str(out_root),
    "depth_root": str(depth_root),
    "experiments": [
        {
            "name": x["exp"],
            "root": x["root"],
            "exists": x["exists"],
            "record_count": len(x.get("records", [])),
        }
        for x in loaded
    ],
    "base_experiment": base["exp"],
    "max_images": max_images,
    "tile_size": tile_size,
    "grid_columns": grid_columns,
    "tile_mode": tile_mode,
    "panel_format": panel_format,
    "png_compress_level": png_compress_level,
    "panel_count": len(rows),
    "rows": rows,
}
(out_root / "multigrid_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

with (out_root / "multigrid_rows.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["index", "key", "relative", "source", "depth", "reference", *experiments, "panel"])
    for row in rows:
        writer.writerow([
            row["index"],
            row["key"],
            row["relative"],
            row["source"],
            row["depth"],
            row["reference"],
            *[row["experiments"].get(exp, "") for exp in experiments],
            row["panel"],
        ])

for exp in experiments:
    src = exp_root / exp
    dst = out_root / "experiments" / exp
    if not src.exists():
        continue
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.jsonl", "summary.json", "uwdf_selection_manifest.json"]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    for sub in ["comparisons", "compare_4panel"]:
        p = src / sub
        if p.exists():
            shutil.copytree(p, dst / sub, dirs_exist_ok=True)

print(json.dumps({
    "base_experiment": base["exp"],
    "panel_count": len(rows),
    "summary": str(out_root / "multigrid_summary.json"),
    "rows": str(out_root / "multigrid_rows.tsv"),
    "panel_dir": str(panel_dir),
}, indent=2, ensure_ascii=False))
PY

echo
echo "Copy matching logs"
shopt -s nullglob
for exp in ${EXPERIMENTS}; do
  for log_file in "${LOG_ROOT}"/*"${exp}"*.log; do
    cp -a "${log_file}" "${OUT_ROOT}/logs/"
  done
done
shopt -u nullglob

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
