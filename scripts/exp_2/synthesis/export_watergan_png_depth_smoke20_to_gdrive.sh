#!/usr/bin/env bash
set -euo pipefail

NUM="${NUM:-20}"
SEED="${SEED:-20260723}"
BATCH_SIZE="${BATCH_SIZE:-64}"

RESULT_ROOT="${RESULT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_flat_48shards/smoke_png_64}"
SOURCE_SHARD_ROOT="${SOURCE_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_step1564_official_base_48shards/train}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPORT_ROOT="${EXPORT_ROOT:-/media/HDD2/XCX/exp_2/watergan_png_depth_review20_${STAMP}}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${EXPORT_ROOT}.tar.gz}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/watergan_png_depth_review}"

UPLOAD="${UPLOAD:-1}"
UPLOAD_EXPANDED="${UPLOAD_EXPANDED:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

section() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

count_generated() {
  find "$RESULT_ROOT" -mindepth 1 -maxdepth 2 \
    -type f -name 'fake_*.png' 2>/dev/null | wc -l
}

section "WaterGAN direct-PNG-depth random review export"
echo "NUM:              $NUM"
echo "SEED:             $SEED"
echo "BATCH_SIZE:       $BATCH_SIZE"
echo "RESULT_ROOT:      $RESULT_ROOT"
echo "SOURCE_SHARD_ROOT:$SOURCE_SHARD_ROOT"
echo "EXPORT_ROOT:      $EXPORT_ROOT"
echo "ARCHIVE_PATH:     $ARCHIVE_PATH"
echo "RCLONE_DEST:      $RCLONE_DEST"
echo "UPLOAD:           $UPLOAD"
echo "UPLOAD_EXPANDED:  $UPLOAD_EXPANDED"

[ -d "$RESULT_ROOT" ] || {
  echo "Error: direct-PNG result directory is missing: $RESULT_ROOT" >&2
  exit 1
}
[ -d "$SOURCE_SHARD_ROOT" ] || {
  echo "Error: source shard root is missing: $SOURCE_SHARD_ROOT" >&2
  exit 1
}

generated_count="$(count_generated)"
manifest_count="$({
  find "$SOURCE_SHARD_ROOT" \
    -mindepth 2 -maxdepth 2 \
    -type f \
    -name watergan_air_manifest.jsonl \
    -print0 2>/dev/null |
    xargs -0 -r cat
} | wc -l)"

echo
echo "direct-PNG generated images: $generated_count"
echo "source manifest rows:        $manifest_count"

[ "$generated_count" -ge "$NUM" ] || {
  echo "Error: found only $generated_count generated images; need $NUM" >&2
  exit 1
}

if [ "$CHECK_ONLY" = "1" ]; then
  section "Check only completed"
  exit 0
fi

[ ! -e "$EXPORT_ROOT" ] || {
  echo "Error: export directory already exists: $EXPORT_ROOT" >&2
  echo "Set a different STAMP or EXPORT_ROOT before retrying." >&2
  exit 1
}
[ ! -e "$ARCHIVE_PATH" ] || {
  echo "Error: archive already exists: $ARCHIVE_PATH" >&2
  exit 1
}

mkdir -p "$EXPORT_ROOT"

export NUM SEED BATCH_SIZE RESULT_ROOT SOURCE_SHARD_ROOT EXPORT_ROOT

section "Select, validate, and render 20 strict triplets"
python - <<'PY'
from __future__ import print_function

import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NUM = int(os.environ["NUM"])
SEED = int(os.environ["SEED"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
RESULT_ROOT = Path(os.environ["RESULT_ROOT"])
SOURCE_SHARD_ROOT = Path(os.environ["SOURCE_SHARD_ROOT"])
EXPORT_ROOT = Path(os.environ["EXPORT_ROOT"])

PATTERN = re.compile(r"^fake_(\d+)_(\d+)_(\d+)\.png$")
TILE_SIZE = (640, 480)
HEADER_HEIGHT = 42


def resolve_record_path(value, source_shard):
    path = Path(value)
    if not path.is_absolute():
        path = source_shard / path
    return path


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def fit_rgb(image, size):
    image = image.convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def depth_preview(image, size):
    array = np.asarray(image.convert("F"), dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        values = array[finite]
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            low = float(values.min())
            high = float(values.max())
        scale = max(high - low, 1e-6)
        normalized = np.clip((array - low) / scale, 0.0, 1.0)
        normalized = np.nan_to_num(normalized)
        normalized = (normalized * 255.0).astype(np.uint8)
    return fit_rgb(Image.fromarray(normalized, mode="L"), size)


def write_panel(source_path, depth_path, generated_path, destination):
    with Image.open(str(source_path)) as image:
        source = fit_rgb(image, TILE_SIZE)
    with Image.open(str(depth_path)) as image:
        depth = depth_preview(image, TILE_SIZE)
    with Image.open(str(generated_path)) as image:
        generated = fit_rgb(image, TILE_SIZE)

    width = TILE_SIZE[0] * 3
    height = TILE_SIZE[1] + HEADER_HEIGHT
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(source, (0, HEADER_HEIGHT))
    panel.paste(depth, (TILE_SIZE[0], HEADER_HEIGHT))
    panel.paste(generated, (TILE_SIZE[0] * 2, HEADER_HEIGHT))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=(28, 28, 28))
    labels = ("Source image", "PNG depth input", "WaterGAN generated")
    for column, label in enumerate(labels):
        draw.text(
            (column * TILE_SIZE[0] + 14, 13),
            label,
            fill=(255, 255, 255),
        )
    draw.line(
        (TILE_SIZE[0], 0, TILE_SIZE[0], height),
        fill=(255, 255, 255),
        width=2,
    )
    draw.line(
        (TILE_SIZE[0] * 2, 0, TILE_SIZE[0] * 2, height),
        fill=(255, 255, 255),
        width=2,
    )
    panel.save(str(destination), quality=95, subsampling=0)


candidates = []
direct_outputs = list(RESULT_ROOT.glob("fake_*.png"))
if direct_outputs:
    result_shards = [(RESULT_ROOT, SOURCE_SHARD_ROOT / "shard0of48")]
else:
    result_shards = [
        (path, SOURCE_SHARD_ROOT / path.name)
        for path in sorted(RESULT_ROOT.iterdir())
        if path.is_dir()
    ]

for result_shard, source_shard in result_shards:
    manifest = source_shard / "watergan_air_manifest.jsonl"
    if not manifest.is_file():
        continue

    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                record = json.loads(line)
                record["_manifest_index"] = line_number
                records.append(record)

    for generated_path in sorted(result_shard.glob("fake_*.png")):
        match = PATTERN.match(generated_path.name)
        if match is None:
            continue
        item_index = int(match.group(2))
        batch_index = int(match.group(3))
        manifest_index = batch_index * BATCH_SIZE + item_index
        if manifest_index >= len(records):
            continue

        record = records[manifest_index]
        source_path = resolve_record_path(record["air_image"], source_shard)
        depth_path = resolve_record_path(record["air_depth"], source_shard)
        if not source_path.is_file() or not depth_path.is_file():
            continue
        if depth_path.suffix.lower() != ".png":
            continue
        global_index = int(record.get("global_index", manifest_index))
        candidates.append((
            global_index,
            source_path,
            depth_path,
            generated_path,
            record,
        ))

if len(candidates) < NUM:
    raise SystemExit(
        "Error: found only {} strict PNG-depth triplets; need {}".format(
            len(candidates), NUM
        )
    )

random.seed(SEED)
selected = random.sample(candidates, NUM)
selected.sort(key=lambda item: item[0])

source_dir = EXPORT_ROOT / "source"
depth_dir = EXPORT_ROOT / "depth_png"
generated_dir = EXPORT_ROOT / "generated"
panel_dir = EXPORT_ROOT / "panels"
for directory in (source_dir, depth_dir, generated_dir, panel_dir):
    directory.mkdir(parents=True, exist_ok=True)

output_records = []
for order, item in enumerate(selected, 1):
    manifest_index, source_path, depth_path, generated_path, record = item
    synset = safe_component(record.get("synset", source_path.parent.name))
    stem = safe_component(record.get("original_stem", source_path.stem))
    prefix = "{:02d}_{:06d}_{}_{}".format(order, manifest_index, synset, stem)

    source_output = source_dir / (prefix + source_path.suffix.lower())
    depth_output = depth_dir / (prefix + "_depth.png")
    generated_output = generated_dir / (prefix + "_generated.png")
    panel_output = panel_dir / (prefix + "_panel.jpg")

    shutil.copy2(str(source_path), str(source_output))
    shutil.copy2(str(depth_path), str(depth_output))
    shutil.copy2(str(generated_path), str(generated_output))
    write_panel(source_path, depth_path, generated_path, panel_output)

    output_records.append({
        "order": order,
        "manifest_index": manifest_index,
        "source": str(source_path),
        "depth_png": str(depth_path),
        "generated": str(generated_path),
        "exported_source": str(source_output.relative_to(EXPORT_ROOT)),
        "exported_depth": str(depth_output.relative_to(EXPORT_ROOT)),
        "exported_generated": str(generated_output.relative_to(EXPORT_ROOT)),
        "panel": str(panel_output.relative_to(EXPORT_ROOT)),
        "checkpoint": "DCGAN.model-1564",
        "depth_input_mode": "png",
    })

manifest_output = EXPORT_ROOT / "selection_manifest.jsonl"
with manifest_output.open("w", encoding="utf-8") as handle:
    for record in output_records:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

readme = EXPORT_ROOT / "README.txt"
readme.write_text(
    "WaterGAN direct-PNG-depth visual review\n"
    "checkpoint: DCGAN.model-1564\n"
    "depth input mode: png\n"
    "samples: {}\n"
    "seed: {}\n"
    "panels: Source image | PNG depth input | WaterGAN generated\n".format(
        NUM, SEED
    ),
    encoding="utf-8",
)

print("strict candidates:", len(candidates))
print("selected:", len(selected))
print("source copies:", len(list(source_dir.iterdir())))
print("depth copies:", len(list(depth_dir.iterdir())))
print("generated copies:", len(list(generated_dir.iterdir())))
print("panels:", len(list(panel_dir.iterdir())))
print("export root:", EXPORT_ROOT)
PY

section "Package and checksum"
tar -C "$(dirname "$EXPORT_ROOT")" \
  -czf "$ARCHIVE_PATH" \
  "$(basename "$EXPORT_ROOT")"

sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"

ls -lh "$ARCHIVE_PATH" "${ARCHIVE_PATH}.sha256"
cat "${ARCHIVE_PATH}.sha256"

if [ "$UPLOAD" != "1" ]; then
  section "Local export completed; upload disabled"
  exit 0
fi

command -v rclone >/dev/null 2>&1 || {
  echo "Error: rclone is not installed" >&2
  exit 1
}

rclone listremotes | grep -Fxq 'fcp:' || {
  echo "Error: rclone remote fcp: is not configured" >&2
  exit 1
}

section "Upload review package to Google Drive"
remote_run="$RCLONE_DEST/$(basename "$EXPORT_ROOT")"

if [ "$UPLOAD_EXPANDED" = "1" ]; then
  rclone copy \
    --progress \
    --transfers 8 \
    --checkers 16 \
    "$EXPORT_ROOT" \
    "$remote_run"
fi

rclone copy \
  --progress \
  --transfers 4 \
  --checkers 8 \
  "$ARCHIVE_PATH" \
  "$RCLONE_DEST"

rclone copy \
  --progress \
  --transfers 4 \
  --checkers 8 \
  "${ARCHIVE_PATH}.sha256" \
  "$RCLONE_DEST"

section "Upload completed"
echo "Google Drive directory: $remote_run"
echo "Google Drive archive:   $RCLONE_DEST/$(basename "$ARCHIVE_PATH")"
echo "Local export:           $EXPORT_ROOT"
echo "Local archive:          $ARCHIVE_PATH"
