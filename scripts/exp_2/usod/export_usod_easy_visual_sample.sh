#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Visualize a random sample of USOD easy annotations, package it as a zip, and
# upload it with rclone. The output images are the original images with COCO
# bbox annotations drawn on top, so the converted detection labels can be
# inspected quickly.

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
USOD_ROOT="${USOD_ROOT:-/media/HDD0/XCX/exp_2/USOD10K}"
EASY_ANN="${EASY_ANN:-$USOD_ROOT/annotations/cross_split_det/easy_merged.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$USOD_ROOT/images}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-20260615}"
OUT_BASE="${OUT_BASE:-exports/usod_easy_visual_samples}"
RUN_NAME="${RUN_NAME:-usod_easy_${NUM_SAMPLES}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$OUT_BASE/$RUN_NAME"
ZIP_PATH="$OUT_BASE/${RUN_NAME}.zip"
RCLONE_DEST="${RCLONE_DEST:-syn:exp_2/usod_easy_visual_samples/}"
UPLOAD="${UPLOAD:-1}"

mkdir -p "$OUT_BASE"

if [ ! -f "$EASY_ANN" ]; then
    echo "Error: USOD easy annotation not found: $EASY_ANN"
    echo "Run first:"
    echo "  bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh"
    exit 1
fi

if [ ! -d "$IMAGE_ROOT" ]; then
    echo "Error: USOD image root not found: $IMAGE_ROOT"
    exit 1
fi

echo "========================================="
echo "USOD easy bbox visualization sample"
echo "========================================="
echo "USOD_ROOT: $USOD_ROOT"
echo "EASY_ANN: $EASY_ANN"
echo "IMAGE_ROOT: $IMAGE_ROOT"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "SEED: $SEED"
echo "OUT_DIR: $OUT_DIR"
echo "ZIP_PATH: $ZIP_PATH"
echo "UPLOAD: $UPLOAD"
echo "RCLONE_DEST: $RCLONE_DEST"
echo "========================================="

"$PYTHON" tools/visualize_coco_bbox_samples.py \
    --dataset usod_easy "$EASY_ANN" "$IMAGE_ROOT" \
    --out-dir "$OUT_DIR" \
    --num "$NUM_SAMPLES" \
    --seed "$SEED" \
    --threshold 0.0 \
    --strict

"$PYTHON" - "$OUT_DIR" "$ZIP_PATH" <<'PY'
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
dst.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(src.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(src.parent))

print(f"Zip written: {dst}")
PY

if [ "$UPLOAD" = "1" ]; then
    if ! command -v rclone >/dev/null 2>&1; then
        echo "Error: rclone not found. Set UPLOAD=0 to skip upload."
        exit 1
    fi
    rclone copy -P "$ZIP_PATH" "$RCLONE_DEST"
    echo "Uploaded: $ZIP_PATH -> $RCLONE_DEST"
else
    echo "Skip upload because UPLOAD=$UPLOAD"
fi

echo "========================================="
echo "Done"
echo "========================================="
echo "Output dir: $OUT_DIR"
echo "Zip file: $ZIP_PATH"
