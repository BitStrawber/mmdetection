#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/media/SSD1/XCX/exp_2/REALUW/imagefolder/train/realuw}"
OUT_DIR="${OUT_DIR:-/media/SSD1/XCX/exp_2/REALUW/quality_check}"
WORKERS="${WORKERS:-32}"
CHUNKSIZE="${CHUNKSIZE:-64}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10000}"

cd "$(dirname "$0")/../../.."

python -u tools/check_realuw_bad_images.py \
  --root "$ROOT" \
  --out-dir "$OUT_DIR" \
  --workers "$WORKERS" \
  --chunksize "$CHUNKSIZE" \
  --progress-every "$PROGRESS_EVERY" \
  "$@"
