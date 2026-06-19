#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/media/HDD1/XCX/exp_2/REALUW_SSL}"
OUT_ROOT="${OUT_ROOT:-/media/SSD1/XCX/exp_2/REALUW}"
WORKERS="${WORKERS:-8}"
CHECK_WORKERS="${CHECK_WORKERS:-16}"

cd "$(dirname "$0")/../../.."

python tools/materialize_realuw_imagefolder.py \
  --src-root "$SRC_ROOT" \
  --out-root "$OUT_ROOT" \
  --workers "$WORKERS" \
  --copy-meta \
  --merge-to-train \
  --check-images \
  --check-workers "$CHECK_WORKERS" \
  "$@"
