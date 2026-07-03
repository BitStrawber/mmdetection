#!/usr/bin/env bash
set -euo pipefail

# Randomly sample SyreaNet generated images, package them, and optionally upload
# to Google Drive via rclone.
#
# Usage:
#   cd ~/xcx/exp_2/mmdetection
#   NUM=50 bash scripts/exp_2/synthesis/export_syreanet_generated_sample_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-50}"
SEED="${SEED:-2026}"
SPLITS="${SPLITS:-train val}"
GENERATED_ROOT="${GENERATED_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/syreanet_generated_random${NUM}}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

if [[ -z "${GENERATED_ROOT}" ]]; then
  for candidate in \
    /media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis_official/generated \
    /media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis/generated \
    /media/SSD1/XCX/exp_2/synthesis_work/syreanet_synthesis/generated
  do
    if [[ -d "${candidate}" ]]; then
      GENERATED_ROOT="${candidate}"
      break
    fi
  done
fi

echo "========================================="
echo "Export SyreaNet generated random sample"
echo "========================================="
echo "GENERATED_ROOT: ${GENERATED_ROOT:-<not found>}"
echo "OUT_ROOT:       ${OUT_ROOT}"
echo "ARCHIVE_PATH:   ${ARCHIVE_PATH}"
echo "NUM:            ${NUM}"
echo "SEED:           ${SEED}"
echo "SPLITS:         ${SPLITS}"
echo "UPLOAD:         ${UPLOAD}"
echo "RCLONE_DEST:    ${RCLONE_DEST}"
echo "OVERWRITE:      ${OVERWRITE}"
echo "========================================="

if [[ -z "${GENERATED_ROOT}" || ! -d "${GENERATED_ROOT}" ]]; then
  echo "Error: GENERATED_ROOT not found. Set it explicitly, for example:" >&2
  echo "  GENERATED_ROOT=/media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis_official/generated \\" >&2
  echo "  bash scripts/exp_2/synthesis/export_syreanet_generated_sample_to_gdrive.sh" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT_ROOT}" "${ARCHIVE_PATH}"
fi
mkdir -p "${OUT_ROOT}/images"

GENERATED_ROOT="${GENERATED_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
SPLITS="${SPLITS}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

generated_root = Path(os.environ["GENERATED_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])
num = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
splits = os.environ["SPLITS"].split()
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []
for split in splits:
    split_root = generated_root / split
    if not split_root.exists():
        continue
    for path in split_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            records.append((split, path))

if not records:
    raise RuntimeError(f"No generated images found under {generated_root} for splits={splits}")

records.sort(key=lambda item: str(item[1]))
rng = random.Random(seed)
selected = rng.sample(records, min(num, len(records)))

manifest = []
for index, (split, src) in enumerate(tqdm(selected, desc="copy generated samples", unit="image")):
    rel = src.relative_to(generated_root / split)
    dst = out_root / "images" / split / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({
        "index": index,
        "split": split,
        "relative": str(rel).replace("\\", "/"),
        "source_path": str(src),
        "export_path": str(dst),
    })

summary = {
    "generated_root": str(generated_root),
    "out_root": str(out_root),
    "splits": splits,
    "requested": num,
    "available": len(records),
    "exported": len(manifest),
    "seed": seed,
}

(out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
(out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "Create archive"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  echo
  echo "Upload archive to Google Drive"
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
fi

echo
echo "Done."
echo "Export dir: ${OUT_ROOT}"
echo "Archive:    ${ARCHIVE_PATH}"
if [[ "${UPLOAD}" == "1" ]]; then
  echo "Remote:     ${RCLONE_DEST}"
fi
