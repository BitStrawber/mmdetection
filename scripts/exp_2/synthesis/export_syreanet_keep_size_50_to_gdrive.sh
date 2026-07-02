#!/usr/bin/env bash
set -euo pipefail

# Package and upload the SyreaNet keep-original-size 50-image smoke output.
#
# Usage:
#   bash scripts/exp_2/synthesis/export_syreanet_keep_size_50_to_gdrive.sh
#
# Common overrides:
#   RCLONE_DEST=fcp:datasets/exp2_synthesis_visual/ \
#   WORK_ROOT=/media/SSD1/XCX/exp_2/synthesis_work/syreanet_keep_size_50 \
#   bash scripts/exp_2/synthesis/export_syreanet_keep_size_50_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-train}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/syreanet_keep_size_50}"
PREP_DIR="${PREP_DIR:-${WORK_ROOT}/prepared/${SPLIT}}"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${WORK_ROOT}/generated_flat/${SPLIT}}"
RESTORE_DIR="${RESTORE_DIR:-${WORK_ROOT}/generated/${SPLIT}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/syreanet_keep_size_50.log}"

OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/syreanet_keep_size_50_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Export SyreaNet keep-size 50 outputs"
echo "========================================="
echo "SPLIT:         ${SPLIT}"
echo "WORK_ROOT:     ${WORK_ROOT}"
echo "PREP_DIR:      ${PREP_DIR}"
echo "FLAT_SAVE_DIR: ${FLAT_SAVE_DIR}"
echo "RESTORE_DIR:   ${RESTORE_DIR}"
echo "LOG_FILE:      ${LOG_FILE}"
echo "OUT_ROOT:      ${OUT_ROOT}"
echo "ARCHIVE_PATH:  ${ARCHIVE_PATH}"
echo "UPLOAD:        ${UPLOAD}"
echo "RCLONE_DEST:   ${RCLONE_DEST}"
echo "OVERWRITE:     ${OVERWRITE}"
echo "========================================="

if [[ ! -d "${PREP_DIR}" ]]; then
  echo "Error: PREP_DIR not found: ${PREP_DIR}" >&2
  exit 1
fi
if [[ ! -d "${FLAT_SAVE_DIR}" ]]; then
  echo "Error: FLAT_SAVE_DIR not found: ${FLAT_SAVE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RESTORE_DIR}" ]]; then
  echo "Error: RESTORE_DIR not found: ${RESTORE_DIR}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi

mkdir -p \
  "${OUT_ROOT}/prepared" \
  "${OUT_ROOT}/generated_flat" \
  "${OUT_ROOT}/generated" \
  "${OUT_ROOT}/logs"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  else
    echo "Warning: missing, skip: ${src}" >&2
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  else
    echo "Warning: missing directory, skip: ${src}" >&2
  fi
}

echo
echo "Copy generated outputs"
copy_dir_if_exists "${RESTORE_DIR}" "${OUT_ROOT}/generated/${SPLIT}"
copy_dir_if_exists "${FLAT_SAVE_DIR}" "${OUT_ROOT}/generated_flat/${SPLIT}"

echo
echo "Copy prepared image/depth pairs and manifest"
copy_dir_if_exists "${PREP_DIR}/image" "${OUT_ROOT}/prepared/${SPLIT}/image"
copy_dir_if_exists "${PREP_DIR}/depth" "${OUT_ROOT}/prepared/${SPLIT}/depth"
copy_if_exists "${PREP_DIR}/manifest.jsonl" "${OUT_ROOT}/prepared/${SPLIT}/manifest.jsonl"

echo
echo "Copy log"
copy_if_exists "${LOG_FILE}" "${OUT_ROOT}/logs/$(basename "${LOG_FILE}")"

echo
echo "Build export summary"
OUT_ROOT="${OUT_ROOT}" \
SPLIT="${SPLIT}" \
PREP_DIR="${PREP_DIR}" \
FLAT_SAVE_DIR="${FLAT_SAVE_DIR}" \
RESTORE_DIR="${RESTORE_DIR}" \
python - <<'PY'
from pathlib import Path
import json
import os

out_root = Path(os.environ["OUT_ROOT"])
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

summary = {
    "split": os.environ["SPLIT"],
    "prep_dir": os.environ["PREP_DIR"],
    "flat_save_dir": os.environ["FLAT_SAVE_DIR"],
    "restore_dir": os.environ["RESTORE_DIR"],
    "out_root": str(out_root),
    "counts": {
        "prepared_images": count_images(out_root / "prepared" / os.environ["SPLIT"] / "image"),
        "prepared_depth": count_images(out_root / "prepared" / os.environ["SPLIT"] / "depth"),
        "generated_flat": count_images(out_root / "generated_flat" / os.environ["SPLIT"]),
        "generated": count_images(out_root / "generated" / os.environ["SPLIT"]),
    },
}
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
