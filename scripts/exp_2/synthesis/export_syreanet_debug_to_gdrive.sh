#!/usr/bin/env bash
set -euo pipefail

# Package SyreaNet synthesis debug outputs and upload them to Google Drive.
#
# Default package content:
#   - generated restored outputs
#   - generated flat outputs
#   - prepared image/depth pairs and manifest
#   - MegaDepth maps
#   - relevant logs
#
# Example:
#   bash scripts/exp_2/synthesis/export_syreanet_debug_to_gdrive.sh
#
# Optional:
#   DEBUG_ROOT=/media/SSD1/XCX/exp_2/synthesis_work/syreanet_debug \
#   RCLONE_DEST=fcp:datasets/exp2_synthesis_visual/ \
#   bash scripts/exp_2/synthesis/export_syreanet_debug_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-train}"
DEBUG_ROOT="${DEBUG_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/syreanet_debug}"
DEPTH_DIR="${DEPTH_DIR:-${DEBUG_ROOT}/depth/${SPLIT}}"
PREP_DIR="${PREP_DIR:-${DEBUG_ROOT}/prepared/${SPLIT}}"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${DEBUG_ROOT}/generated_flat/${SPLIT}}"
RESTORE_DIR="${RESTORE_DIR:-${DEBUG_ROOT}/generated/${SPLIT}}"

OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/syreanet_debug_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/media/HDD1/XCX/exp_2/syreanet_debug_export.tar.gz}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"
UPLOAD="${UPLOAD:-1}"

echo "========================================="
echo "Export SyreaNet debug outputs"
echo "========================================="
echo "SPLIT:         ${SPLIT}"
echo "DEBUG_ROOT:    ${DEBUG_ROOT}"
echo "DEPTH_DIR:     ${DEPTH_DIR}"
echo "PREP_DIR:      ${PREP_DIR}"
echo "FLAT_SAVE_DIR: ${FLAT_SAVE_DIR}"
echo "RESTORE_DIR:   ${RESTORE_DIR}"
echo "OUT_ROOT:      ${OUT_ROOT}"
echo "ARCHIVE_PATH:  ${ARCHIVE_PATH}"
echo "UPLOAD:        ${UPLOAD}"
echo "RCLONE_DEST:   ${RCLONE_DEST}"
echo "========================================="
echo

check_optional_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ -d "${src}" ]]; then
    echo "copy ${label}: ${src}"
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  else
    echo "warning: skip missing ${label}: ${src}" >&2
  fi
}

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT_ROOT}"
  rm -f "${ARCHIVE_PATH}"
fi
mkdir -p "${OUT_ROOT}"

check_optional_dir "${RESTORE_DIR}" "${OUT_ROOT}/generated/${SPLIT}" "restored generated outputs"
check_optional_dir "${FLAT_SAVE_DIR}" "${OUT_ROOT}/generated_flat/${SPLIT}" "flat generated outputs"
check_optional_dir "${PREP_DIR}" "${OUT_ROOT}/prepared/${SPLIT}" "prepared image/depth pairs"
check_optional_dir "${DEPTH_DIR}" "${OUT_ROOT}/depth/${SPLIT}" "MegaDepth maps"

mkdir -p "${OUT_ROOT}/logs"
for log_path in \
  "${REPO_ROOT}/logs/syreanet_debug_50.log" \
  "${REPO_ROOT}/logs/syreanet_synthesis_${SPLIT}.log" \
  "${REPO_ROOT}/logs/synthesis_smoke/syreanet_synthesis/${SPLIT}.log"
do
  if [[ -f "${log_path}" ]]; then
    cp -a "${log_path}" "${OUT_ROOT}/logs/"
  fi
done

export SPLIT DEBUG_ROOT DEPTH_DIR PREP_DIR FLAT_SAVE_DIR RESTORE_DIR OUT_ROOT

python - <<'PY'
from pathlib import Path
import json
import os

out_root = Path(os.environ["OUT_ROOT"])
paths = {
    "depth": Path(os.environ["DEPTH_DIR"]),
    "prepared": Path(os.environ["PREP_DIR"]),
    "generated_flat": Path(os.environ["FLAT_SAVE_DIR"]),
    "generated": Path(os.environ["RESTORE_DIR"]),
}
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
summary = {
    "split": os.environ["SPLIT"],
    "debug_root": os.environ["DEBUG_ROOT"],
    "paths": {k: str(v) for k, v in paths.items()},
    "counts": {},
}
for key, path in paths.items():
    if path.exists():
        summary["counts"][key] = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    else:
        summary["counts"][key] = 0

manifest = paths["prepared"] / "manifest.jsonl"
summary["manifest_lines"] = 0
if manifest.exists():
    summary["manifest_lines"] = sum(1 for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())

(out_root / "export_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "Create archive"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  echo
  echo "Upload to Google Drive"
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
fi

echo
echo "Done."
echo "Archive: ${ARCHIVE_PATH}"
echo "Remote:  ${RCLONE_DEST}"
