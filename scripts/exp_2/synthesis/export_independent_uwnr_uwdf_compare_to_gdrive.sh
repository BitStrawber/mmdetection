#!/usr/bin/env bash
set -euo pipefail

# Package and upload independent UWNR/UWDF comparison outputs.
#
# Usage:
#   NUM=20 SEED=2026 RCLONE_DEST=fcp:datasets/exp2_synthesis_visual/ \
#     bash scripts/exp_2/synthesis/export_independent_uwnr_uwdf_compare_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
UPLOAD="${UPLOAD:-1}"
OVERWRITE="${OVERWRITE:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"

UWNR_ROOT="${UWNR_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwnr_lnrud_ref/independent_random${NUM}_seed${SEED}}"
UWDF_ROOT="${UWDF_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/independent_train_random${NUM}_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"

OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/independent_uwnr_uwdf_compare_random${NUM}_seed${SEED}}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"

echo "========================================="
echo "Export independent UWNR/UWDF comparisons"
echo "========================================="
echo "NUM:          ${NUM}"
echo "SEED:         ${SEED}"
echo "UWNR_ROOT:    ${UWNR_ROOT}"
echo "UWDF_ROOT:    ${UWDF_ROOT}"
echo "LOG_ROOT:     ${LOG_ROOT}"
echo "OUT_ROOT:     ${OUT_ROOT}"
echo "ARCHIVE_PATH: ${ARCHIVE_PATH}"
echo "UPLOAD:       ${UPLOAD}"
echo "RCLONE_DEST:  ${RCLONE_DEST}"
echo "OVERWRITE:    ${OVERWRITE}"
echo "========================================="

if [[ ! -d "${UWNR_ROOT}" ]]; then
  echo "Error: UWNR_ROOT not found: ${UWNR_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${UWDF_ROOT}" ]]; then
  echo "Error: UWDF_ROOT not found: ${UWDF_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi

mkdir -p \
  "${OUT_ROOT}/uwnr" \
  "${OUT_ROOT}/uwdf" \
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
echo "Copy UWNR outputs"
copy_dir_if_exists "${UWNR_ROOT}/triplets" "${OUT_ROOT}/uwnr/triplets"
copy_dir_if_exists "${UWNR_ROOT}/generated/train" "${OUT_ROOT}/uwnr/generated"
copy_dir_if_exists "${UWNR_ROOT}/generated_flat/train" "${OUT_ROOT}/uwnr/generated_flat"
copy_if_exists "${UWNR_ROOT}/random_selection_manifest.json" "${OUT_ROOT}/uwnr/random_selection_manifest.json"
copy_if_exists "${UWNR_ROOT}/prepared/train/pair_manifest.jsonl" "${OUT_ROOT}/uwnr/pair_manifest.jsonl"
copy_if_exists "${UWNR_ROOT}/megadepth/train/megadepth_summary.json" "${OUT_ROOT}/uwnr/megadepth_summary.json"

echo
echo "Copy UWDF outputs"
copy_dir_if_exists "${UWDF_ROOT}/compare_4panel" "${OUT_ROOT}/uwdf/compare_4panel"
copy_dir_if_exists "${UWDF_ROOT}/uwdf_text_imagenet/generated" "${OUT_ROOT}/uwdf/text_imagenet_generated"
copy_dir_if_exists "${UWDF_ROOT}/uwdf_text_ref_imagenet/generated" "${OUT_ROOT}/uwdf/text_ref_imagenet_generated"
copy_if_exists "${UWDF_ROOT}/uwdf_selection_manifest.json" "${OUT_ROOT}/uwdf/uwdf_selection_manifest.json"
copy_if_exists "${UWDF_ROOT}/uwdf_text_imagenet/manifest.jsonl" "${OUT_ROOT}/uwdf/text_imagenet_manifest.jsonl"
copy_if_exists "${UWDF_ROOT}/uwdf_text_imagenet/summary.json" "${OUT_ROOT}/uwdf/text_imagenet_summary.json"
copy_if_exists "${UWDF_ROOT}/uwdf_text_ref_imagenet/manifest.jsonl" "${OUT_ROOT}/uwdf/text_ref_imagenet_manifest.jsonl"
copy_if_exists "${UWDF_ROOT}/uwdf_text_ref_imagenet/summary.json" "${OUT_ROOT}/uwdf/text_ref_imagenet_summary.json"

echo
echo "Copy logs"
copy_if_exists "${LOG_ROOT}/uwnr_independent_compare${NUM}.log" "${OUT_ROOT}/logs/uwnr_independent_compare${NUM}.log"
copy_if_exists "${LOG_ROOT}/uwdf_independent_compare${NUM}.log" "${OUT_ROOT}/logs/uwdf_independent_compare${NUM}.log"
copy_dir_if_exists "${UWDF_ROOT}/logs" "${OUT_ROOT}/logs/uwdf_internal_logs"

echo
echo "Build export summary"
OUT_ROOT="${OUT_ROOT}" \
UWNR_ROOT="${UWNR_ROOT}" \
UWDF_ROOT="${UWDF_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
python - <<'PY'
from pathlib import Path
import json
import os

out = Path(os.environ["OUT_ROOT"])
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

summary = {
    "num": int(os.environ["NUM"]),
    "seed": int(os.environ["SEED"]),
    "uwnr_root": os.environ["UWNR_ROOT"],
    "uwdf_root": os.environ["UWDF_ROOT"],
    "out_root": str(out),
    "counts": {
        "uwnr_triplets": count_images(out / "uwnr" / "triplets"),
        "uwnr_generated": count_images(out / "uwnr" / "generated"),
        "uwdf_4panel": count_images(out / "uwdf" / "compare_4panel"),
        "uwdf_text_imagenet_generated": count_images(out / "uwdf" / "text_imagenet_generated"),
        "uwdf_text_ref_imagenet_generated": count_images(out / "uwdf" / "text_ref_imagenet_generated"),
    },
}
(out / "export_summary.json").write_text(
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
