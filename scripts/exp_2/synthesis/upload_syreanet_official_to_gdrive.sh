#!/usr/bin/env bash
set -euo pipefail

# Upload SyreaNet official-parameter synthetic ImageNet outputs to Google Drive.
# Use directory copy instead of a giant tarball so interrupted transfers can be
# resumed by rerunning the same command.
#
# Example:
#   SPLITS=train bash scripts/exp_2/synthesis/upload_syreanet_official_to_gdrive.sh
#
# Optional:
#   GENERATED_ROOT=/media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis_official/generated \
#   RCLONE_DEST=fcp:datasets/exp2_synthetic_imagenet/syreanet_synthesis_official/generated \
#   SPLITS="train val" \
#   bash scripts/exp_2/synthesis/upload_syreanet_official_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GENERATED_ROOT="${GENERATED_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis_official/generated}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthetic_imagenet/syreanet_synthesis_official/generated}"
SPLITS="${SPLITS:-train}"
LOG_SOURCE="${LOG_SOURCE:-${REPO_ROOT}/logs/synthesis_full/syreanet_synthesis_official}"
UPLOAD_LOGS="${UPLOAD_LOGS:-1}"
TRANSFERS="${TRANSFERS:-8}"
CHECKERS="${CHECKERS:-16}"
DRY_RUN="${DRY_RUN:-0}"

echo "========================================="
echo "Upload SyreaNet official synthesis outputs"
echo "========================================="
echo "GENERATED_ROOT: ${GENERATED_ROOT}"
echo "RCLONE_DEST:    ${RCLONE_DEST}"
echo "SPLITS:         ${SPLITS}"
echo "LOG_SOURCE:     ${LOG_SOURCE}"
echo "UPLOAD_LOGS:    ${UPLOAD_LOGS}"
echo "TRANSFERS:      ${TRANSFERS}"
echo "CHECKERS:       ${CHECKERS}"
echo "DRY_RUN:        ${DRY_RUN}"
echo "========================================="
echo

RCLONE_ARGS=(copy -P --transfers "${TRANSFERS}" --checkers "${CHECKERS}")
if [[ "${DRY_RUN}" == "1" ]]; then
  RCLONE_ARGS+=(--dry-run)
fi

count_images() {
  local dir="$1"
  find "${dir}" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' -o -iname '*.JPEG' \
  \) | wc -l
}

for split in ${SPLITS}; do
  src="${GENERATED_ROOT}/${split}"
  dst="${RCLONE_DEST}/${split}"
  if [[ ! -d "${src}" ]]; then
    echo "Error: generated split directory not found: ${src}" >&2
    exit 1
  fi

  echo "-----------------------------------------"
  echo "split: ${split}"
  echo "src:   ${src}"
  echo "dst:   ${dst}"
  echo "count: $(count_images "${src}")"
  echo "-----------------------------------------"
  rclone "${RCLONE_ARGS[@]}" "${src}" "${dst}"
done

if [[ "${UPLOAD_LOGS}" == "1" && -d "${LOG_SOURCE}" ]]; then
  echo
  echo "Upload logs"
  rclone "${RCLONE_ARGS[@]}" "${LOG_SOURCE}" "${RCLONE_DEST}/logs"
else
  echo
  echo "Skip logs upload."
fi

echo
echo "Done."
echo "Remote root: ${RCLONE_DEST}"
