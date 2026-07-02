#!/usr/bin/env bash
set -euo pipefail

# Package and upload UWDF ablation outputs.
#
# Expected experiment directories by default:
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_ablation/exp1_strength020
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_ablation/exp2_prompt_preserve
#   /media/SSD1/XCX/exp_2/synthesis_work/uwdf_ablation/exp3_strength020_prompt_preserve
#
# Usage:
#   bash scripts/exp_2/synthesis/export_uwdf_ablation_to_gdrive.sh
#
# Common overrides:
#   RCLONE_DEST=fcp:datasets/exp2_synthesis_visual/ \
#   EXP_ROOT=/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ablation \
#   bash scripts/exp_2/synthesis/export_uwdf_ablation_to_gdrive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ablation}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_ablation_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
OVERWRITE="${OVERWRITE:-1}"

EXPERIMENTS="${EXPERIMENTS:-exp1_strength020 exp2_prompt_preserve exp3_strength020_prompt_preserve}"

echo "========================================="
echo "Export UWDF ablation outputs"
echo "========================================="
echo "EXP_ROOT:     ${EXP_ROOT}"
echo "EXPERIMENTS:  ${EXPERIMENTS}"
echo "OUT_ROOT:     ${OUT_ROOT}"
echo "ARCHIVE_PATH: ${ARCHIVE_PATH}"
echo "LOG_ROOT:     ${LOG_ROOT}"
echo "UPLOAD:       ${UPLOAD}"
echo "RCLONE_DEST:  ${RCLONE_DEST}"
echo "OVERWRITE:    ${OVERWRITE}"
echo "========================================="

if [[ ! -d "${EXP_ROOT}" ]]; then
  echo "Error: EXP_ROOT not found: ${EXP_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}/experiments" "${OUT_ROOT}/logs"

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

found=0
for exp in ${EXPERIMENTS}; do
  src="${EXP_ROOT}/${exp}"
  dst="${OUT_ROOT}/experiments/${exp}"
  echo
  echo "Copy experiment: ${exp}"
  echo "  src: ${src}"
  echo "  dst: ${dst}"

  if [[ ! -d "${src}" ]]; then
    echo "Warning: experiment directory not found, skip: ${src}" >&2
    continue
  fi
  found=$((found + 1))

  mkdir -p "${dst}"

  copy_dir_if_exists "${src}/compare_4panel" "${dst}/compare_4panel"
  copy_dir_if_exists "${src}/uwdf_text_imagenet/generated" "${dst}/text_imagenet_generated"
  copy_dir_if_exists "${src}/uwdf_text_ref_imagenet/generated" "${dst}/text_ref_imagenet_generated"
  copy_dir_if_exists "${src}/source" "${dst}/source"
  copy_dir_if_exists "${src}/reference" "${dst}/reference"
  copy_dir_if_exists "${src}/logs" "${dst}/logs"

  copy_if_exists "${src}/uwdf_selection_manifest.json" "${dst}/uwdf_selection_manifest.json"
  copy_if_exists "${src}/uwdf_text_imagenet/manifest.jsonl" "${dst}/text_imagenet_manifest.jsonl"
  copy_if_exists "${src}/uwdf_text_imagenet/summary.json" "${dst}/text_imagenet_summary.json"
  copy_if_exists "${src}/uwdf_text_ref_imagenet/manifest.jsonl" "${dst}/text_ref_imagenet_manifest.jsonl"
  copy_if_exists "${src}/uwdf_text_ref_imagenet/summary.json" "${dst}/text_ref_imagenet_summary.json"
done

if [[ "${found}" -eq 0 ]]; then
  echo "Error: no experiment directories were found under ${EXP_ROOT}" >&2
  exit 1
fi

echo
echo "Copy top-level matching logs"
for log_name in \
  uwdf_ablation_exp1_strength020.log \
  uwdf_ablation_exp2_prompt_preserve.log \
  uwdf_ablation_exp3_strength020_prompt_preserve.log
do
  copy_if_exists "${LOG_ROOT}/${log_name}" "${OUT_ROOT}/logs/${log_name}"
done

echo
echo "Build export summary"
OUT_ROOT="${OUT_ROOT}" \
EXP_ROOT="${EXP_ROOT}" \
EXPERIMENTS="${EXPERIMENTS}" \
python - <<'PY'
from pathlib import Path
import json
import os

out_root = Path(os.environ["OUT_ROOT"])
exp_root = Path(os.environ["EXP_ROOT"])
experiments = os.environ["EXPERIMENTS"].split()
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

records = []
for exp in experiments:
    root = out_root / "experiments" / exp
    records.append({
        "experiment": exp,
        "source_root": str(exp_root / exp),
        "export_root": str(root),
        "exists": root.exists(),
        "compare_4panel": count_images(root / "compare_4panel"),
        "text_imagenet_generated": count_images(root / "text_imagenet_generated"),
        "text_ref_imagenet_generated": count_images(root / "text_ref_imagenet_generated"),
        "source_images": count_images(root / "source"),
        "reference_images": count_images(root / "reference"),
    })

summary = {
    "exp_root": str(exp_root),
    "out_root": str(out_root),
    "experiments": records,
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
