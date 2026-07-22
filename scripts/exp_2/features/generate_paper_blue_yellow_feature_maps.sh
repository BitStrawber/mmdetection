#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-work_dirs/exp_2/feature_maps/torchvision_resnet50_imagenet_train_vs_j2_ruod_random10_fixed_preprocess}"
CASCADE_CONFIG="${CASCADE_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-/media/SSD1/XCX/exp_2/BitStrawber_Output/J2/det/checkpoint/best_coco_bbox_mAP_epoch_18.pth}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/feature_maps_paper_blue_yellow_${STAMP}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUT_ROOT}/archives}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/feature_maps/resnet50_random10_paper_blue_yellow}"

DEVICE="${DEVICE:-cuda:0}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-10}"
TILE_WIDTH="${TILE_WIDTH:-640}"
TILE_HEIGHT="${TILE_HEIGHT:-480}"
LABEL_HEIGHT="${LABEL_HEIGHT:-42}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.30}"
MAX_BOXES="${MAX_BOXES:-20}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1.0}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99.0}"
OVERWRITE="${OVERWRITE:-0}"
UPLOAD="${UPLOAD:-1}"
UPLOAD_EXPANDED="${UPLOAD_EXPANDED:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

VARIANTS=(
  imagenet_resnet50_blue_yellow_no_box
  ruod_cascade_resnet50_blue_yellow_no_box
  ruod_cascade_resnet50_blue_yellow_with_box
)

section() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

require_file() {
  [[ -s "$1" ]] || {
    echo "Error: required file is missing or empty: $1" >&2
    exit 1
  }
}

section "Paper blue-yellow feature-map generation"
echo "SOURCE_ROOT:        ${SOURCE_ROOT}"
echo "CASCADE_CONFIG:     ${CASCADE_CONFIG}"
echo "CASCADE_CHECKPOINT: ${CASCADE_CHECKPOINT}"
echo "OUT_ROOT:           ${OUT_ROOT}"
echo "ARCHIVE_DIR:        ${ARCHIVE_DIR}"
echo "DEVICE:             ${DEVICE}"
echo "TILE:               ${TILE_WIDTH}x${TILE_HEIGHT}"
echo "SCORE_THRESHOLD:    ${SCORE_THRESHOLD}"
echo "MAX_BOXES:          ${MAX_BOXES}"
echo "NORMALIZATION:      p${LOW_PERCENTILE}-p${HIGH_PERCENTILE}"
echo "UPLOAD:             ${UPLOAD}"
echo "UPLOAD_EXPANDED:    ${UPLOAD_EXPANDED}"
echo "RCLONE_DEST:        ${RCLONE_DEST}"

require_file "${SOURCE_ROOT}/manifest.tsv"
require_file "${CASCADE_CONFIG}"
require_file "${CASCADE_CHECKPOINT}"

feature_count="$(find "${SOURCE_ROOT}" -type f -name features.pt | wc -l)"
input_count="$(find "${SOURCE_ROOT}" -type f -name 'input.*' | wc -l)"
imagenet_rows="$(awk -F '\t' '$1 == "imagenet" {count++} END {print count + 0}' "${SOURCE_ROOT}/manifest.tsv")"
ruod_rows="$(awk -F '\t' '$1 == "ruod" {count++} END {print count + 0}' "${SOURCE_ROOT}/manifest.tsv")"

echo
echo "features.pt:       ${feature_count}/$((EXPECTED_SAMPLES * 2))"
echo "input images:      ${input_count}/$((EXPECTED_SAMPLES * 2))"
echo "ImageNet records:  ${imagenet_rows}/${EXPECTED_SAMPLES}"
echo "RUOD records:      ${ruod_rows}/${EXPECTED_SAMPLES}"

[[ "${feature_count}" -eq "$((EXPECTED_SAMPLES * 2))" ]] || {
  echo "Error: unexpected features.pt count" >&2; exit 1;
}
[[ "${input_count}" -eq "$((EXPECTED_SAMPLES * 2))" ]] || {
  echo "Error: unexpected input image count" >&2; exit 1;
}
[[ "${imagenet_rows}" -eq "${EXPECTED_SAMPLES}" ]] || {
  echo "Error: unexpected ImageNet manifest count" >&2; exit 1;
}
[[ "${ruod_rows}" -eq "${EXPECTED_SAMPLES}" ]] || {
  echo "Error: unexpected RUOD manifest count" >&2; exit 1;
}

if [[ "${CHECK_ONLY}" == 1 ]]; then
  section "Input validation completed"
  exit 0
fi

render_args=(
  --source-root "${SOURCE_ROOT}"
  --cascade-config "${CASCADE_CONFIG}"
  --cascade-checkpoint "${CASCADE_CHECKPOINT}"
  --out-dir "${OUT_ROOT}"
  --device "${DEVICE}"
  --expected-samples "${EXPECTED_SAMPLES}"
  --tile-width "${TILE_WIDTH}"
  --tile-height "${TILE_HEIGHT}"
  --label-height "${LABEL_HEIGHT}"
  --score-threshold "${SCORE_THRESHOLD}"
  --max-boxes "${MAX_BOXES}"
  --low-percentile "${LOW_PERCENTILE}"
  --high-percentile "${HIGH_PERCENTILE}"
)
[[ "${OVERWRITE}" == 1 ]] && render_args+=(--overwrite)

section "Render fixed-size feature maps"
python tools/exp_2/render_paper_feature_maps.py "${render_args[@]}"

section "Validate rendered image counts"
for variant in "${VARIANTS[@]}"; do
  root="${OUT_ROOT}/${variant}"
  originals="$(find "${root}/originals" -maxdepth 1 -type f -name '*.png' | wc -l)"
  feature_maps="$(find "${root}/feature_maps" -type f -name '*.png' | wc -l)"
  panels="$(find "${root}/five_panels" -maxdepth 1 -type f -name '*.png' | wc -l)"
  total=$((originals + feature_maps + panels))
  printf '%-48s originals=%d maps=%d panels=%d total=%d\n' \
    "${variant}" "${originals}" "${feature_maps}" "${panels}" "${total}"
  [[ "${originals}" -eq "${EXPECTED_SAMPLES}" && \
     "${feature_maps}" -eq "$((EXPECTED_SAMPLES * 4))" && \
     "${panels}" -eq "${EXPECTED_SAMPLES}" && \
     "${total}" -eq "$((EXPECTED_SAMPLES * 6))" ]] || {
    echo "Error: invalid rendered counts for ${variant}" >&2
    exit 1
  }
done

mkdir -p "${ARCHIVE_DIR}"
export OUT_ROOT ARCHIVE_DIR

section "Create and verify three ZIP archives"
python - <<'PY'
import hashlib
import os
import zipfile
from pathlib import Path

out_root = Path(os.environ['OUT_ROOT']).resolve()
archive_dir = Path(os.environ['ARCHIVE_DIR']).resolve()
variants = [
    'imagenet_resnet50_blue_yellow_no_box',
    'ruod_cascade_resnet50_blue_yellow_no_box',
    'ruod_cascade_resnet50_blue_yellow_with_box',
]
archive_dir.mkdir(parents=True, exist_ok=True)
checksum_lines = []

for variant in variants:
    source = out_root / variant
    archive_path = archive_dir / '{}.zip'.format(variant)
    with zipfile.ZipFile(
            str(archive_path), 'w', compression=zipfile.ZIP_DEFLATED,
            compresslevel=6) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                archive.write(str(path), str(path.relative_to(out_root)))
    with zipfile.ZipFile(str(archive_path), 'r') as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(
                'ZIP integrity failure in {}: {}'.format(archive_path, bad_file))
    digest = hashlib.sha256()
    with archive_path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    checksum_lines.append('{}  {}'.format(digest.hexdigest(), archive_path.name))
    print('{}: {} bytes'.format(archive_path, archive_path.stat().st_size))

(archive_dir / 'SHA256SUMS.txt').write_text(
    '\n'.join(checksum_lines) + '\n', encoding='utf-8')
PY

ls -lh "${ARCHIVE_DIR}"/*.zip "${ARCHIVE_DIR}/SHA256SUMS.txt"

if [[ "${UPLOAD}" != 1 ]]; then
  section "Generation completed; upload disabled"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "ARCHIVE_DIR=${ARCHIVE_DIR}"
  exit 0
fi

command -v rclone >/dev/null 2>&1 || {
  echo "Error: rclone is not installed" >&2; exit 1;
}
rclone listremotes | grep -Fxq 'fcp:' || {
  echo "Error: rclone remote fcp: is not configured" >&2; exit 1;
}

section "Upload feature-map archives to Google Drive"
rclone copy \
  --progress \
  --transfers 3 \
  --checkers 8 \
  --include '*.zip' \
  --include 'SHA256SUMS.txt' \
  "${ARCHIVE_DIR}" \
  "${RCLONE_DEST}"

if [[ "${UPLOAD_EXPANDED}" == 1 ]]; then
  remote_expanded="${RCLONE_DEST}/expanded_$(basename "${OUT_ROOT}")"
  for variant in "${VARIANTS[@]}"; do
    rclone copy \
      --progress \
      --transfers 8 \
      --checkers 16 \
      "${OUT_ROOT}/${variant}" \
      "${remote_expanded}/${variant}"
  done
  echo "Expanded remote: ${remote_expanded}"
fi

section "Feature-map package completed"
echo "Local output:  ${OUT_ROOT}"
echo "Archives:      ${ARCHIVE_DIR}"
echo "Google Drive:  ${RCLONE_DEST}"
rclone lsl "${RCLONE_DEST}" | tail -n 20
