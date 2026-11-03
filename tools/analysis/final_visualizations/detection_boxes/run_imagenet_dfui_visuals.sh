#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the visualization output directory}"
RUOD_IMAGE_LIST="${RUOD_IMAGE_LIST:-${OUTPUT_ROOT}/ruod_images100.txt}"
UIIS_IMAGE_LIST="${UIIS_IMAGE_LIST:-${OUTPUT_ROOT}/uiis_images100.txt}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.50}"
MAX_DETECTIONS="${MAX_DETECTIONS:-15}"

R50_DET_CONFIG="${R50_DET_CONFIG:?set R50_DET_CONFIG}"
VITS_DET_CONFIG="${VITS_DET_CONFIG:?set VITS_DET_CONFIG}"
R50_MASK_CONFIG="${R50_MASK_CONFIG:?set R50_MASK_CONFIG}"
VITS_MASK_CONFIG="${VITS_MASK_CONFIG:?set VITS_MASK_CONFIG}"

R50_DET_CHECKPOINT="${R50_DET_CHECKPOINT:?set R50_DET_CHECKPOINT}"
VITS_DET_CHECKPOINT="${VITS_DET_CHECKPOINT:?set VITS_DET_CHECKPOINT}"
R50_MASK_CHECKPOINT="${R50_MASK_CHECKPOINT:?set R50_MASK_CHECKPOINT}"
VITS_MASK_CHECKPOINT="${VITS_MASK_CHECKPOINT:?set VITS_MASK_CHECKPOINT}"

RENDER="${SCRIPT_DIR}/render_supervision_predictions.py"
COMPOSE="${SCRIPT_DIR}/compose_prediction_panels.py"

for required in \
  "${RENDER}" "${COMPOSE}" \
  "${RUOD_IMAGE_LIST}" "${UIIS_IMAGE_LIST}" \
  "${R50_DET_CONFIG}" "${VITS_DET_CONFIG}" \
  "${R50_MASK_CONFIG}" "${VITS_MASK_CONFIG}" \
  "${R50_DET_CHECKPOINT}" "${VITS_DET_CHECKPOINT}" \
  "${R50_MASK_CHECKPOINT}" "${VITS_MASK_CHECKPOINT}"
do
  [ -s "${required}" ] || {
    echo "ERROR: missing or empty file: ${required}" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_ROOT}/logs"

safe_remove() {
  local target="$1"
  [ -e "${target}" ] || return 0
  local real
  real="$(readlink -f "${target}")"
  case "${real}" in
    "${OUTPUT_ROOT}"/*)
      rm -rf -- "${real}"
      echo "removed: ${real}"
      ;;
    *)
      echo "REFUSE unexpected path: ${real}" >&2
      exit 1
      ;;
  esac
}

for target in \
  "${OUTPUT_ROOT}/ruod/r50" \
  "${OUTPUT_ROOT}/ruod/vits" \
  "${OUTPUT_ROOT}/uiis/r50" \
  "${OUTPUT_ROOT}/uiis/vits" \
  "${OUTPUT_ROOT}/panels"
do
  safe_remove "${target}"
done

mkdir -p \
  "${OUTPUT_ROOT}/ruod/r50" \
  "${OUTPUT_ROOT}/ruod/vits" \
  "${OUTPUT_ROOT}/uiis/r50" \
  "${OUTPUT_ROOT}/uiis/vits" \
  "${OUTPUT_ROOT}/panels"

run_model() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local checkpoint="$4"
  local images="$5"
  local task="$6"
  local output="$7"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RENDER}" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --image-list "${images}" \
    --output-root "${output}" \
    --model-name "${label}" \
    --task "${task}" \
    --device cuda:0 \
    --score-threshold "${SCORE_THRESHOLD}" \
    --max-detections "${MAX_DETECTIONS}"
}

run_model r50 2 "${R50_DET_CONFIG}" "${R50_DET_CHECKPOINT}" \
  "${RUOD_IMAGE_LIST}" detect "${OUTPUT_ROOT}/ruod/r50" \
  > "${OUTPUT_ROOT}/logs/ruod_r50.log" 2>&1 & P1=$!

run_model vits 3 "${VITS_DET_CONFIG}" "${VITS_DET_CHECKPOINT}" \
  "${RUOD_IMAGE_LIST}" detect "${OUTPUT_ROOT}/ruod/vits" \
  > "${OUTPUT_ROOT}/logs/ruod_vits.log" 2>&1 & P2=$!

run_model r50 4 "${R50_MASK_CONFIG}" "${R50_MASK_CHECKPOINT}" \
  "${UIIS_IMAGE_LIST}" segment "${OUTPUT_ROOT}/uiis/r50" \
  > "${OUTPUT_ROOT}/logs/uiis_r50.log" 2>&1 & P3=$!

run_model vits 5 "${VITS_MASK_CONFIG}" "${VITS_MASK_CHECKPOINT}" \
  "${UIIS_IMAGE_LIST}" segment "${OUTPUT_ROOT}/uiis/vits" \
  > "${OUTPUT_ROOT}/logs/uiis_vits.log" 2>&1 & P4=$!

FAILED=0
wait "${P1}" || FAILED=1
wait "${P2}" || FAILED=1
wait "${P3}" || FAILED=1
wait "${P4}" || FAILED=1

if [ "${FAILED}" -ne 0 ]; then
  echo "RESULT=FAIL_SUPERVISION_GENERATION" >&2
  grep -HnE 'ERROR|Traceback|RESULT=FAIL' "${OUTPUT_ROOT}"/logs/*.log >&2 || true
  exit 1
fi

"${PYTHON_BIN}" "${COMPOSE}" \
  --image-list "${RUOD_IMAGE_LIST}" \
  --left-dir "${OUTPUT_ROOT}/ruod/r50/box" \
  --right-dir "${OUTPUT_ROOT}/ruod/vits/box" \
  --output-dir "${OUTPUT_ROOT}/panels/ruod_box" \
  --left-label "R50: box" \
  --right-label "ViT-S: box"

for variant in box mask box_mask; do
  "${PYTHON_BIN}" "${COMPOSE}" \
    --image-list "${UIIS_IMAGE_LIST}" \
    --left-dir "${OUTPUT_ROOT}/uiis/r50/${variant}" \
    --right-dir "${OUTPUT_ROOT}/uiis/vits/${variant}" \
    --output-dir "${OUTPUT_ROOT}/panels/uiis_${variant}" \
    --left-label "R50: ${variant}" \
    --right-label "ViT-S: ${variant}"
done

printf '%s\n' \
  "RESULT=PASS_SUPERVISION_GENERATION_AND_PANELS" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "SCORE_THRESHOLD=${SCORE_THRESHOLD}" \
  "MAX_DETECTIONS=${MAX_DETECTIONS}"
