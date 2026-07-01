#!/usr/bin/env bash
set -euo pipefail

# Generate Depth Anything V2 maps for all sampled synthetic ImageNet sources.
# The output mirrors each method's source tree and preserves every source
# image's original spatial size.
#
# Smoke:
#   MODE=smoke LIMIT=50 bash scripts/exp_2/synthesis/run_depthanything_v2_all_sources.sh
#
# Full:
#   MODE=full bash scripts/exp_2/synthesis/run_depthanything_v2_all_sources.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
OUT_ROOT="${OUT_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR:-/home/fcp/xcx/exp_2/syn/Depth-Anything-V2}"
ENCODER="${ENCODER:-vitb}"
case "${ENCODER}" in
  vits) DEFAULT_CKPT="${DEPTHANYTHING_DIR}/checkpoints/depth_anything_v2_vits.pth" ;;
  vitb) DEFAULT_CKPT="${DEPTHANYTHING_DIR}/checkpoints/depth_anything_v2_vitb.pth" ;;
  vitl) DEFAULT_CKPT="${DEPTHANYTHING_DIR}/checkpoints/depth_anything_v2_vitl.pth" ;;
  *) echo "Error: ENCODER must be vits, vitb, or vitl. Got: ${ENCODER}" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CKPT}}"

MODE="${MODE:-full}"
METHODS="${METHODS:-uwnr syreanet watergan cut uwdf}"
SPLITS="${SPLITS:-train val}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
INPUT_SIZE="${INPUT_SIZE:-518}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
INVERT="${INVERT:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/depthanything_v2}"

if [[ "${MODE}" == "smoke" && "${LIMIT}" == "0" ]]; then
  LIMIT=50
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Error: MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

IFS=', ' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: no GPU ids provided. Set GPU_IDS=2,3,4,5" >&2
  exit 1
fi
if [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "Error: PROCS_PER_GPU must be >= 1" >&2
  exit 1
fi

EXPANDED_GPU_IDS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
  for _ in $(seq 1 "${PROCS_PER_GPU}"); do
    EXPANDED_GPU_IDS+=("${gpu_id}")
  done
done
NUM_SHARDS="${#EXPANDED_GPU_IDS[@]}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Depth Anything V2 all-source generation"
echo "========================================="
echo "SOURCE_ROOT:       ${SOURCE_ROOT}"
echo "OUT_ROOT:          ${OUT_ROOT}"
echo "DEPTHANYTHING_DIR: ${DEPTHANYTHING_DIR}"
echo "CHECKPOINT:        ${CHECKPOINT}"
echo "ENCODER:           ${ENCODER}"
echo "MODE:              ${MODE}"
echo "METHODS:           ${METHODS}"
echo "SPLITS:            ${SPLITS}"
echo "GPU_IDS:           ${GPU_ARRAY[*]}"
echo "PROCS_PER_GPU:     ${PROCS_PER_GPU}"
echo "NUM_SHARDS:        ${NUM_SHARDS}"
echo "INPUT_SIZE:        ${INPUT_SIZE}"
echo "LIMIT:             ${LIMIT}"
echo "OVERWRITE:         ${OVERWRITE}"
echo "INVERT:            ${INVERT}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

if [[ ! -d "${DEPTHANYTHING_DIR}" ]]; then
  echo "Error: Depth Anything V2 repository not found: ${DEPTHANYTHING_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: Depth Anything V2 checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

extra_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "${INVERT}" == "1" ]]; then
  extra_args+=(--invert)
fi

for method in ${METHODS}; do
  for split in ${SPLITS}; do
    image_dir="${SOURCE_ROOT}/${method}/source/${split}"
    out_dir="${OUT_ROOT}/${method}/${split}"
    if [[ ! -d "${image_dir}" ]]; then
      echo "Warning: skip missing source directory: ${image_dir}" | tee -a "${LOG_DIR}/missing_sources.log"
      continue
    fi

    echo
    echo "-----------------------------------------"
    echo "method/split: ${method}/${split}"
    echo "image_dir:    ${image_dir}"
    echo "out_dir:      ${out_dir}"
    echo "-----------------------------------------"

    pids=()
    failed=0
    for idx in "${!EXPANDED_GPU_IDS[@]}"; do
      gpu_id="${EXPANDED_GPU_IDS[$idx]}"
      shard_log="${LOG_DIR}/${method}_${split}_${ENCODER}_shard${idx}of${NUM_SHARDS}.log"
      echo "  launch shard ${idx}/${NUM_SHARDS} on GPU ${gpu_id}; log=${shard_log}"
      (
        python tools/generate_depthanything_maps.py \
          --image-dir "${image_dir}" \
          --out-dir "${out_dir}" \
          --depthanything-dir "${DEPTHANYTHING_DIR}" \
          --checkpoint "${CHECKPOINT}" \
          --encoder "${ENCODER}" \
          --device "cuda:${gpu_id}" \
          --input-size "${INPUT_SIZE}" \
          --limit "${LIMIT}" \
          --num-shards "${NUM_SHARDS}" \
          --shard-index "${idx}" \
          "${extra_args[@]}"
      ) > "${shard_log}" 2>&1 &
      pids+=("$!")
    done

    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if [[ "${failed}" != "0" ]]; then
      echo "Depth Anything V2 failed for ${method}/${split}. Check ${LOG_DIR}/${method}_${split}_${ENCODER}_shard* logs." >&2
      exit 1
    fi
  done
done

echo
echo "Done."
echo "Depth output root: ${OUT_ROOT}"
