#!/usr/bin/env bash
set -euo pipefail

# Prepare the flat data layout expected by the original TensorFlow WaterGAN
# code:
#   air_images/*.png   ImageNet RGB images resized to 640x480
#   air_depth/*.mat    matching pseudo-depth maps, or air_depth/*.png when
#                      DEPTH_FORMAT=png and WaterGAN has been patched
#   water_images/*.png RUOD real underwater reference images resized to 1360x1024
#
# Smoke:
#   conda activate /media/SSD1/conda_envs/syreanet
#   SPLIT=train AIR_LIMIT=1000 WATER_LIMIT=1000 GPU=2 bash scripts/exp_2/synthesis/run_watergan_prepare_dataset.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
SPLIT="${SPLIT:-train}"
GPU="${GPU:-2}"

SOURCE_DIR="${SOURCE_DIR:-${SOURCE_ROOT}/watergan/source/${SPLIT}}"
WATER_SOURCE="${WATER_SOURCE:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_${SPLIT}_smoke}"
DEPTH_DIR="${DEPTH_DIR:-${WORK_ROOT}/watergan/depth/${SPLIT}}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"

MEGADEPTH_DIR="${MEGADEPTH_DIR:-/home/fcp/xcx/exp_2/syn/MegaDepth}"
MEGADEPTH_CKPT="${MEGADEPTH_CKPT:-${MEGADEPTH_DIR}/checkpoints/best_generalization_net_G.pth}"

AIR_LIMIT="${AIR_LIMIT:-1000}"
WATER_LIMIT="${WATER_LIMIT:-1000}"
AIR_PER_CLASS="${AIR_PER_CLASS:-0}"
WATER_REPEAT_TO="${WATER_REPEAT_TO:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
AIR_WIDTH="${AIR_WIDTH:-640}"
AIR_HEIGHT="${AIR_HEIGHT:-480}"
WATER_WIDTH="${WATER_WIDTH:-1360}"
WATER_HEIGHT="${WATER_HEIGHT:-1024}"
DEPTH_FORMAT="${DEPTH_FORMAT:-mat}"
RUN_DEPTH="${RUN_DEPTH:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
RESUME="${RESUME:-1}"
VERIFY_EXISTING="${VERIFY_EXISTING:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "Error: RESUME=1 and OVERWRITE=1 are mutually exclusive." >&2
  exit 1
fi

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

echo "========================================="
echo "Prepare WaterGAN ImageNet + RUOD dataset"
echo "========================================="
echo "SYN_ROOT:       ${SYN_ROOT}"
echo "SOURCE_ROOT:    ${SOURCE_ROOT}"
echo "WORK_ROOT:      ${WORK_ROOT}"
echo "SPLIT:          ${SPLIT}"
echo "GPU:            ${GPU}"
echo "SOURCE_DIR:     ${SOURCE_DIR}"
echo "WATER_SOURCE:   ${WATER_SOURCE}"
echo "DEPTH_DIR:      ${DEPTH_DIR}"
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "MEGADEPTH_DIR:  ${MEGADEPTH_DIR}"
echo "MEGADEPTH_CKPT: ${MEGADEPTH_CKPT}"
echo "RUN_DEPTH:      ${RUN_DEPTH}"
echo "AIR_LIMIT:      ${AIR_LIMIT}"
echo "WATER_LIMIT:    ${WATER_LIMIT}"
echo "AIR_PER_CLASS:  ${AIR_PER_CLASS}"
echo "WATER_REPEAT_TO:${WATER_REPEAT_TO}"
echo "SAMPLE_SEED:    ${SAMPLE_SEED}"
echo "AIR_SIZE:       ${AIR_WIDTH}x${AIR_HEIGHT}"
echo "WATER_SIZE:     ${WATER_WIDTH}x${WATER_HEIGHT}"
echo "DEPTH_FORMAT:   ${DEPTH_FORMAT}"
echo "NUM_WORKERS:    ${NUM_WORKERS}"
echo "RESUME:         ${RESUME}"
echo "VERIFY_EXISTING:${VERIFY_EXISTING}"
echo "OVERWRITE:      ${OVERWRITE}"
echo "========================================="
echo

check_path "${SOURCE_DIR}" "WaterGAN sampled ImageNet source"
check_path "${WATER_SOURCE}" "RUOD/reference underwater image source"
if [[ "${RUN_DEPTH}" == "1" ]]; then
  check_path "${MEGADEPTH_DIR}" "MegaDepth directory"
  check_path "${MEGADEPTH_CKPT}" "MegaDepth checkpoint"
fi

mkdir -p "${DEPTH_DIR}" "${DATA_ROOT}"

if [[ "${RUN_DEPTH}" == "1" ]]; then
  echo "Step 1/2: Generate MegaDepth maps"
  python tools/generate_megadepth_maps.py \
    --image-dir "${SOURCE_DIR}" \
    --out-dir "${DEPTH_DIR}" \
    --megadepth-dir "${MEGADEPTH_DIR}" \
    --checkpoint "${MEGADEPTH_CKPT}" \
    --device "cuda:${GPU}" \
    --limit "${AIR_LIMIT}"
else
  echo "Step 1/2: Skip MegaDepth generation"
fi

echo
echo "Step 2/2: Build WaterGAN flat training folders"
EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
elif [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ "${VERIFY_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--verify-existing)
fi
if (( AIR_PER_CLASS > 0 )); then
  EXTRA_ARGS+=(--air-per-class "${AIR_PER_CLASS}")
fi
if (( WATER_REPEAT_TO > 0 )); then
  EXTRA_ARGS+=(--water-repeat-to "${WATER_REPEAT_TO}")
fi

python tools/prepare_watergan_imagenet_ruod_dataset.py \
  --air-source "${SOURCE_DIR}" \
  --depth-source "${DEPTH_DIR}" \
  --water-source "${WATER_SOURCE}" \
  --out-dir "${DATA_ROOT}" \
  --air-limit "${AIR_LIMIT}" \
  --water-limit "${WATER_LIMIT}" \
  --air-width "${AIR_WIDTH}" \
  --air-height "${AIR_HEIGHT}" \
  --water-width "${WATER_WIDTH}" \
  --water-height "${WATER_HEIGHT}" \
  --depth-format "${DEPTH_FORMAT}" \
  --workers "${NUM_WORKERS}" \
  --seed "${SAMPLE_SEED}" \
  "${EXTRA_ARGS[@]}"

echo
echo "Done."
echo "WaterGAN dataset: ${DATA_ROOT}"
echo "  air_images:   ${DATA_ROOT}/air_images"
echo "  air_depth:    ${DATA_ROOT}/air_depth"
echo "  water_images: ${DATA_ROOT}/water_images"
