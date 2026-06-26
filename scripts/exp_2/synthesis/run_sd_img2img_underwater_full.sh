#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/stable_diffusion_img2img}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
SPLITS="${SPLITS:-train val}"
SD_STEPS="${SD_STEPS:-20}"
SD_STRENGTH="${SD_STRENGTH:-0.35}"
SD_GUIDANCE_SCALE="${SD_GUIDANCE_SCALE:-5.0}"
SD_BATCH_SIZE="${SD_BATCH_SIZE:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Stable Diffusion img2img full generation"
echo "========================================="
echo "Source sample family: uwdf"
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "PROCS_PER_GPU:     ${PROCS_PER_GPU}"
echo "SPLITS:            ${SPLITS}"
echo "SD_STEPS:          ${SD_STEPS}"
echo "SD_STRENGTH:       ${SD_STRENGTH}"
echo "SD_GUIDANCE_SCALE: ${SD_GUIDANCE_SCALE}"
echo "SD_BATCH_SIZE:     ${SD_BATCH_SIZE}"
echo "OMP_THREADS:       ${OMP_NUM_THREADS}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

MODE=full \
METHODS="stable_diffusion_img2img" \
SPLITS="${SPLITS}" \
GPU="${GPU}" \
FULL_LIMIT=0 \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

for split in ${SPLITS}; do
  SOURCE_DIR="${WORK_ROOT}/sources/stable_diffusion_img2img/${split}" \
  OUT_DIR="${SYN_ROOT}/stable_diffusion_img2img/generated/${split}" \
  SPLIT="${split}" \
  LIMIT=0 \
  GPU_IDS="${GPU_IDS}" \
  PROCS_PER_GPU="${PROCS_PER_GPU}" \
  STEPS="${SD_STEPS}" \
  STRENGTH="${SD_STRENGTH}" \
  GUIDANCE_SCALE="${SD_GUIDANCE_SCALE}" \
  BATCH_SIZE="${SD_BATCH_SIZE}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate_multi_gpu.sh \
    2>&1 | tee "${LOG_DIR}/${split}.log"
done

echo "Stable Diffusion img2img full outputs:"
echo "  ${SYN_ROOT}/stable_diffusion_img2img/generated/train"
echo "  ${SYN_ROOT}/stable_diffusion_img2img/generated/val"
