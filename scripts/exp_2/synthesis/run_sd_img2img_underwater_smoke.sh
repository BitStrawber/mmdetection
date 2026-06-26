#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/stable_diffusion_img2img}"
GPU="${GPU:-2}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"
SD_STEPS="${SD_STEPS:-20}"
SD_STRENGTH="${SD_STRENGTH:-0.35}"
SD_GUIDANCE_SCALE="${SD_GUIDANCE_SCALE:-5.0}"
SD_BATCH_SIZE="${SD_BATCH_SIZE:-1}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Stable Diffusion img2img smoke test"
echo "========================================="
echo "Source sample family: uwdf"
echo "SYN_ROOT:            ${SYN_ROOT}"
echo "WORK_ROOT:           ${WORK_ROOT}"
echo "GPU:                 ${GPU}"
echo "SMOKE_TRAIN_LIMIT:   ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:     ${SMOKE_VAL_LIMIT}"
echo "SD_STEPS:            ${SD_STEPS}"
echo "SD_STRENGTH:         ${SD_STRENGTH}"
echo "SD_GUIDANCE_SCALE:   ${SD_GUIDANCE_SCALE}"
echo "SD_BATCH_SIZE:       ${SD_BATCH_SIZE}"
echo "LOG_DIR:             ${LOG_DIR}"
echo "========================================="

MODE=smoke \
METHODS="stable_diffusion_img2img" \
SPLITS="train val" \
GPU="${GPU}" \
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

for split in train val; do
  SOURCE_DIR="${WORK_ROOT}/sources/stable_diffusion_img2img/${split}" \
  OUT_DIR="${WORK_ROOT}/stable_diffusion_img2img/generated/${split}" \
  SPLIT="${split}" \
  LIMIT=0 \
  GPU="${GPU}" \
  STEPS="${SD_STEPS}" \
  STRENGTH="${SD_STRENGTH}" \
  GUIDANCE_SCALE="${SD_GUIDANCE_SCALE}" \
  BATCH_SIZE="${SD_BATCH_SIZE}" \
  bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh \
    2>&1 | tee "${LOG_DIR}/${split}.log"
done

echo "Stable Diffusion img2img smoke outputs:"
echo "  ${WORK_ROOT}/stable_diffusion_img2img/generated/train"
echo "  ${WORK_ROOT}/stable_diffusion_img2img/generated/val"
