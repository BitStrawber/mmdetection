#!/usr/bin/env bash
set -euo pipefail

# Stable Diffusion img2img underwater baseline.
# Two guidance sources are used:
#   1. ImageNet image encoded by the VAE into an init latent.
#   2. Underwater text prompt encoded by CLIP.
#
# Smoke:
#   SPLIT=train LIMIT=100 GPU=2 bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh
#
# Original CompVis checkpoint:
#   MODEL=/home/fcp/xcx/exp_2/syn/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt \
#   SD_SINGLE_FILE=1 SPLIT=train LIMIT=100 GPU=2 \
#     bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh
#
# Full shard:
#   SPLIT=train LIMIT=0 GPU=2 NUM_SHARDS=4 SHARD_INDEX=0 bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-2}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

SOURCE_DIR="${SOURCE_DIR:-${SYN_ROOT}/stable_diffusion_img2img/source/${SPLIT}}"
OUT_DIR="${OUT_DIR:-${SYN_ROOT}/stable_diffusion_img2img/generated/${SPLIT}}"
MODEL="${MODEL:-runwayml/stable-diffusion-v1-5}"
SD_SINGLE_FILE="${SD_SINGLE_FILE:-0}"
PROMPT="${PROMPT:-a realistic underwater photograph of the same scene, blue-green water, underwater haze, natural color attenuation, low contrast, realistic lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, extra objects, fish, coral, diver, text, watermark, blurry, low quality}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEED="${SEED:-2026}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
DISABLE_SAFETY_CHECKER="${DISABLE_SAFETY_CHECKER:-1}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

SHARD_TAG=""
if [[ "${NUM_SHARDS}" != "1" ]]; then
  SHARD_TAG="_shard${SHARD_INDEX}of${NUM_SHARDS}"
fi

echo "========================================="
echo "Stable Diffusion img2img underwater generation"
echo "========================================="
echo "SOURCE_DIR:      ${SOURCE_DIR}"
echo "OUT_DIR:         ${OUT_DIR}"
echo "MODEL:           ${MODEL}"
echo "SD_SINGLE_FILE:  ${SD_SINGLE_FILE}"
echo "SPLIT:           ${SPLIT}"
echo "LIMIT:           ${LIMIT}"
echo "GPU:             ${GPU}"
echo "NUM_SHARDS:      ${NUM_SHARDS}"
echo "SHARD_INDEX:     ${SHARD_INDEX}"
echo "SIZE:            ${WIDTH}x${HEIGHT}"
echo "STEPS:           ${STEPS}"
echo "STRENGTH:        ${STRENGTH}"
echo "GUIDANCE_SCALE:  ${GUIDANCE_SCALE}"
echo "BATCH_SIZE:      ${BATCH_SIZE}"
echo "PROMPT:          ${PROMPT}"
echo "NEGATIVE_PROMPT: ${NEGATIVE_PROMPT}"
echo "========================================="

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Error: source dir not found: ${SOURCE_DIR}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${DISABLE_SAFETY_CHECKER}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-safety-checker)
fi
if [[ "${SD_SINGLE_FILE}" == "1" ]]; then
  EXTRA_ARGS+=(--single-file)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python tools/sd_img2img_underwater.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --steps "${STEPS}" \
  --strength "${STRENGTH}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --limit "${LIMIT}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --device cuda:0 \
  --save-manifest "${OUT_DIR}/sd_img2img_manifest${SHARD_TAG}.jsonl" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/sd_img2img_underwater_${SPLIT}${SHARD_TAG}.log"

echo
echo "Done."
echo "Output: ${OUT_DIR}"
