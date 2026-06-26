#!/usr/bin/env bash
set -euo pipefail

# Smoke-test launcher for the ImageNet underwater synthesis baselines.
#
# Default models:
#   uwnr syreanet_synthesis cut watergan stable_diffusion_img2img
#
# Examples:
#   MODELS="uwnr stable_diffusion_img2img" GPU=2 bash scripts/exp_2/synthesis/run_synthesis_smoke_tests.sh
#   MODELS="cut" GPU=2 bash scripts/exp_2/synthesis/run_synthesis_smoke_tests.sh
#
# Recommended environments:
#   - uwnr/syreanet_synthesis/watergan prepare: /media/SSD1/conda_envs/syreanet or uwnr
#   - cut: /media/SSD1/conda_envs/cut
#   - stable_diffusion_img2img: /media/SSD1/conda_envs/stable_diffusion
#   - watergan train: /media/SSD1/conda_envs/watergan_tf1
#
# If running all models, use an environment that satisfies the current model,
# or run one MODEL group at a time after activating the proper env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke}"
MODELS="${MODELS:-uwnr syreanet_synthesis cut watergan stable_diffusion_img2img}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-${GPU}}"

SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"
SD_STEPS="${SD_STEPS:-20}"
SD_STRENGTH="${SD_STRENGTH:-0.35}"
SD_GUIDANCE_SCALE="${SD_GUIDANCE_SCALE:-5.0}"
SD_BATCH_SIZE="${SD_BATCH_SIZE:-1}"

CUT_EPOCHS="${CUT_EPOCHS:-2}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-1}"
WATERGAN_EPOCH="${WATERGAN_EPOCH:-2}"
WATERGAN_BATCH_SIZE="${WATERGAN_BATCH_SIZE:-4}"

mkdir -p "${LOG_DIR}"

run_prepare() {
  local models="$1"
  echo
  echo "========================================="
  echo "Smoke prepare on SSD: ${models}"
  echo "========================================="
  MODE=smoke \
  METHODS="${models}" \
  SPLITS="train val" \
  GPU="${GPU}" \
  SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
  SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
  SYN_ROOT="${SYN_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh
}

run_uwnr() {
  run_prepare "uwnr"
  echo
  echo "========================================="
  echo "Smoke generate: UWNR"
  echo "========================================="
  for split in train val; do
    SOURCE_DIR="${WORK_ROOT}/sources/uwnr/${split}" \
    DEPTH_DIR="${WORK_ROOT}/uwnr_ruod_ref/megadepth/${split}" \
    PREP_DIR="${WORK_ROOT}/uwnr_ruod_ref/prepared/${split}" \
    RUOD_REF_ROOT="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}" \
    FID_REF_DIR="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}_fid_resized" \
    FLAT_SAVE_DIR="${WORK_ROOT}/uwnr_ruod_ref/generated_flat/${split}" \
    RESTORE_DIR="${WORK_ROOT}/uwnr_ruod_ref/generated/${split}" \
    SPLIT="${split}" \
    LIMIT=0 \
    GPU="${GPU}" \
    RUN_DEPTH=0 RUN_PREPARE=0 RUN_RUOD_REF=0 RUN_UWNR=1 RUN_RESTORE=1 \
    bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh \
      2>&1 | tee "${LOG_DIR}/uwnr_${split}_smoke.log"
  done
}

run_syreanet_synthesis() {
  run_prepare "syreanet_synthesis"
  echo
  echo "========================================="
  echo "Smoke generate: SyreaNet synthesis"
  echo "========================================="
  for split in train val; do
    SOURCE_DIR="${WORK_ROOT}/sources/syreanet_synthesis/${split}" \
    DEPTH_DIR="${WORK_ROOT}/syreanet_synthesis/depth/${split}" \
    PREP_DIR="${WORK_ROOT}/syreanet_synthesis/prepared/${split}" \
    FLAT_SAVE_DIR="${WORK_ROOT}/syreanet_synthesis/generated_flat/${split}" \
    RESTORE_DIR="${WORK_ROOT}/syreanet_synthesis/generated/${split}" \
    SPLIT="${split}" \
    LIMIT=0 \
    GPU="${GPU}" \
    RUN_DEPTH=0 RUN_PREPARE=0 RUN_SYREANET=1 RUN_RESTORE=1 \
    bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh \
      2>&1 | tee "${LOG_DIR}/syreanet_synthesis_${split}_smoke.log"
  done
}

run_cut() {
  run_prepare "cut"
  local data_name="imagenet_ruod_cut_smoke_ssd"
  local data_root="${WORK_ROOT}/cut/datasets/${data_name}"
  echo
  echo "========================================="
  echo "Smoke train/generate: CUT"
  echo "========================================="
  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  EXP_NAME="${data_name}" \
  GPU_IDS="${GPU}" \
  BATCH_SIZE="${CUT_BATCH_SIZE}" \
  N_EPOCHS="${CUT_EPOCHS}" \
  N_EPOCHS_DECAY=0 \
  SAVE_EPOCH_FREQ=1 \
  bash scripts/exp_2/synthesis/run_cut_train.sh \
    2>&1 | tee "${LOG_DIR}/cut_train_smoke.log"

  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  EXP_NAME="${data_name}" \
  SPLIT=val \
  GPU_IDS="${GPU}" \
  NUM_TEST="${SMOKE_VAL_LIMIT}" \
  RESULTS_ROOT="${WORK_ROOT}/cut/results/${data_name}_val" \
  RESTORE_DIR="${WORK_ROOT}/cut/generated/val" \
  MANIFEST="${data_root}/manifests/testA_manifest.jsonl" \
  bash scripts/exp_2/synthesis/run_cut_generate.sh \
    2>&1 | tee "${LOG_DIR}/cut_generate_val_smoke.log"
}

run_watergan() {
  run_prepare "watergan"
  local data_name="imagenet_ruod_watergan_train_smoke_ssd"
  local data_root="${WORK_ROOT}/watergan/datasets/${data_name}"
  echo
  echo "========================================="
  echo "Smoke train: WaterGAN"
  echo "========================================="
  echo "Note: WaterGAN original repo is training-first. Full batch generation needs a generator-export step after smoke confirms checkpoints."
  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  GPU="${GPU}" \
  EPOCH="${WATERGAN_EPOCH}" \
  BATCH_SIZE="${WATERGAN_BATCH_SIZE}" \
  TRAIN_SIZE="${SMOKE_TRAIN_LIMIT}" \
  bash scripts/exp_2/synthesis/run_watergan_train.sh \
    2>&1 | tee "${LOG_DIR}/watergan_train_smoke.log"
}

run_stable_diffusion_img2img() {
  run_prepare "stable_diffusion_img2img"
  echo
  echo "========================================="
  echo "Smoke generate: Stable Diffusion img2img"
  echo "========================================="
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
      2>&1 | tee "${LOG_DIR}/sd_img2img_${split}_smoke.log"
  done
}

echo "========================================="
echo "Synthesis smoke launcher"
echo "========================================="
echo "MODELS:            ${MODELS}"
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "SMOKE_TRAIN_LIMIT: ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:   ${SMOKE_VAL_LIMIT}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

for model in ${MODELS}; do
  case "${model}" in
    uwnr) run_uwnr ;;
    syreanet_synthesis) run_syreanet_synthesis ;;
    cut) run_cut ;;
    watergan) run_watergan ;;
    stable_diffusion_img2img) run_stable_diffusion_img2img ;;
    *)
      echo "Error: unknown smoke model: ${model}" >&2
      exit 1
      ;;
  esac
done

echo
echo "All requested smoke tests completed."
echo "Logs: ${LOG_DIR}"
