#!/usr/bin/env bash
set -euo pipefail

# Full launcher for ImageNet underwater synthesis baselines.
#
# This script assumes SSD-side preparation is desired and will call
# prepare_synthesis_ssd_inputs.sh for each selected model before generation.
#
# Examples:
#   MODELS="uwnr syreanet_synthesis stable_diffusion_img2img" GPU_IDS=2,3,4,5 bash scripts/exp_2/synthesis/run_synthesis_full_generation.sh
#   MODELS="cut" GPU=2 bash scripts/exp_2/synthesis/run_synthesis_full_generation.sh
#
# WaterGAN note:
#   The original WaterGAN repo is training-first and does not yet have a clean
#   batch ImageNet inference wrapper in this repo. This launcher prepares and
#   trains WaterGAN at full scale, then stops with checkpoint/sample locations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full}"
MODELS="${MODELS:-uwnr syreanet_synthesis cut watergan stable_diffusion_img2img}"
SPLITS="${SPLITS:-train val}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

SD_STEPS="${SD_STEPS:-20}"
SD_STRENGTH="${SD_STRENGTH:-0.35}"
SD_GUIDANCE_SCALE="${SD_GUIDANCE_SCALE:-5.0}"
SD_BATCH_SIZE="${SD_BATCH_SIZE:-1}"

CUT_EPOCHS="${CUT_EPOCHS:-100}"
CUT_EPOCHS_DECAY="${CUT_EPOCHS_DECAY:-100}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-1}"
CUT_NUM_TEST="${CUT_NUM_TEST:-100000000}"

WATERGAN_EPOCH="${WATERGAN_EPOCH:-26}"
WATERGAN_BATCH_SIZE="${WATERGAN_BATCH_SIZE:-4}"
WATERGAN_TRAIN_SIZE="${WATERGAN_TRAIN_SIZE:-0}"

mkdir -p "${LOG_DIR}"

run_prepare() {
  local models="$1"
  echo
  echo "========================================="
  echo "Full prepare on SSD: ${models}"
  echo "========================================="
  MODE=full \
  METHODS="${models}" \
  SPLITS="${SPLITS}" \
  GPU="${GPU}" \
  FULL_LIMIT=0 \
  SYN_ROOT="${SYN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh
}

run_uwnr() {
  run_prepare "uwnr"
  echo
  echo "========================================="
  echo "Full generate: UWNR"
  echo "========================================="
  for split in ${SPLITS}; do
    SOURCE_DIR="${WORK_ROOT}/sources/uwnr/${split}" \
    DEPTH_DIR="${WORK_ROOT}/uwnr_ruod_ref/megadepth/${split}" \
    PREP_DIR="${WORK_ROOT}/uwnr_ruod_ref/prepared/${split}" \
    RUOD_REF_ROOT="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}" \
    FID_REF_DIR="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}_fid_resized" \
    FLAT_SAVE_DIR="${WORK_ROOT}/uwnr_ruod_ref/generated_flat/${split}" \
    RESTORE_DIR="${SYN_ROOT}/uwnr_ruod_ref/generated/${split}" \
    SPLIT="${split}" \
    LIMIT=0 \
    GPU_IDS="${GPU_IDS}" \
    PROCS_PER_GPU="${PROCS_PER_GPU}" \
    RUN_DEPTH=0 RUN_PREPARE=0 RUN_RUOD_REF=0 RUN_UWNR=1 RUN_RESTORE=1 \
    bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate_multi_gpu.sh \
      2>&1 | tee "${LOG_DIR}/uwnr_${split}_full.log"
  done
}

run_syreanet_synthesis() {
  run_prepare "syreanet_synthesis"
  echo
  echo "========================================="
  echo "Full generate: SyreaNet synthesis"
  echo "========================================="
  for split in ${SPLITS}; do
    SOURCE_DIR="${WORK_ROOT}/sources/syreanet_synthesis/${split}" \
    DEPTH_DIR="${WORK_ROOT}/syreanet_synthesis/depth/${split}" \
    PREP_DIR="${WORK_ROOT}/syreanet_synthesis/prepared/${split}" \
    FLAT_SAVE_DIR="${WORK_ROOT}/syreanet_synthesis/generated_flat/${split}" \
    RESTORE_DIR="${SYN_ROOT}/syreanet_synthesis/generated/${split}" \
    SPLIT="${split}" \
    LIMIT=0 \
    GPU_IDS="${GPU_IDS}" \
    PROCS_PER_GPU="${PROCS_PER_GPU}" \
    RUN_DEPTH=0 RUN_PREPARE=0 RUN_SYREANET=1 RUN_RESTORE=1 \
    bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate_multi_gpu.sh \
      2>&1 | tee "${LOG_DIR}/syreanet_synthesis_${split}_full.log"
  done
}

run_cut() {
  run_prepare "cut"
  local data_name="imagenet_ruod_cut_full_ssd"
  local data_root="${WORK_ROOT}/cut/datasets/${data_name}"
  local train_gen_name="${data_name}_train_as_test"
  local train_gen_root="${WORK_ROOT}/cut/datasets/${train_gen_name}"
  echo
  echo "========================================="
  echo "Full train/generate: CUT"
  echo "========================================="
  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  EXP_NAME="${data_name}" \
  GPU_IDS="${GPU}" \
  BATCH_SIZE="${CUT_BATCH_SIZE}" \
  N_EPOCHS="${CUT_EPOCHS}" \
  N_EPOCHS_DECAY="${CUT_EPOCHS_DECAY}" \
  SAVE_EPOCH_FREQ=10 \
  bash scripts/exp_2/synthesis/run_cut_train.sh \
    2>&1 | tee "${LOG_DIR}/cut_train_full.log"

  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  EXP_NAME="${data_name}" \
  SPLIT=val \
  GPU_IDS="${GPU}" \
  NUM_TEST="${CUT_NUM_TEST}" \
  RESULTS_ROOT="${WORK_ROOT}/cut/results/${data_name}_val" \
  RESTORE_DIR="${SYN_ROOT}/cut/generated/val" \
  MANIFEST="${data_root}/manifests/testA_manifest.jsonl" \
  bash scripts/exp_2/synthesis/run_cut_generate.sh \
    2>&1 | tee "${LOG_DIR}/cut_generate_val_full.log"

  echo
  echo "Prepare CUT train-as-testA dataset for train split generation"
  DATA_NAME="${train_gen_name}" \
  DATA_ROOT="${train_gen_root}" \
  TRAIN_A_SOURCE="${WORK_ROOT}/sources/cut/train" \
  TEST_A_SOURCE="${WORK_ROOT}/sources/cut/train" \
  TRAIN_B_SOURCE="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}" \
  TEST_B_SOURCE="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}" \
  TRAIN_A_LIMIT=0 TEST_A_LIMIT=0 TRAIN_B_LIMIT=0 TEST_B_LIMIT=1000 \
  LINK_MODE="${COPY_MODE:-copy}" \
  OVERWRITE=1 \
  bash scripts/exp_2/synthesis/run_cut_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/cut_prepare_train_as_test_full.log"

  DATA_NAME="${train_gen_name}" \
  DATA_ROOT="${train_gen_root}" \
  EXP_NAME="${data_name}" \
  SPLIT=train \
  GPU_IDS="${GPU}" \
  NUM_TEST="${CUT_NUM_TEST}" \
  RESULTS_ROOT="${WORK_ROOT}/cut/results/${data_name}_train" \
  RESTORE_DIR="${SYN_ROOT}/cut/generated/train" \
  MANIFEST="${train_gen_root}/manifests/testA_manifest.jsonl" \
  bash scripts/exp_2/synthesis/run_cut_generate.sh \
    2>&1 | tee "${LOG_DIR}/cut_generate_train_full.log"
}

run_watergan() {
  run_prepare "watergan"
  local data_name="imagenet_ruod_watergan_train_full_ssd"
  local data_root="${WORK_ROOT}/watergan/datasets/${data_name}"
  echo
  echo "========================================="
  echo "Full train: WaterGAN"
  echo "========================================="
  echo "Note: WaterGAN batch generation is not launched here because the original TF1 repo needs a separate generator-export/inference wrapper."
  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  GPU="${GPU}" \
  EPOCH="${WATERGAN_EPOCH}" \
  BATCH_SIZE="${WATERGAN_BATCH_SIZE}" \
  TRAIN_SIZE="${WATERGAN_TRAIN_SIZE}" \
  bash scripts/exp_2/synthesis/run_watergan_train.sh \
    2>&1 | tee "${LOG_DIR}/watergan_train_full.log"
}

run_stable_diffusion_img2img() {
  run_prepare "stable_diffusion_img2img"
  echo
  echo "========================================="
  echo "Full generate: Stable Diffusion img2img"
  echo "========================================="
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
    bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate_multi_gpu.sh \
      2>&1 | tee "${LOG_DIR}/sd_img2img_${split}_full.log"
  done
}

echo "========================================="
echo "Synthesis full launcher"
echo "========================================="
echo "MODELS:          ${MODELS}"
echo "SPLITS:          ${SPLITS}"
echo "SYN_ROOT:        ${SYN_ROOT}"
echo "WORK_ROOT:       ${WORK_ROOT}"
echo "GPU:             ${GPU}"
echo "GPU_IDS:         ${GPU_IDS}"
echo "PROCS_PER_GPU:   ${PROCS_PER_GPU}"
echo "LOG_DIR:         ${LOG_DIR}"
echo "========================================="

for model in ${MODELS}; do
  case "${model}" in
    uwnr) run_uwnr ;;
    syreanet_synthesis) run_syreanet_synthesis ;;
    cut) run_cut ;;
    watergan) run_watergan ;;
    stable_diffusion_img2img) run_stable_diffusion_img2img ;;
    *)
      echo "Error: unknown full model: ${model}" >&2
      exit 1
      ;;
  esac
done

echo
echo "All requested full jobs completed."
echo "Logs: ${LOG_DIR}"
