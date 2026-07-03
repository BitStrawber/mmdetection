#!/usr/bin/env bash
set -euo pipefail

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
CUT_DIR="${CUT_DIR:-/home/fcp/xcx/exp_2/syn/contrastive-unpaired-translation}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_cut_smoke}"
DATA_ROOT="${DATA_ROOT:-${SYN_ROOT}/cut/datasets/${DATA_NAME}}"
EXP_NAME="${EXP_NAME:-${DATA_NAME}}"
GPU_IDS="${GPU_IDS:-2}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_SIZE="${LOAD_SIZE:-286}"
CROP_SIZE="${CROP_SIZE:-256}"
N_EPOCHS="${N_EPOCHS:-2}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-0}"
NUM_THREADS="${NUM_THREADS:-4}"
PRINT_FREQ="${PRINT_FREQ:-50}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-1}"
NO_HTML="${NO_HTML:-1}"

echo "========================================="
echo "Train CUT ImageNet -> RUOD"
echo "========================================="
echo "CUT_DIR:        ${CUT_DIR}"
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "EXP_NAME:       ${EXP_NAME}"
echo "GPU_IDS:        ${GPU_IDS}"
echo "CHECKPOINTS:    ${CHECKPOINTS_DIR}"
echo "BATCH_SIZE:     ${BATCH_SIZE}"
echo "LOAD/CROP:      ${LOAD_SIZE}/${CROP_SIZE}"
echo "EPOCHS:         ${N_EPOCHS}+${N_EPOCHS_DECAY}"
echo "========================================="

if [[ ! -d "${CUT_DIR}" ]]; then
  echo "Error: CUT repo not found: ${CUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Error: CUT dataset not found: ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${SYN_ROOT}/cut/logs"

echo "Dataset file counts:"
echo "  trainA: $(find "${DATA_ROOT}/trainA" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  trainB: $(find "${DATA_ROOT}/trainB" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  testA:  $(find "${DATA_ROOT}/testA" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  testB:  $(find "${DATA_ROOT}/testB" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "Training log: ${SYN_ROOT}/cut/logs/${EXP_NAME}_train.log"
echo

EXTRA_ARGS=()
if [[ "${NO_HTML}" == "1" ]]; then
  EXTRA_ARGS+=(--no_html)
fi

(
  cd "${CUT_DIR}"
  python train.py \
    --dataroot "${DATA_ROOT}" \
    --name "${EXP_NAME}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --CUT_mode CUT \
    --model cut \
    --dataset_mode unaligned \
    --direction AtoB \
    --gpu_ids "${GPU_IDS}" \
    --batch_size "${BATCH_SIZE}" \
    --load_size "${LOAD_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --n_epochs "${N_EPOCHS}" \
    --n_epochs_decay "${N_EPOCHS_DECAY}" \
    --display_id -1 \
    --print_freq "${PRINT_FREQ}" \
    --save_epoch_freq "${SAVE_EPOCH_FREQ}" \
    --num_threads "${NUM_THREADS}" \
    "${EXTRA_ARGS[@]}"
) 2>&1 | tee "${SYN_ROOT}/cut/logs/${EXP_NAME}_train.log"

echo
echo "Checkpoint dir: ${CUT_DIR}/${CHECKPOINTS_DIR}/${EXP_NAME}"
