#!/usr/bin/env bash
set -euo pipefail

# Generate full ImageNet train split with the existing CUT 5epoch checkpoint.
# Output organization follows the existing convention:
#   /media/HDD1/XCX/exp_2/synthetic_imagenet/cut/generated/train/<synset>/<image>.png
#
# Example:
#   conda activate /media/SSD1/conda_envs/cut
#   GPU_IDS="3,4" bash scripts/exp_2/synthesis/run_cut_5epoch_generate_train_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
CUT_DIR="${CUT_DIR:-/home/fcp/xcx/exp_2/syn/contrastive-unpaired-translation}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${CUT_DIR}/checkpoints}"
EXP_NAME="${EXP_NAME:-imagenet_ruod_cut_full_bs2_5epoch_gpu5}"
GPU_IDS="${GPU_IDS:-3,4}"
NUM_TEST="${NUM_TEST:-250000}"
COPY_MODE="${COPY_MODE:-copy}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

LOAD_SIZE="${LOAD_SIZE:-256}"
CROP_SIZE="${CROP_SIZE:-256}"
PREPROCESS="${PREPROCESS:-resize_and_crop}"
EPOCH="${EPOCH:-latest}"

DATA_NAME="${DATA_NAME:-imagenet_ruod_cut_full_ssd_train_as_test}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/cut/datasets/${DATA_NAME}}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORK_ROOT}/cut/results/${EXP_NAME}_train_full}"
RESTORE_DIR="${RESTORE_DIR:-${SYN_ROOT}/cut/generated/train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/cut_5epoch_train_full}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/testA_manifest.jsonl}"

mkdir -p "${LOG_DIR}"

count_images() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

cat <<EOF
=========================================
CUT 5epoch full train generation
=========================================
SYN_ROOT:        ${SYN_ROOT}
SOURCE_ROOT:     ${SOURCE_ROOT}
WORK_ROOT:       ${WORK_ROOT}
CUT_DIR:         ${CUT_DIR}
CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}
EXP_NAME:        ${EXP_NAME}
GPU_IDS:         ${GPU_IDS}
NUM_TEST:        ${NUM_TEST}
DATA_ROOT:       ${DATA_ROOT}
RESULTS_ROOT:    ${RESULTS_ROOT}
RESTORE_DIR:     ${RESTORE_DIR}
MANIFEST:        ${MANIFEST}
PREPROCESS:      ${PREPROCESS}
LOAD/CROP:       ${LOAD_SIZE}/${CROP_SIZE}
RESET_OUTPUTS:   ${RESET_OUTPUTS}
=========================================
EOF

check_dir "${CUT_DIR}" "CUT_DIR"
check_dir "${CHECKPOINTS_DIR}/${EXP_NAME}" "CUT 5epoch checkpoint directory"
check_dir "${SOURCE_ROOT}/cut/source/train" "CUT train source"
check_dir "${RUOD_REF_SRC}" "RUOD reference source"

if [[ "${RESET_OUTPUTS}" == "1" ]]; then
  echo "Reset CUT full train outputs"
  rm -rf "${RESULTS_ROOT}" "${RESTORE_DIR}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo
  echo "Prepare CUT train-as-test dataset: ${DATA_ROOT}"
  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  TRAIN_A_SOURCE="${SOURCE_ROOT}/cut/source/train" \
  TEST_A_SOURCE="${SOURCE_ROOT}/cut/source/train" \
  TRAIN_B_SOURCE="${RUOD_REF_SRC}" \
  TEST_B_SOURCE="${RUOD_REF_SRC}" \
  TRAIN_A_LIMIT=0 TEST_A_LIMIT=0 TRAIN_B_LIMIT=0 TEST_B_LIMIT=1000 \
  LINK_MODE="${COPY_MODE}" \
  OVERWRITE=1 \
  bash scripts/exp_2/synthesis/run_cut_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/prepare_train_as_test.log"
else
  echo "Reuse existing CUT manifest: ${MANIFEST}"
fi

DATA_NAME="${DATA_NAME}" \
DATA_ROOT="${DATA_ROOT}" \
EXP_NAME="${EXP_NAME}" \
SPLIT=train \
GPU_IDS="${GPU_IDS}" \
NUM_TEST="${NUM_TEST}" \
EPOCH="${EPOCH}" \
CHECKPOINTS_DIR="${CHECKPOINTS_DIR}" \
LOAD_SIZE="${LOAD_SIZE}" \
CROP_SIZE="${CROP_SIZE}" \
PREPROCESS="${PREPROCESS}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
RESTORE_DIR="${RESTORE_DIR}" \
MANIFEST="${MANIFEST}" \
bash scripts/exp_2/synthesis/run_cut_generate.sh \
  2>&1 | tee "${LOG_DIR}/generate_train.log"

cat <<EOF
=========================================
CUT 5epoch generation done
=========================================
RESTORE_DIR: ${RESTORE_DIR}
restored_images: $(count_images "${RESTORE_DIR}")
expected_train: 250000
=========================================
EOF