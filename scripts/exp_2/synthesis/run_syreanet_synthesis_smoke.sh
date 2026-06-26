#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/syreanet_synthesis}"
GPU="${GPU:-2}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "SyreaNet synthesis smoke test"
echo "========================================="
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "SMOKE_TRAIN_LIMIT: ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:   ${SMOKE_VAL_LIMIT}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

MODE=smoke \
METHODS="syreanet_synthesis" \
SPLITS="train val" \
GPU="${GPU}" \
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

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
    2>&1 | tee "${LOG_DIR}/${split}.log"
done

echo "SyreaNet synthesis smoke outputs:"
echo "  ${WORK_ROOT}/syreanet_synthesis/generated/train"
echo "  ${WORK_ROOT}/syreanet_synthesis/generated/val"
