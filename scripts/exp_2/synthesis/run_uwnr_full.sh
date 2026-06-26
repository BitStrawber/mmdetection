#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/uwnr}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
SPLITS="${SPLITS:-train val}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "UWNR full generation"
echo "========================================="
echo "SYN_ROOT:      ${SYN_ROOT}"
echo "WORK_ROOT:     ${WORK_ROOT}"
echo "GPU:           ${GPU}"
echo "GPU_IDS:       ${GPU_IDS}"
echo "PROCS_PER_GPU: ${PROCS_PER_GPU}"
echo "SPLITS:        ${SPLITS}"
echo "LOG_DIR:       ${LOG_DIR}"
echo "========================================="

MODE=full \
METHODS="uwnr" \
SPLITS="${SPLITS}" \
GPU="${GPU}" \
FULL_LIMIT=0 \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

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
    2>&1 | tee "${LOG_DIR}/${split}.log"
done

echo "UWNR full outputs:"
echo "  ${SYN_ROOT}/uwnr_ruod_ref/generated/train"
echo "  ${SYN_ROOT}/uwnr_ruod_ref/generated/val"
