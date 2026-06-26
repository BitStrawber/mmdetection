#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/uwnr}"
GPU="${GPU:-2}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-100}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-30}"
N_CPU="${N_CPU:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "UWNR smoke test"
echo "========================================="
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "GPU:               ${GPU}"
echo "SMOKE_TRAIN_LIMIT: ${SMOKE_TRAIN_LIMIT}"
echo "SMOKE_VAL_LIMIT:   ${SMOKE_VAL_LIMIT}"
echo "N_CPU:             ${N_CPU}"
echo "OMP_THREADS:       ${OMP_NUM_THREADS}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

MODE=smoke \
METHODS="uwnr" \
SPLITS="train val" \
GPU="${GPU}" \
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT}" \
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT}" \
SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
  2>&1 | tee "${LOG_DIR}/prepare.log"

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
  N_CPU="${N_CPU}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  RUN_DEPTH=0 RUN_PREPARE=0 RUN_RUOD_REF=0 RUN_UWNR=1 RUN_RESTORE=1 \
  bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh \
    2>&1 | tee "${LOG_DIR}/${split}.log"
done

echo "UWNR smoke outputs:"
echo "  ${WORK_ROOT}/uwnr_ruod_ref/generated/train"
echo "  ${WORK_ROOT}/uwnr_ruod_ref/generated/val"
