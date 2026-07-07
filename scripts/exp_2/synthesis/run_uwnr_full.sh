#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/uwnr}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwnr/source}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwnr}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
SPLITS="${SPLITS:-train val}"
TEST_SIZE="${TEST_SIZE:-256}"
N_CPU="${N_CPU:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "UWNR full generation"
echo "========================================="
echo "SYN_ROOT:      ${SYN_ROOT}"
echo "WORK_ROOT:     ${WORK_ROOT}"
echo "SOURCE_ROOT:   ${SOURCE_ROOT}"
echo "DEPTH_ROOT:    ${DEPTH_ROOT}"
echo "GPU:           ${GPU}"
echo "GPU_IDS:       ${GPU_IDS}"
echo "PROCS_PER_GPU: ${PROCS_PER_GPU}"
echo "SPLITS:        ${SPLITS}"
echo "TEST_SIZE:     ${TEST_SIZE}"
echo "N_CPU:         ${N_CPU}"
echo "OMP_THREADS:   ${OMP_NUM_THREADS}"
echo "LOG_DIR:       ${LOG_DIR}"
echo "RESET_OUTPUTS: ${RESET_OUTPUTS}"
echo "========================================="

echo "Skip prepare_synthesis_ssd_inputs.sh; using SOURCE_ROOT and precomputed DEPTH_ROOT." | tee "${LOG_DIR}/prepare.log"

for split in ${SPLITS}; do
  if [[ "${RESET_OUTPUTS}" == "1" ]]; then
    echo "Reset UWNR outputs for ${split}" | tee -a "${LOG_DIR}/prepare.log"
    rm -rf \
      "${WORK_ROOT}/uwnr_ruod_ref/prepared/${split}"_shard* \
      "${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}"_shard* \
      "${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}"_shard*_fid_resized \
      "${WORK_ROOT}/uwnr_ruod_ref/generated_flat/${split}"_shard* \
      "${SYN_ROOT}/uwnr_ruod_ref/generated/${split}"
  fi

  IFS=', ' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
  EXPANDED_GPU_IDS=()
  for gpu_id in "${GPU_ARRAY[@]}"; do
    for _ in $(seq 1 "${PROCS_PER_GPU}"); do
      EXPANDED_GPU_IDS+=("${gpu_id}")
    done
  done
  NUM_SHARDS="${#EXPANDED_GPU_IDS[@]}"
  echo "Launch UWNR ${split}: ${NUM_SHARDS} shards on ${EXPANDED_GPU_IDS[*]}" | tee "${LOG_DIR}/${split}.log"

  pids=()
  for idx in "${!EXPANDED_GPU_IDS[@]}"; do
    gpu_id="${EXPANDED_GPU_IDS[$idx]}"
    shard_tag="_shard${idx}of${NUM_SHARDS}"
    shard_log="${LOG_DIR}/${split}${shard_tag}.log"
    echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu_id}; log=${shard_log}" | tee -a "${LOG_DIR}/${split}.log"
    (
      SOURCE_DIR="${SOURCE_ROOT}/${split}" \
      DEPTH_DIR="${DEPTH_ROOT}/${split}" \
      PREP_DIR="${WORK_ROOT}/uwnr_ruod_ref/prepared/${split}${shard_tag}" \
      RUOD_REF_ROOT="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}${shard_tag}" \
      FID_REF_DIR="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}${shard_tag}_fid_resized" \
      FLAT_SAVE_DIR="${WORK_ROOT}/uwnr_ruod_ref/generated_flat/${split}${shard_tag}" \
      RESTORE_DIR="${SYN_ROOT}/uwnr_ruod_ref/generated/${split}" \
      SPLIT="${split}" \
      LIMIT=0 \
      GPU="${gpu_id}" \
      NUM_SHARDS="${NUM_SHARDS}" \
      SHARD_INDEX="${idx}" \
      TEST_SIZE="${TEST_SIZE}" \
      N_CPU="${N_CPU}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
      RUN_DEPTH=0 RUN_PREPARE=1 RUN_RUOD_REF=1 RUN_UWNR=1 RUN_RESTORE=1 \
      bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh
    ) > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "UWNR ${split} failed. Check ${LOG_DIR}/${split}_shard* logs." >&2
    exit 1
  fi
done

echo "UWNR full outputs:"
echo "  ${SYN_ROOT}/uwnr_ruod_ref/generated/train"
echo "  ${SYN_ROOT}/uwnr_ruod_ref/generated/val"
