#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/syreanet_synthesis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SYN_ROOT}/syreanet_synthesis/generated}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
SPLITS="${SPLITS:-train val}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "SyreaNet synthesis full generation"
echo "========================================="
echo "SYN_ROOT:      ${SYN_ROOT}"
echo "WORK_ROOT:     ${WORK_ROOT}"
echo "OUTPUT_ROOT:   ${OUTPUT_ROOT}"
echo "SOURCE_ROOT:   ${SOURCE_ROOT:-<prepare to SSD first>}"
echo "GPU:           ${GPU}"
echo "GPU_IDS:       ${GPU_IDS}"
echo "PROCS_PER_GPU: ${PROCS_PER_GPU}"
echo "SPLITS:        ${SPLITS}"
echo "OMP_THREADS:   ${OMP_NUM_THREADS}"
echo "LOG_DIR:       ${LOG_DIR}"
echo "========================================="

if [[ -z "${SOURCE_ROOT}" ]]; then
  MODE=full \
  METHODS="syreanet_synthesis" \
  SPLITS="${SPLITS}" \
  GPU="${GPU}" \
  FULL_LIMIT=0 \
  SOURCE_ONLY=1 \
  SYN_ROOT="${SYN_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh \
    2>&1 | tee "${LOG_DIR}/prepare.log"
else
  echo "Skip prepare_synthesis_ssd_inputs.sh because SOURCE_ROOT is set." | tee "${LOG_DIR}/prepare.log"
fi

for split in ${SPLITS}; do
  IFS=', ' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
  EXPANDED_GPU_IDS=()
  for gpu_id in "${GPU_ARRAY[@]}"; do
    for _ in $(seq 1 "${PROCS_PER_GPU}"); do
      EXPANDED_GPU_IDS+=("${gpu_id}")
    done
  done
  NUM_SHARDS="${#EXPANDED_GPU_IDS[@]}"
  echo "Launch SyreaNet synthesis ${split}: ${NUM_SHARDS} shards on ${EXPANDED_GPU_IDS[*]}" | tee "${LOG_DIR}/${split}.log"

  pids=()
  for idx in "${!EXPANDED_GPU_IDS[@]}"; do
    gpu_id="${EXPANDED_GPU_IDS[$idx]}"
    shard_tag="_shard${idx}of${NUM_SHARDS}"
    shard_log="${LOG_DIR}/${split}${shard_tag}.log"
    if [[ -n "${SOURCE_ROOT}" ]]; then
      source_dir="${SOURCE_ROOT}/${split}"
    else
      source_dir="${WORK_ROOT}/sources/syreanet_synthesis/${split}"
    fi
    echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu_id}; log=${shard_log}" | tee -a "${LOG_DIR}/${split}.log"
    (
      SOURCE_DIR="${source_dir}" \
      DEPTH_DIR="${WORK_ROOT}/syreanet_synthesis/depth/${split}${shard_tag}" \
      PREP_DIR="${WORK_ROOT}/syreanet_synthesis/prepared/${split}${shard_tag}" \
      FLAT_SAVE_DIR="${WORK_ROOT}/syreanet_synthesis/generated_flat/${split}${shard_tag}" \
      RESTORE_DIR="${OUTPUT_ROOT}/${split}" \
      SPLIT="${split}" \
      LIMIT=0 \
      GPU="${gpu_id}" \
      NUM_SHARDS="${NUM_SHARDS}" \
      SHARD_INDEX="${idx}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
      RUN_DEPTH=1 RUN_PREPARE=1 RUN_SYREANET=1 RUN_RESTORE=1 \
      bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh
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
    echo "SyreaNet synthesis ${split} failed. Check ${LOG_DIR}/${split}_shard* logs." >&2
    exit 1
  fi
done

echo "SyreaNet synthesis full outputs:"
for split in ${SPLITS}; do
  echo "  ${OUTPUT_ROOT}/${split}"
done
