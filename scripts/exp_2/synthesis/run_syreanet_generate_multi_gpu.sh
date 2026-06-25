#!/usr/bin/env bash
set -euo pipefail

# Launch SyreaNet generation with one independent shard per GPU.
#
# Examples:
#   SPLIT=train LIMIT=200 GPU_IDS=2,3,4,5 bash scripts/exp_2/synthesis/run_syreanet_generate_multi_gpu.sh
#   SPLIT=train LIMIT=0   GPU_IDS=2,3,4,5 bash scripts/exp_2/synthesis/run_syreanet_generate_multi_gpu.sh
#   SPLIT=val   LIMIT=0   GPU_IDS=2,3,4,5 bash scripts/exp_2/synthesis/run_syreanet_generate_multi_gpu.sh

SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-200}"
GPU_IDS="${GPU_IDS:-2,3,4,5}"
LOG_DIR="${LOG_DIR:-logs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_SCRIPT="${SCRIPT_DIR}/run_syreanet_generate.sh"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${#GPUS[@]}"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "Error: GPU_IDS is empty." >&2
  exit 1
fi
if [[ ! -f "${WORK_SCRIPT}" ]]; then
  echo "Error: worker script not found: ${WORK_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "SyreaNet multi-GPU generation launcher"
echo "========================================="
echo "SPLIT:      ${SPLIT}"
echo "LIMIT:      ${LIMIT}"
echo "GPU_IDS:    ${GPU_IDS}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "LOG_DIR:    ${LOG_DIR}"
echo "========================================="
echo

pids=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  log="${LOG_DIR}/syreanet_${SPLIT}_shard${idx}of${NUM_SHARDS}_launcher.log"
  echo "Start shard ${idx}/${NUM_SHARDS} on GPU ${gpu}"
  (
    SPLIT="${SPLIT}" \
    LIMIT="${LIMIT}" \
    GPU="${gpu}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    SHARD_INDEX="${idx}" \
    bash "${WORK_SCRIPT}"
  ) > "${log}" 2>&1 &
  pids+=("$!")
  echo "  pid: ${pids[-1]}"
  echo "  log: ${log}"
done

echo
echo "All shards launched. Waiting..."

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "One or more SyreaNet shards failed. Check logs in ${LOG_DIR}." >&2
  exit 1
fi

echo "All SyreaNet shards finished."
