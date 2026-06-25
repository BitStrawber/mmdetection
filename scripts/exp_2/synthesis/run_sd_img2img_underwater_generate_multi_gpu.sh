#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for the Stable Diffusion img2img underwater baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-train}"
GPU_IDS_RAW="${GPU_IDS:-2,3,4,5}"
LIMIT="${LIMIT:-0}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "Error: no GPU ids provided. Set GPU_IDS=2,3,4,5" >&2
  exit 1
fi
if [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "Error: PROCS_PER_GPU must be >= 1" >&2
  exit 1
fi

EXPANDED_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  for _ in $(seq 1 "${PROCS_PER_GPU}"); do
    EXPANDED_GPU_IDS+=("${gpu}")
  done
done

NUM_SHARDS="${NUM_SHARDS:-${#EXPANDED_GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -ne "${#EXPANDED_GPU_IDS[@]}" ]]; then
  echo "Error: NUM_SHARDS (${NUM_SHARDS}) must equal launched process count (${#EXPANDED_GPU_IDS[@]})." >&2
  echo "GPU_IDS=${GPU_IDS_RAW}, PROCS_PER_GPU=${PROCS_PER_GPU}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "SD img2img underwater multi-GPU launcher"
echo "========================================="
echo "SPLIT:       ${SPLIT}"
echo "LIMIT:       ${LIMIT}"
echo "GPU_IDS:     ${GPU_IDS[*]}"
echo "PROCS/GPU:   ${PROCS_PER_GPU}"
echo "LAUNCH_GPUS: ${EXPANDED_GPU_IDS[*]}"
echo "NUM_SHARDS:  ${NUM_SHARDS}"
echo "LOG_DIR:     ${LOG_DIR}"
echo "========================================="

pids=()
for idx in "${!EXPANDED_GPU_IDS[@]}"; do
  gpu="${EXPANDED_GPU_IDS[$idx]}"
  shard_log="${LOG_DIR}/sd_img2img_underwater_${SPLIT}_shard${idx}of${NUM_SHARDS}_launcher.log"
  echo "Launch shard ${idx}/${NUM_SHARDS} on GPU ${gpu}; log=${shard_log}"
  (
    SPLIT="${SPLIT}" \
    LIMIT="${LIMIT}" \
    GPU="${gpu}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    SHARD_INDEX="${idx}" \
    bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh
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
  echo "At least one SD img2img shard failed. Check ${LOG_DIR}/sd_img2img_underwater_${SPLIT}_shard*of${NUM_SHARDS}_launcher.log" >&2
  exit 1
fi

echo "All SD img2img shards completed."
