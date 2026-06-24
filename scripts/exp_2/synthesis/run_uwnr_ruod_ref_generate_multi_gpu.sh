#!/usr/bin/env bash
set -euo pipefail

# Launch UWNR+RUOD-reference generation shards in parallel. Each GPU gets one
# shard of the sorted ImageNet source list. The underlying generation script
# keeps official UWNR inference logic and restores each shard into the shared
# generated/<split>/<synset>/ tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-train}"
GPU_IDS_RAW="${GPU_IDS:-0,1,2,3}"
LIMIT="${LIMIT:-0}"

IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "Error: no GPU ids provided. Set GPU_IDS=0,1,2,3" >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Error: NUM_SHARDS (${NUM_SHARDS}) must equal GPU count (${#GPU_IDS[@]}) for this launcher." >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

echo "========================================="
echo "UWNR + RUOD reference multi-GPU launcher"
echo "========================================="
echo "SPLIT:      ${SPLIT}"
echo "GPU_IDS:    ${GPU_IDS[*]}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "LIMIT:      ${LIMIT}"
echo "LOG_DIR:    ${LOG_DIR}"
echo "========================================="

pids=()
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  shard_log="${LOG_DIR}/uwnr_ruod_ref_${SPLIT}_shard${idx}of${NUM_SHARDS}_launcher.log"
  echo "Launch shard ${idx}/${NUM_SHARDS} on GPU ${gpu}; log=${shard_log}"
  (
    SPLIT="${SPLIT}" \
    LIMIT="${LIMIT}" \
    GPU="${gpu}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    SHARD_INDEX="${idx}" \
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
  echo "At least one shard failed. Check ${LOG_DIR}/uwnr_ruod_ref_${SPLIT}_shard*of${NUM_SHARDS}_launcher.log" >&2
  exit 1
fi

echo "All shards completed."
echo "Restored output:"
echo "/media/HDD1/XCX/exp_2/synthetic_imagenet/uwnr_ruod_ref/generated/${SPLIT}"
