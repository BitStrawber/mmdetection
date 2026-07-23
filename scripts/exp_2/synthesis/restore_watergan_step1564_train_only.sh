#!/usr/bin/env bash
set -euo pipefail

# Restore completed model-1564 flat train outputs directly into the final
# ImageNet class layout. By default exactly one Python restore process runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BASE_SHARD_ROOT="${BASE_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_step1564_official_base_48shards}"
FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_flat_48shards}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated_step1564_official_mat}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_official_mat_48shards}"
NUM_SHARDS="${NUM_SHARDS:-48}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OVERWRITE="${OVERWRITE:-0}"
RESTORE_WORKERS="${RESTORE_WORKERS:-16}"

FINAL_TRAIN="${FINAL_ROOT}/train"
RESTORE_LOG_ROOT="${LOG_ROOT}/train_restore_only"
PROGRESS_FILE="${RESTORE_LOG_ROOT}/progress.tsv"
SUMMARY_ROOT="${RESTORE_LOG_ROOT}/summaries"
DONE_ROOT="${RESTORE_LOG_ROOT}/done"
LOCK_FILE="${LOG_ROOT}/restore_train_only.lock"

[[ "${NUM_SHARDS}" -eq 48 ]] || {
  echo "Error: NUM_SHARDS must be 48" >&2
  exit 1
}
[[ "${OVERWRITE}" == 0 || "${OVERWRITE}" == 1 ]] || {
  echo "Error: OVERWRITE must be 0 or 1" >&2
  exit 1
}
[[ "${RESTORE_WORKERS}" -gt 0 && "${RESTORE_WORKERS}" -le "${NUM_SHARDS}" ]] || {
  echo "Error: RESTORE_WORKERS must be between 1 and ${NUM_SHARDS}" >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || {
  echo "Error: flock is required" >&2
  exit 1
}

mkdir -p "${FINAL_TRAIN}" "${RESTORE_LOG_ROOT}" "${SUMMARY_ROOT}" "${DONE_ROOT}"
exec 9>>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Error: another standalone train restore holds ${LOCK_FILE}" >&2
  exit 1
fi

printf 'shard\twritten\tskipped\tfinished_at\n' > "${PROGRESS_FILE}"
find "${DONE_ROOT}" -maxdepth 1 -type f -name 'shard*.done' -delete

count_images() {
  find "$1" -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

echo "============================================================"
echo "WaterGAN model-1564 standalone train restore"
echo "============================================================"
echo "BASE SHARDS: ${BASE_SHARD_ROOT}/train"
echo "FLAT ROOT:   ${FLAT_ROOT}/train"
echo "FINAL TRAIN: ${FINAL_TRAIN}"
echo "SHARDS:      ${NUM_SHARDS}"
echo "PY WORKERS:  ${RESTORE_WORKERS}"
echo "OVERWRITE:   ${OVERWRITE}"
echo "PROGRESS:    ${PROGRESS_FILE}"
echo "============================================================"

restore_one() {
  local index="$1" shard manifest results log summary
  local expected valid written skipped missing bad_names duplicates
  local -a args
  shard="shard${index}of${NUM_SHARDS}"
  manifest="${BASE_SHARD_ROOT}/train/${shard}/watergan_air_manifest.jsonl"
  results="${FLAT_ROOT}/train/${shard}"
  log="${RESTORE_LOG_ROOT}/${shard}.log"
  summary="${SUMMARY_ROOT}/${shard}.json"

  [[ -s "${manifest}" ]] || {
    echo "Error: missing manifest: ${manifest}" >&2
    exit 1
  }
  expected="$(wc -l < "${manifest}" | tr -d ' ')"
  valid="$(
    find "${results}" -maxdepth 1 -type f -name 'fake_*.png' -size +0c \
      2>/dev/null | wc -l | tr -d ' '
  )"
  [[ "${valid}" -eq "${expected}" ]] || {
    echo "Error: ${shard} has ${valid}/${expected} valid flat outputs" >&2
    exit 1
  }

  echo "restore ${shard}: ${valid}/${expected}"
  args=(
    --manifest "${manifest}"
    --results-dir "${results}"
    --out-dir "${FINAL_TRAIN}"
    --batch-size "${BATCH_SIZE}"
    --summary-path "${summary}"
  )
  [[ "${OVERWRITE}" == 1 ]] && args+=(--overwrite)
  python tools/restore_watergan_fake.py "${args[@]}" > "${log}" 2>&1

  read -r written skipped missing bad_names duplicates < <(
    python - "${summary}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding='utf-8'))
print(
    summary['written'],
    summary['skipped_existing'],
    summary['missing'],
    summary['bad_names'],
    summary['duplicate_indices'],
)
PY
  )
  [[ "${missing}" == 0 && "${bad_names}" == 0 && "${duplicates}" == 0 ]] || {
    echo "Error: invalid restore summary for ${shard}; see ${log}" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\n' \
    "${shard}" "${written}" "${skipped}" "$(date --iso-8601=seconds)" \
    >> "${PROGRESS_FILE}"
  : > "${DONE_ROOT}/${shard}.done"
}

failed=0
for ((batch_start=0; batch_start<NUM_SHARDS; batch_start+=RESTORE_WORKERS)); do
  pids=()
  labels=()
  batch_end=$((batch_start + RESTORE_WORKERS))
  (( batch_end > NUM_SHARDS )) && batch_end="${NUM_SHARDS}"

  for ((index=batch_start; index<batch_end; index++)); do
    restore_one "${index}" &
    pids+=("$!")
    labels+=("shard${index}of${NUM_SHARDS}")
  done

  for offset in "${!pids[@]}"; do
    if wait "${pids[$offset]}"; then
      echo "completed ${labels[$offset]}"
    else
      echo "FAILED ${labels[$offset]}" >&2
      failed=1
    fi
  done

  [[ "${failed}" == 0 ]] || {
    echo "Error: one or more standalone restore workers failed" >&2
    exit 1
  }
done

rm -f "${FINAL_TRAIN}/restore_watergan_fake_summary.json"

images="$(count_images "${FINAL_TRAIN}")"
classes="$(
  find "${FINAL_TRAIN}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | wc -l | tr -d ' '
)"
zero="$(find "${FINAL_TRAIN}" -type f -size 0 2>/dev/null | wc -l | tr -d ' ')"

[[ "${images}" -eq 250000 && "${classes}" -eq 1000 && "${zero}" -eq 0 ]] || {
  echo "Error: final train validation failed: images=${images}, classes=${classes}, zero=${zero}" >&2
  exit 1
}

cat <<EOF
============================================================
Standalone train restore complete
============================================================
images:  ${images}/250000
classes: ${classes}/1000
zero:    ${zero}
root:    ${FINAL_TRAIN}
============================================================
EOF
