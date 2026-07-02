#!/usr/bin/env bash
set -euo pipefail

# Generate remaining Depth Anything V2 maps serially and continue on failures.
#
# This script is designed for large synthetic ImageNet source splits where only
# a few very large images may fail with OOM. Each task is run with the normal
# multi-GPU shard script, but a failed task does not stop the remaining tasks.
#
# Default task order:
#   uwdf/val
#   uwnr/train
#   uwnr/val
#   watergan/train
#   watergan/val
#
# Usage:
#   bash scripts/exp_2/synthesis/run_depthanything_v2_remaining_serial_continue.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
OUT_ROOT="${OUT_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR:-/home/fcp/xcx/exp_2/syn/Depth-Anything-V2}"
ENCODER="${ENCODER:-vitb}"
CHECKPOINT="${CHECKPOINT:-${DEPTHANYTHING_DIR}/checkpoints/depth_anything_v2_${ENCODER}.pth}"
GPU_IDS="${GPU_IDS:-2,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
INPUT_SIZE="${INPUT_SIZE:-518}"
OVERWRITE="${OVERWRITE:-0}"
INVERT="${INVERT:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/depthanything_v2_remaining_serial}"
TASKS="${TASKS:-uwdf:val uwnr:train uwnr:val watergan:train watergan:val}"

mkdir -p "${LOG_DIR}"

SUMMARY_TSV="${LOG_DIR}/serial_summary.tsv"
SUMMARY_JSONL="${LOG_DIR}/serial_summary.jsonl"
: > "${SUMMARY_TSV}"
: > "${SUMMARY_JSONL}"
printf "task\tmethod\tsplit\tstatus\tstarted_at\tfinished_at\tlog\n" >> "${SUMMARY_TSV}"

echo "========================================="
echo "Depth Anything V2 remaining serial run"
echo "========================================="
echo "SOURCE_ROOT:       ${SOURCE_ROOT}"
echo "OUT_ROOT:          ${OUT_ROOT}"
echo "DEPTHANYTHING_DIR: ${DEPTHANYTHING_DIR}"
echo "CHECKPOINT:        ${CHECKPOINT}"
echo "ENCODER:           ${ENCODER}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "PROCS_PER_GPU:     ${PROCS_PER_GPU}"
echo "INPUT_SIZE:        ${INPUT_SIZE}"
echo "OVERWRITE:         ${OVERWRITE}"
echo "INVERT:            ${INVERT}"
echo "TASKS:             ${TASKS}"
echo "LOG_DIR:           ${LOG_DIR}"
echo "========================================="

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Error: SOURCE_ROOT not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DEPTHANYTHING_DIR}" ]]; then
  echo "Error: DEPTHANYTHING_DIR not found: ${DEPTHANYTHING_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: CHECKPOINT not found: ${CHECKPOINT}" >&2
  exit 1
fi

count_images() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" \
    -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.JPEG' -o -iname '*.webp' \) \
    2>/dev/null | wc -l
}

count_depths() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" -type f -name '*.png' 2>/dev/null | wc -l
}

failed_tasks=()
completed_tasks=()

for task in ${TASKS}; do
  method="${task%%:*}"
  split="${task#*:}"
  task_name="${method}_${split}"
  started_at="$(date '+%F %T')"
  task_log="${LOG_DIR}/${task_name}.log"
  source_dir="${SOURCE_ROOT}/${method}/source/${split}"
  out_dir="${OUT_ROOT}/${method}/${split}"

  echo
  echo "-----------------------------------------"
  echo "Task: ${method}/${split}"
  echo "source: ${source_dir}"
  echo "out:    ${out_dir}"
  echo "log:    ${task_log}"
  echo "-----------------------------------------"
  echo "source count before: $(count_images "${source_dir}")"
  echo "depth count before:  $(count_depths "${out_dir}")"

  set +e
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  MODE=full \
  METHODS="${method}" \
  SPLITS="${split}" \
  ENCODER="${ENCODER}" \
  CHECKPOINT="${CHECKPOINT}" \
  DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  GPU_IDS="${GPU_IDS}" \
  PROCS_PER_GPU="${PROCS_PER_GPU}" \
  INPUT_SIZE="${INPUT_SIZE}" \
  OVERWRITE="${OVERWRITE}" \
  INVERT="${INVERT}" \
  LOG_DIR="${REPO_ROOT}/logs/depthanything_v2" \
  bash scripts/exp_2/synthesis/run_depthanything_v2_all_sources.sh \
    2>&1 | tee "${task_log}"
  status_code="${PIPESTATUS[0]}"
  set -e

  finished_at="$(date '+%F %T')"
  source_count="$(count_images "${source_dir}")"
  depth_count="$(count_depths "${out_dir}")"

  if [[ "${status_code}" == "0" ]]; then
    status="completed"
    completed_tasks+=("${task}")
  else
    status="failed_continue"
    failed_tasks+=("${task}")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${method}" "${split}" "${status}" "${started_at}" "${finished_at}" "${task_log}" \
    >> "${SUMMARY_TSV}"

  TASK="${task}" METHOD="${method}" SPLIT="${split}" STATUS="${status}" \
  STARTED_AT="${started_at}" FINISHED_AT="${finished_at}" TASK_LOG="${task_log}" \
  SOURCE_COUNT="${source_count}" DEPTH_COUNT="${depth_count}" STATUS_CODE="${status_code}" \
  python - <<'PY' >> "${SUMMARY_JSONL}"
import json
import os

print(json.dumps({
    "task": os.environ["TASK"],
    "method": os.environ["METHOD"],
    "split": os.environ["SPLIT"],
    "status": os.environ["STATUS"],
    "status_code": int(os.environ["STATUS_CODE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "log": os.environ["TASK_LOG"],
    "source_count": int(os.environ["SOURCE_COUNT"]),
    "depth_count": int(os.environ["DEPTH_COUNT"]),
}, ensure_ascii=False))
PY

  echo "source count after: ${source_count}"
  echo "depth count after:  ${depth_count}"
  echo "task status:        ${status}"
done

echo
echo "========================================="
echo "Depth Anything V2 remaining serial done"
echo "========================================="
echo "completed tasks: ${completed_tasks[*]:-<none>}"
echo "failed tasks:    ${failed_tasks[*]:-<none>}"
echo "summary tsv:     ${SUMMARY_TSV}"
echo "summary jsonl:   ${SUMMARY_JSONL}"
echo "========================================="

# Do not return a failing exit code for partial task failures: the goal is to
# keep the queue moving and leave sparse OOM cases for later focused repair.
exit 0
