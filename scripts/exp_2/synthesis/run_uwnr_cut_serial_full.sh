#!/usr/bin/env bash
set -uo pipefail

# Serially run full UWNR and CUT synthesis. A failed task is recorded but does
# not stop the next task from running.
#
# Typical use:
#   bash scripts/exp_2/synthesis/run_uwnr_cut_serial_full.sh \
#     2>&1 | tee logs/synthesis_full/uwnr_cut_serial_full.log
#
# Common overrides:
#   TASKS="uwnr cut" GPU_IDS="3,4,5,6,7" RESET_OUTPUTS=1 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"

UWNR_DIR="${UWNR_DIR:-/home/fcp/xcx/exp_2/syn/UWNR}"
UWNR_CKPT="${UWNR_CKPT:-${UWNR_DIR}/checkpoints/uwnr_pretrained.pk}"
CUT_DIR="${CUT_DIR:-/home/fcp/xcx/exp_2/syn/contrastive-unpaired-translation}"

CONDA_UWNR_ENV="${CONDA_UWNR_ENV:-/media/SSD1/conda_envs/uwnr}"
CONDA_CUT_ENV="${CONDA_CUT_ENV:-/media/SSD1/conda_envs/cut}"
CONDA_SH="${CONDA_SH:-}"

GPU_IDS="${GPU_IDS:-3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"
TASKS="${TASKS:-uwnr cut}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_FINAL_CHECK="${RUN_FINAL_CHECK:-1}"

N_CPU="${N_CPU:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CUT_EPOCHS="${CUT_EPOCHS:-100}"
CUT_EPOCHS_DECAY="${CUT_EPOCHS_DECAY:-100}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-5}"
CUT_NUM_THREADS="${CUT_NUM_THREADS:-12}"
CUT_NUM_TEST="${CUT_NUM_TEST:-100000000}"
COPY_MODE="${COPY_MODE:-copy}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/uwnr_cut_serial_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_ROOT}"

STATUS_TSV="${LOG_ROOT}/status.tsv"
: > "${STATUS_TSV}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

count_img() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
    -iname '*.bmp' -o -iname '*.webp' -o -iname '*.JPEG' \
  \) 2>/dev/null | wc -l
}

first_gpu_id() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2086
  set -- ${raw}
  echo "$1"
}

resolve_conda_sh() {
  if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
    echo "${CONDA_SH}"
    return 0
  fi
  local candidate=""
  candidate="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${candidate}" && -f "${candidate}/etc/profile.d/conda.sh" ]]; then
    echo "${candidate}/etc/profile.d/conda.sh"
    return 0
  fi
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/media/SSD1/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

activate_env() {
  local env_path="$1"
  local conda_sh_path
  if ! conda_sh_path="$(resolve_conda_sh)"; then
    echo "Error: cannot find conda.sh. Set CONDA_SH=/path/to/conda.sh" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${conda_sh_path}"
  conda activate "${env_path}"
}

record_status() {
  local task="$1"
  local status="$2"
  local log_path="$3"
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "${task}" "${status}" "${log_path}" | tee -a "${STATUS_TSV}"
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    echo "[OK]      ${label}: ${path}"
    return 0
  fi
  echo "[MISSING] ${label}: ${path}" >&2
  return 1
}

preflight() {
  local failed=0
  echo "========================================="
  echo "UWNR + CUT serial preflight"
  echo "========================================="
  echo "SYN_ROOT:       ${SYN_ROOT}"
  echo "SOURCE_ROOT:    ${SOURCE_ROOT}"
  echo "DEPTH_ROOT:     ${DEPTH_ROOT}"
  echo "WORK_ROOT:      ${WORK_ROOT}"
  echo "RUOD_REF_SRC:   ${RUOD_REF_SRC}"
  echo "UWNR_DIR:       ${UWNR_DIR}"
  echo "UWNR_CKPT:      ${UWNR_CKPT}"
  echo "CUT_DIR:        ${CUT_DIR}"
  echo "CONDA_UWNR_ENV: ${CONDA_UWNR_ENV}"
  echo "CONDA_CUT_ENV:  ${CONDA_CUT_ENV}"
  echo "GPU_IDS:        ${GPU_IDS}"
  echo "TASKS:          ${TASKS}"
  echo "RESET_OUTPUTS:  ${RESET_OUTPUTS}"
  echo "LOG_ROOT:       ${LOG_ROOT}"
  echo "========================================="

  nvidia-smi -i "${GPU_IDS}" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || failed=1

  check_path "${UWNR_DIR}/test.py" "UWNR test.py" || failed=1
  check_path "${UWNR_CKPT}" "UWNR checkpoint" || failed=1
  check_path "${CUT_DIR}/train.py" "CUT train.py" || failed=1
  check_path "${CUT_DIR}/test.py" "CUT test.py" || failed=1
  check_path "${RUOD_REF_SRC}" "RUOD reference/domain" || failed=1
  check_path "${CONDA_UWNR_ENV}" "UWNR conda env" || failed=1
  check_path "${CONDA_CUT_ENV}" "CUT conda env" || failed=1

  check_path "${SOURCE_ROOT}/uwnr/source/train" "uwnr train source" || failed=1
  check_path "${SOURCE_ROOT}/uwnr/source/val" "uwnr val source" || failed=1
  check_path "${DEPTH_ROOT}/uwnr/train" "uwnr train depth" || failed=1
  check_path "${DEPTH_ROOT}/uwnr/val" "uwnr val depth" || failed=1
  check_path "${SOURCE_ROOT}/cut/source/train" "cut train source" || failed=1
  check_path "${SOURCE_ROOT}/cut/source/val" "cut val source" || failed=1

  bash -n scripts/exp_2/synthesis/run_uwnr_full.sh || failed=1
  bash -n scripts/exp_2/synthesis/run_cut_full.sh || failed=1
  bash -n scripts/exp_2/synthesis/check_synthesis_generation_completion.sh || failed=1

  echo
  echo "Dataset counts:"
  echo "  uwnr/train source: $(count_img "${SOURCE_ROOT}/uwnr/source/train")"
  echo "  uwnr/val   source: $(count_img "${SOURCE_ROOT}/uwnr/source/val")"
  echo "  uwnr/train depth:  $(count_img "${DEPTH_ROOT}/uwnr/train")"
  echo "  uwnr/val   depth:  $(count_img "${DEPTH_ROOT}/uwnr/val")"
  echo "  cut/train  source: $(count_img "${SOURCE_ROOT}/cut/source/train")"
  echo "  cut/val    source: $(count_img "${SOURCE_ROOT}/cut/source/val")"
  echo "  RUOD domain:       $(count_img "${RUOD_REF_SRC}")"

  if [[ "${failed}" != "0" ]]; then
    echo "Preflight failed. Fix missing paths/configs before launching full generation." >&2
    return 1
  fi
  echo "Preflight passed."
  return 0
}

run_task() {
  local task="$1"
  local log_path="${LOG_ROOT}/${task}.log"
  echo
  echo "========================================="
  echo "Start task: ${task}"
  echo "Log: ${log_path}"
  echo "Time: $(timestamp)"
  echo "========================================="

  record_status "${task}" "START" "${log_path}"

  case "${task}" in
    uwnr)
      (
        set -euo pipefail
        activate_env "${CONDA_UWNR_ENV}"
        SYN_ROOT="${SYN_ROOT}" \
        WORK_ROOT="${WORK_ROOT}" \
        SOURCE_ROOT="${SOURCE_ROOT}/uwnr/source" \
        DEPTH_ROOT="${DEPTH_ROOT}/uwnr" \
        RUOD_REF_SRC="${RUOD_REF_SRC}" \
        UWNR_DIR="${UWNR_DIR}" \
        UWNR_CKPT="${UWNR_CKPT}" \
        SPLITS="train val" \
        GPU_IDS="${GPU_IDS}" \
        PROCS_PER_GPU="${PROCS_PER_GPU}" \
        N_CPU="${N_CPU}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        RESET_OUTPUTS="${RESET_OUTPUTS}" \
        LOG_DIR="${LOG_ROOT}/uwnr_shards" \
        bash scripts/exp_2/synthesis/run_uwnr_full.sh
      ) 2>&1 | tee "${log_path}"
      ;;
    cut)
      (
        set -euo pipefail
        activate_env "${CONDA_CUT_ENV}"
        SYN_ROOT="${SYN_ROOT}" \
        SOURCE_ROOT="${SOURCE_ROOT}" \
        WORK_ROOT="${WORK_ROOT}" \
        RUOD_REF_SRC="${RUOD_REF_SRC}" \
        CUT_DIR="${CUT_DIR}" \
        GPU="$(first_gpu_id "${GPU_IDS}")" \
        GPU_IDS="${GPU_IDS}" \
        CUT_EPOCHS="${CUT_EPOCHS}" \
        CUT_EPOCHS_DECAY="${CUT_EPOCHS_DECAY}" \
        CUT_BATCH_SIZE="${CUT_BATCH_SIZE}" \
        CUT_NUM_THREADS="${CUT_NUM_THREADS}" \
        CUT_NUM_TEST="${CUT_NUM_TEST}" \
        COPY_MODE="${COPY_MODE}" \
        RESET_OUTPUTS="${RESET_OUTPUTS}" \
        LOG_DIR="${LOG_ROOT}/cut_steps" \
        bash scripts/exp_2/synthesis/run_cut_full.sh
      ) 2>&1 | tee "${log_path}"
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      return 2
      ;;
  esac

  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" == "0" ]]; then
    record_status "${task}" "OK" "${log_path}"
  else
    record_status "${task}" "FAILED:${rc}" "${log_path}"
  fi
  return 0
}

final_check() {
  local log_path="${LOG_ROOT}/final_completion_check.log"
  echo
  echo "========================================="
  echo "Final completion check"
  echo "========================================="
  SYN_ROOT="${SYN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  DEPTH_ROOT="${DEPTH_ROOT}" \
  MODELS="uwnr_ruod_ref cut_ruod" \
  bash scripts/exp_2/synthesis/check_synthesis_generation_completion.sh \
    2>&1 | tee "${log_path}"
}

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  if ! preflight 2>&1 | tee "${LOG_ROOT}/preflight.log"; then
    record_status "preflight" "FAILED" "${LOG_ROOT}/preflight.log"
    echo "Stop before generation because preflight failed." >&2
    exit 1
  fi
  record_status "preflight" "OK" "${LOG_ROOT}/preflight.log"
fi

for task in ${TASKS}; do
  run_task "${task}"
done

if [[ "${RUN_FINAL_CHECK}" == "1" ]]; then
  final_check || true
fi

echo
echo "========================================="
echo "UWNR + CUT serial run finished"
echo "========================================="
echo "LOG_ROOT:   ${LOG_ROOT}"
echo "STATUS_TSV: ${STATUS_TSV}"
echo "Status summary:"
cat "${STATUS_TSV}"
echo "========================================="