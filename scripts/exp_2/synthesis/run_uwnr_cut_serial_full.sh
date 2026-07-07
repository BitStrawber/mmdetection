#!/usr/bin/env bash
set -uo pipefail

# Serially run UWNR full generation and CUT training sweeps. A failed task or
# CUT sweep item is recorded but does not stop the remaining workflow.
#
# Default workflow:
#   1) UWNR train/val generation at TEST_SIZE=512, one process per GPU.
#   2) CUT training sweeps with batch size 2:
#        1 epoch on GPU 2, 2 epochs on GPU 3, 3 epochs on GPU 4, 5 epochs on GPU 5.

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
TASKS="${TASKS:-uwnr cut_sweeps}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_FINAL_CHECK="${RUN_FINAL_CHECK:-1}"

TEST_SIZE="${TEST_SIZE:-512}"
N_CPU="${N_CPU:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-2}"
CUT_NUM_THREADS="${CUT_NUM_THREADS:-8}"
CUT_SWEEPS="${CUT_SWEEPS:-1:2 2:3 3:4 5:5}"
COPY_MODE="${COPY_MODE:-copy}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/uwnr512_cut_bs2_sweeps_$(date +%Y%m%d_%H%M%S)}"
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
  echo "UWNR512 generation + CUT bs2 sweep preflight"
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
  echo "PROCS_PER_GPU:  ${PROCS_PER_GPU}"
  echo "TEST_SIZE:      ${TEST_SIZE}"
  echo "N_CPU:          ${N_CPU}"
  echo "TASKS:          ${TASKS}"
  echo "CUT_BATCH_SIZE: ${CUT_BATCH_SIZE}"
  echo "CUT_SWEEPS:     ${CUT_SWEEPS}"
  echo "RESET_OUTPUTS:  ${RESET_OUTPUTS}"
  echo "LOG_ROOT:       ${LOG_ROOT}"
  echo "========================================="

  nvidia-smi -i "${GPU_IDS}" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || failed=1
  for item in ${CUT_SWEEPS}; do
    gpu="${item#*:}"
    nvidia-smi -i "${gpu}" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || failed=1
  done

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
  bash -n scripts/exp_2/synthesis/run_cut_prepare_dataset.sh || failed=1
  bash -n scripts/exp_2/synthesis/run_cut_train.sh || failed=1
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

run_uwnr() {
  local log_path="${LOG_ROOT}/uwnr.log"
  record_status "uwnr" "START" "${log_path}"
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
    TEST_SIZE="${TEST_SIZE}" \
    N_CPU="${N_CPU}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    RESET_OUTPUTS="${RESET_OUTPUTS}" \
    LOG_DIR="${LOG_ROOT}/uwnr_shards" \
    bash scripts/exp_2/synthesis/run_uwnr_full.sh
  ) 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" == "0" ]]; then
    record_status "uwnr" "OK" "${log_path}"
  else
    record_status "uwnr" "FAILED:${rc}" "${log_path}"
  fi
  return 0
}

prepare_cut_dataset() {
  local log_path="${LOG_ROOT}/cut_prepare.log"
  record_status "cut_prepare" "START" "${log_path}"
  (
    set -euo pipefail
    activate_env "${CONDA_CUT_ENV}"
    MODE=full \
    METHODS=cut \
    SPLITS="train val" \
    GPU=2 \
    FULL_LIMIT=0 \
    SYN_ROOT="${SYN_ROOT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    WORK_ROOT="${WORK_ROOT}" \
    RUOD_REF_SRC="${RUOD_REF_SRC}" \
    COPY_MODE="${COPY_MODE}" \
    bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh
  ) 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" == "0" ]]; then
    record_status "cut_prepare" "OK" "${log_path}"
    return 0
  fi
  record_status "cut_prepare" "FAILED:${rc}" "${log_path}"
  return "${rc}"
}

run_cut_sweeps() {
  local log_root="${LOG_ROOT}/cut_sweeps"
  mkdir -p "${log_root}"

  if ! prepare_cut_dataset; then
    echo "CUT dataset preparation failed; skip all CUT sweep items." >&2
    return 0
  fi

  for item in ${CUT_SWEEPS}; do
    local epochs="${item%%:*}"
    local gpu="${item#*:}"
    local exp_name="imagenet_ruod_cut_full_bs${CUT_BATCH_SIZE}_${epochs}epoch_gpu${gpu}"
    local log_path="${log_root}/${exp_name}.log"

    echo
    echo "========================================="
    echo "Start CUT sweep: epochs=${epochs}, gpu=${gpu}, batch=${CUT_BATCH_SIZE}"
    echo "EXP_NAME: ${exp_name}"
    echo "Log: ${log_path}"
    echo "========================================="
    record_status "cut_${epochs}epoch_gpu${gpu}" "START" "${log_path}"

    (
      set -euo pipefail
      activate_env "${CONDA_CUT_ENV}"
      DATA_NAME=imagenet_ruod_cut_full_ssd \
      DATA_ROOT="${WORK_ROOT}/cut/datasets/imagenet_ruod_cut_full_ssd" \
      EXP_NAME="${exp_name}" \
      SYN_ROOT="${SYN_ROOT}" \
      CUT_DIR="${CUT_DIR}" \
      GPU_IDS="${gpu}" \
      BATCH_SIZE="${CUT_BATCH_SIZE}" \
      NUM_THREADS="${CUT_NUM_THREADS}" \
      N_EPOCHS="${epochs}" \
      N_EPOCHS_DECAY=0 \
      SAVE_EPOCH_FREQ=1 \
      PRINT_FREQ=100 \
      NO_HTML=1 \
      bash scripts/exp_2/synthesis/run_cut_train.sh
    ) 2>&1 | tee "${log_path}"
    local rc=${PIPESTATUS[0]}
    if [[ "${rc}" == "0" ]]; then
      record_status "cut_${epochs}epoch_gpu${gpu}" "OK" "${log_path}"
    else
      record_status "cut_${epochs}epoch_gpu${gpu}" "FAILED:${rc}" "${log_path}"
    fi
  done
  return 0
}

run_task() {
  local task="$1"
  case "${task}" in
    uwnr)
      run_uwnr
      ;;
    cut_sweeps|cut_train|cut)
      run_cut_sweeps
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      record_status "${task}" "FAILED:unknown_task" "-"
      ;;
  esac
  return 0
}

final_check() {
  local log_path="${LOG_ROOT}/final_check.log"
  echo
  echo "========================================="
  echo "Final check"
  echo "========================================="
  SYN_ROOT="${SYN_ROOT}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  DEPTH_ROOT="${DEPTH_ROOT}" \
  MODELS="uwnr_ruod_ref" \
  bash scripts/exp_2/synthesis/check_synthesis_generation_completion.sh \
    2>&1 | tee "${log_path}"

  echo | tee -a "${log_path}"
  echo "CUT checkpoint checks:" | tee -a "${log_path}"
  for item in ${CUT_SWEEPS}; do
    epochs="${item%%:*}"
    gpu="${item#*:}"
    exp_name="imagenet_ruod_cut_full_bs${CUT_BATCH_SIZE}_${epochs}epoch_gpu${gpu}"
    echo "===== ${exp_name} =====" | tee -a "${log_path}"
    ls -lh "${CUT_DIR}/checkpoints/${exp_name}" 2>&1 | tee -a "${log_path}" || true
  done
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
  echo
  echo "========================================="
  echo "Run task: ${task}"
  echo "Time: $(timestamp)"
  echo "========================================="
  run_task "${task}"
done

if [[ "${RUN_FINAL_CHECK}" == "1" ]]; then
  final_check || true
fi

echo
echo "========================================="
echo "UWNR512 generation + CUT bs2 sweeps finished"
echo "========================================="
echo "LOG_ROOT:   ${LOG_ROOT}"
echo "STATUS_TSV: ${STATUS_TSV}"
echo "Status summary:"
cat "${STATUS_TSV}"
echo "========================================="