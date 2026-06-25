#!/usr/bin/env bash
set -euo pipefail

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
CUT_DIR="${CUT_DIR:-/home/fcp/xcx/exp_2/syn/contrastive-unpaired-translation}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_cut_smoke}"
DATA_ROOT="${DATA_ROOT:-${SYN_ROOT}/cut/datasets/${DATA_NAME}}"
EXP_NAME="${EXP_NAME:-${DATA_NAME}}"
SPLIT="${SPLIT:-train}"
GPU_IDS="${GPU_IDS:-2}"
NUM_TEST="${NUM_TEST:-100}"

LOAD_SIZE="${LOAD_SIZE:-256}"
CROP_SIZE="${CROP_SIZE:-256}"
PREPROCESS="${PREPROCESS:-resize_and_crop}"
RESULTS_ROOT="${RESULTS_ROOT:-${SYN_ROOT}/cut/results/${EXP_NAME}_${SPLIT}}"
RESTORE_DIR="${RESTORE_DIR:-${SYN_ROOT}/cut/generated/${SPLIT}}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/testA_manifest.jsonl}"

echo "========================================="
echo "Generate CUT fake_B and restore labels"
echo "========================================="
echo "CUT_DIR:      ${CUT_DIR}"
echo "DATA_ROOT:    ${DATA_ROOT}"
echo "EXP_NAME:     ${EXP_NAME}"
echo "SPLIT:        ${SPLIT}"
echo "GPU_IDS:      ${GPU_IDS}"
echo "NUM_TEST:     ${NUM_TEST}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "RESTORE_DIR:  ${RESTORE_DIR}"
echo "MANIFEST:     ${MANIFEST}"
echo "========================================="

if [[ ! -d "${CUT_DIR}" ]]; then
  echo "Error: CUT repo not found: ${CUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Error: CUT dataset not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Error: testA manifest not found: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${SYN_ROOT}/cut/logs" "${RESULTS_ROOT}" "${RESTORE_DIR}"

(
  cd "${CUT_DIR}"
  python test.py \
    --dataroot "${DATA_ROOT}" \
    --name "${EXP_NAME}" \
    --CUT_mode CUT \
    --model cut \
    --dataset_mode unaligned \
    --direction AtoB \
    --phase test \
    --num_test "${NUM_TEST}" \
    --gpu_ids "${GPU_IDS}" \
    --load_size "${LOAD_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --preprocess "${PREPROCESS}" \
    --results_dir "${RESULTS_ROOT}"
) 2>&1 | tee "${SYN_ROOT}/cut/logs/${EXP_NAME}_${SPLIT}_generate.log"

python tools/restore_cut_fake_b.py \
  --manifest "${MANIFEST}" \
  --results-dir "${RESULTS_ROOT}" \
  --out-dir "${RESTORE_DIR}" \
  --overwrite

echo
echo "Generated restored output: ${RESTORE_DIR}"
