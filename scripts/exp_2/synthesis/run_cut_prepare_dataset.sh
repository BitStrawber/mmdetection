#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_cut_smoke}"
DATA_ROOT="${DATA_ROOT:-${SYN_ROOT}/cut/datasets/${DATA_NAME}}"

TRAIN_A_SOURCE="${TRAIN_A_SOURCE:-${SOURCE_ROOT}/cut/source/train}"
TEST_A_SOURCE="${TEST_A_SOURCE:-${SOURCE_ROOT}/cut/source/train}"
TRAIN_B_SOURCE="${TRAIN_B_SOURCE:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
TEST_B_SOURCE="${TEST_B_SOURCE:-${TRAIN_B_SOURCE}}"

TRAIN_A_LIMIT="${TRAIN_A_LIMIT:-1000}"
TRAIN_B_LIMIT="${TRAIN_B_LIMIT:-1000}"
TEST_A_LIMIT="${TEST_A_LIMIT:-100}"
TEST_B_LIMIT="${TEST_B_LIMIT:-100}"
LINK_MODE="${LINK_MODE:-symlink}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Prepare CUT ImageNet -> RUOD dataset"
echo "========================================="
echo "DATA_ROOT:      ${DATA_ROOT}"
echo "SOURCE_ROOT:    ${SOURCE_ROOT}"
echo "TRAIN_A_SOURCE: ${TRAIN_A_SOURCE}"
echo "TRAIN_B_SOURCE: ${TRAIN_B_SOURCE}"
echo "TEST_A_SOURCE:  ${TEST_A_SOURCE}"
echo "TEST_B_SOURCE:  ${TEST_B_SOURCE}"
echo "TRAIN_A_LIMIT:  ${TRAIN_A_LIMIT}"
echo "TRAIN_B_LIMIT:  ${TRAIN_B_LIMIT}"
echo "TEST_A_LIMIT:   ${TEST_A_LIMIT}"
echo "TEST_B_LIMIT:   ${TEST_B_LIMIT}"
echo "LINK_MODE:      ${LINK_MODE}"
echo "========================================="

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python tools/prepare_cut_imagenet_ruod_dataset.py \
  --train-a-source "${TRAIN_A_SOURCE}" \
  --train-b-source "${TRAIN_B_SOURCE}" \
  --test-a-source "${TEST_A_SOURCE}" \
  --test-b-source "${TEST_B_SOURCE}" \
  --out-dir "${DATA_ROOT}" \
  --train-a-limit "${TRAIN_A_LIMIT}" \
  --train-b-limit "${TRAIN_B_LIMIT}" \
  --test-a-limit "${TEST_A_LIMIT}" \
  --test-b-limit "${TEST_B_LIMIT}" \
  --link-mode "${LINK_MODE}" \
  "${EXTRA_ARGS[@]}"

echo
echo "Done: ${DATA_ROOT}"
