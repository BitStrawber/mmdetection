#!/usr/bin/env bash
set -euo pipefail

# Independent UWNR smoke/visual comparison.
#
# Run this only after activating the UWNR environment:
#   conda activate /media/SSD1/conda_envs/uwnr
#   NUM=20 GPU=2 bash scripts/exp_2/synthesis/run_uwnr_independent_compare.sh
#
# This does not share samples with UWDF. It samples clean ImageNet images and
# underwater references only for UWNR, then exports source/reference/UWNR
# triplet images.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU="${GPU:-2}"
TEST_SIZE="${TEST_SIZE:-256}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwnr/source/train}"
REF_ROOT="${REF_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwnr_lnrud_ref/independent_random${NUM}_seed${SEED}}"

echo "========================================="
echo "Independent UWNR comparison"
echo "========================================="
echo "NUM:        ${NUM}"
echo "SEED:       ${SEED}"
echo "GPU:        ${GPU}"
echo "TEST_SIZE:  ${TEST_SIZE}"
echo "SOURCE:     ${SOURCE_ROOT}"
echo "REFERENCE:  ${REF_ROOT}"
echo "WORK_ROOT:  ${WORK_ROOT}"
echo "========================================="

NUM="${NUM}" \
SEED="${SEED}" \
GPU="${GPU}" \
TEST_SIZE="${TEST_SIZE}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
REF_ROOT="${REF_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
RANDOM_SOURCE_DIR="${WORK_ROOT}/source/train" \
RANDOM_REF_ROOT="${WORK_ROOT}/reference" \
DEPTH_DIR="${WORK_ROOT}/megadepth/train" \
PREP_DIR="${WORK_ROOT}/prepared/train" \
FLAT_SAVE_DIR="${WORK_ROOT}/generated_flat/train" \
RESTORE_DIR="${WORK_ROOT}/generated/train" \
TRIPLET_DIR="${WORK_ROOT}/triplets" \
RUN_PIPELINE="${RUN_PIPELINE:-1}" \
RUN_TRIPLET="${RUN_TRIPLET:-1}" \
bash scripts/exp_2/synthesis/run_uwnr_random_ref_smoke.sh

echo
echo "Done."
echo "UWNR generated: ${WORK_ROOT}/generated/train"
echo "UWNR triplets:  ${WORK_ROOT}/triplets"
echo "Manifest:       ${WORK_ROOT}/random_selection_manifest.json"
