#!/usr/bin/env bash
set -euo pipefail

# Rebuild ImageNet source selections for underwater synthesis.
#
# New policy:
#   - 250k train images per model family.
#   - Cross-model overlap is allowed.
#   - Inside each ImageNet class, methods use rotated source windows to cover
#     as many unique class images as possible across all methods.
#
# The selected source trees are written to:
#   ${SYN_ROOT}/${method}/source/{train,val}/<synset>/
#
# Method notes:
#   - stable_diffusion_img2img uses the uwdf source family.
#   - syreanet_synthesis reuses syreanet/source in the SSD preparation script.
#
# Usage:
#   bash scripts/exp_2/synthesis/select_imagenet_synthesis_sources_250k.sh
#
# Optional:
#   METHODS="uwnr syreanet cut watergan uwdf" LINK_MODE=copy bash ...
#   VAL_PER_CLASS=0 bash ...   # skip val source selection

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ROOT="${TRAIN_ROOT:-/media/HDD1/XCX/exp_2/imagenet1k/train}"
VAL_ROOT="${VAL_ROOT:-/media/HDD1/XCX/exp_2/imagenet1k/val}"
SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
METHODS="${METHODS:-uwnr syreanet cut watergan uwdf}"
TRAIN_PER_METHOD="${TRAIN_PER_METHOD:-250000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-10}"
SEED="${SEED:-20260621}"
LINK_MODE="${LINK_MODE:-symlink}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Select ImageNet synthesis sources"
echo "========================================="
echo "TRAIN_ROOT:        ${TRAIN_ROOT}"
echo "VAL_ROOT:          ${VAL_ROOT}"
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "METHODS:           ${METHODS}"
echo "TRAIN_PER_METHOD:  ${TRAIN_PER_METHOD}"
echo "VAL_PER_CLASS:     ${VAL_PER_CLASS}"
echo "SEED:              ${SEED}"
echo "LINK_MODE:         ${LINK_MODE}"
echo "OVERWRITE:         ${OVERWRITE}"
echo "========================================="

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

python tools/select_imagenet_synthesis_sources.py \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --out-root "${SYN_ROOT}" \
  --methods ${METHODS} \
  --train-per-method "${TRAIN_PER_METHOD}" \
  --val-per-class "${VAL_PER_CLASS}" \
  --train-selection rotating-class-balanced \
  --seed "${SEED}" \
  --link-mode "${LINK_MODE}" \
  "${overwrite_args[@]}"
