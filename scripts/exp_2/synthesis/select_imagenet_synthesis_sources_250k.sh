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
# The selected source trees are written to SSD1 by default:
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
#   CLEAN_OLD_ROOTS=1 bash ... # remove old source/manifests from OLD_SYN_ROOTS and SYN_ROOT first

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ROOT="${TRAIN_ROOT:-/media/HDD1/XCX/exp_2/imagenet1k/train}"
VAL_ROOT="${VAL_ROOT:-/media/HDD1/XCX/exp_2/imagenet1k/val}"
SYN_ROOT="${SYN_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
OLD_SYN_ROOTS="${OLD_SYN_ROOTS:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
METHODS="${METHODS:-uwnr syreanet cut watergan uwdf}"
TRAIN_PER_METHOD="${TRAIN_PER_METHOD:-250000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-10}"
SEED="${SEED:-20260621}"
LINK_MODE="${LINK_MODE:-symlink}"
OVERWRITE="${OVERWRITE:-1}"
CLEAN_OLD_ROOTS="${CLEAN_OLD_ROOTS:-0}"

clean_root_sources() {
  local root="$1"
  [[ -n "${root}" ]] || return 0
  case "${root}" in
    /media/HDD1/XCX/exp_2/synthetic_imagenet|/media/SSD1/XCX/exp_2/synthetic_imagenet|/media/SSD0/XCX/exp_2/synthetic_imagenet)
      ;;
    *)
      echo "Refuse to clean unexpected root: ${root}" >&2
      exit 1
      ;;
  esac

  echo "Cleaning source selections under: ${root}"
  for method in ${METHODS}; do
    if [[ -e "${root}/${method}/source" ]]; then
      echo "  rm -rf ${root}/${method}/source"
      rm -rf "${root:?}/${method}/source"
    fi
  done
  if [[ -e "${root}/manifests" ]]; then
    echo "  rm -rf ${root}/manifests"
    rm -rf "${root:?}/manifests"
  fi
}

echo "========================================="
echo "Select ImageNet synthesis sources"
echo "========================================="
echo "TRAIN_ROOT:        ${TRAIN_ROOT}"
echo "VAL_ROOT:          ${VAL_ROOT}"
echo "SYN_ROOT:          ${SYN_ROOT}"
echo "OLD_SYN_ROOTS:     ${OLD_SYN_ROOTS}"
echo "METHODS:           ${METHODS}"
echo "TRAIN_PER_METHOD:  ${TRAIN_PER_METHOD}"
echo "VAL_PER_CLASS:     ${VAL_PER_CLASS}"
echo "SEED:              ${SEED}"
echo "LINK_MODE:         ${LINK_MODE}"
echo "OVERWRITE:         ${OVERWRITE}"
echo "CLEAN_OLD_ROOTS:   ${CLEAN_OLD_ROOTS}"
echo "========================================="

if [[ "${CLEAN_OLD_ROOTS}" == "1" ]]; then
  for root in ${OLD_SYN_ROOTS} "${SYN_ROOT}"; do
    clean_root_sources "${root}"
  done
fi

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
