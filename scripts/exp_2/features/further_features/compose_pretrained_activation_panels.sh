#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

RENDER_ROOT="${RENDER_ROOT:?Set RENDER_ROOT to pretrained_backbone/rendered}"
MANIFEST="${MANIFEST:?Set MANIFEST to sample/manifest.jsonl}"
MODELS="${MODELS:-imagenet_dino100e_backbone,realuw_dino100e_backbone,synthetic5_dino100e_backbone,imagenet_dino100e_dfui_backbone}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
NORMALIZATIONS="${NORMALIZATIONS:-imagenet_reference_dataset_p1_p99,imagenet_reference_per_sample_p1_p99}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-3}"
OVERWRITE="${OVERWRITE:-0}"

args=(--render-root "${RENDER_ROOT}" --manifest "${MANIFEST}"
  --models "${MODELS}" --layers "${LAYERS}"
  --normalizations "${NORMALIZATIONS}"
  --png-compress-level "${PNG_COMPRESS_LEVEL}")
[[ "${OVERWRITE}" == 1 ]] && args+=(--overwrite)
python "${SCRIPT_DIR}/compose_pretrained_activation_panels.py" "${args[@]}"
