#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

AGGREGATE_ROOT="${AGGREGATE_ROOT:?Set AGGREGATE_ROOT to prediction/legacy_image_aggregate}"
CAM_ROOT="${CAM_ROOT:?Set CAM_ROOT to prediction/raw}"
MODELS="${MODELS:-imagenet_dino100e_ruod_cascade,realuw_dino100e_ruod_cascade,synthetic5_dino100e_ruod_cascade,imagenet_dino100e_dfui_ruod_cascade}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
STYLES="${STYLES:-legacy_jet,legacy_turbo_gamma05}"
TILE_WIDTH="${TILE_WIDTH:-480}"
TILE_HEIGHT="${TILE_HEIGHT:-360}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-3}"
OVERWRITE="${OVERWRITE:-0}"

args=(--aggregate-root "${AGGREGATE_ROOT}" --cam-root "${CAM_ROOT}"
  --models "${MODELS}" --layers "${LAYERS}" --styles "${STYLES}"
  --tile-width "${TILE_WIDTH}" --tile-height "${TILE_HEIGHT}"
  --png-compress-level "${PNG_COMPRESS_LEVEL}")
[[ "${OVERWRITE}" == 1 ]] && args+=(--overwrite)
python "${SCRIPT_DIR}/compose_prediction_legacy_aggregate_panels.py" "${args[@]}"
