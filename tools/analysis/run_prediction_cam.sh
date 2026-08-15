#!/usr/bin/env bash
# Prediction-conditioned CAM: max aggregation, JET, per-image min-max.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
RUOD_ANN="${RUOD_ANN:-${RUOD_ROOT}/annotations/instances_val.json}"
CAM_MODELS_CONFIG="${CAM_MODELS_CONFIG:-${SCRIPT_DIR}/models.cam.example.json}"
SAMPLE_ROOT="${SAMPLE_ROOT:?SAMPLE_ROOT must contain manifest.jsonl}"
CAM_OUT_ROOT="${CAM_OUT_ROOT:?CAM_OUT_ROOT is required}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
CAM_DEVICES="${CAM_DEVICES:-cuda:0}"
CAM_PARALLEL_MODELS="${CAM_PARALLEL_MODELS:-1}"
CAM_SCORE_THRESHOLD="${CAM_SCORE_THRESHOLD:-0.05}"
CAM_MAX_PREDICTIONS="${CAM_MAX_PREDICTIONS:-30}"

[[ -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || { echo "Error: missing sample manifest" >&2; exit 1; }
[[ -s "${CAM_MODELS_CONFIG}" ]] || { echo "Error: missing CAM models config" >&2; exit 1; }

echo "============================================================"
echo "Prediction CAM: max + JET + per-image min-max"
echo "============================================================"
echo "Sample:      ${SAMPLE_ROOT}/manifest.jsonl"
echo "Models:      ${CAM_MODELS_CONFIG}"
echo "Devices:     ${CAM_DEVICES}"
echo "Output:      ${CAM_OUT_ROOT}/jet_per_image_max/panels_5x4/legacy_jet"

env \
    RUOD_ROOT="${RUOD_ROOT}" \
    RUOD_ANN="${RUOD_ANN}" \
    MODELS_CONFIG="${CAM_MODELS_CONFIG}" \
    SAMPLE_ROOT="${SAMPLE_ROOT}" \
    OUT_ROOT="${CAM_OUT_ROOT}" \
    CAM_ROOT="${CAM_OUT_ROOT}/raw" \
    RENDER_ROOT="${CAM_OUT_ROOT}/unused_standard_render" \
    LOG_ROOT="${CAM_OUT_ROOT}/logs" \
    LAYERS="${LAYERS}" \
    SCORE_THRESHOLD="${CAM_SCORE_THRESHOLD}" \
    MAX_PREDICTIONS_PER_IMAGE="${CAM_MAX_PREDICTIONS}" \
    DEVICES="${CAM_DEVICES}" \
    PARALLEL_MODELS="${CAM_PARALLEL_MODELS}" \
    RUN_EXTRACT=1 \
    RUN_RENDER=0 \
    RUN_LEGACY_AGGREGATE=1 \
    LEGACY_AGGREGATE_ROOT="${CAM_OUT_ROOT}/jet_per_image_max" \
    LEGACY_AGGREGATION=max \
    LEGACY_STYLES=legacy_jet \
    RESUME=1 \
    bash scripts/exp_2/features/further_features/run_prediction_xgradcam_analysis.sh
