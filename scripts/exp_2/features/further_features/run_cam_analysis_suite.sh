#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:-${RUOD_ROOT}/val}"
RUOD_ANN="${RUOD_ANN:-${RUOD_ROOT}/annotations/instances_val.json}"
DETECTOR_MODELS_CONFIG="${DETECTOR_MODELS_CONFIG:-${SCRIPT_DIR}/models.fixed_gt_cam.json}"
PRETRAINED_MODELS_CONFIG="${PRETRAINED_MODELS_CONFIG:-${SCRIPT_DIR}/models.pretrained_backbone_activation.json}"

SAMPLES="${SAMPLES:-10}"
SEED="${SEED:-2026}"
MATERIALIZE="${MATERIALIZE:-none}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cam_suite_ruod${SAMPLES}_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/further_features/${RUN_NAME}}"
SAMPLE_ROOT="${SAMPLE_ROOT:-${OUT_ROOT}/sample}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

# Branch switches. Fixed-GT remains the default controlled experiment.
RUN_FIXED_GT_CAM="${RUN_FIXED_GT_CAM:-1}"
RUN_PREDICTION_CAM="${RUN_PREDICTION_CAM:-0}"
RUN_PRETRAINED_BACKBONE_ACTIVATION="${RUN_PRETRAINED_BACKBONE_ACTIVATION:-0}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RESUME="${RESUME:-1}"

# Downstream detector settings shared by fixed-GT and prediction branches.
DETECTOR_DEVICES="${DETECTOR_DEVICES:-cuda:4,cuda:5}"
DETECTOR_PARALLEL_MODELS="${DETECTOR_PARALLEL_MODELS:-2}"
CASCADE_STAGE="${CASCADE_STAGE:--1}"
DETECTOR_REFERENCE_MODEL="${DETECTOR_REFERENCE_MODEL:-imagenet_dino100e_ruod_cascade}"

# Fixed-GT controls.
MAX_INSTANCES_PER_IMAGE="${MAX_INSTANCES_PER_IMAGE:-5}"
INSTANCE_ORDER="${INSTANCE_ORDER:-area-desc}"

# Prediction controls.
PREDICTION_SCORE_THRESHOLD="${PREDICTION_SCORE_THRESHOLD:-0.05}"
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-10}"
PREDICTION_MATCH_IOU="${PREDICTION_MATCH_IOU:-0.50}"

# Bare pretrained-backbone controls.
PRETRAINED_DEVICE="${PRETRAINED_DEVICE:-cuda:6}"
PRETRAINED_REFERENCE_MODEL="${PRETRAINED_REFERENCE_MODEL:-imagenet_dino100e_backbone}"
PRETRAINED_RENDER_MODEL_WORKERS="${PRETRAINED_RENDER_MODEL_WORKERS:-4}"

LOW_PERCENTILE="${LOW_PERCENTILE:-1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99}"
DISPLAY_GAMMA="${DISPLAY_GAMMA:-1.0}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.48}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

for value in \
    "${RUN_FIXED_GT_CAM}" \
    "${RUN_PREDICTION_CAM}" \
    "${RUN_PRETRAINED_BACKBONE_ACTIVATION}"; do
    [[ "${value}" == 0 || "${value}" == 1 ]] || {
        echo "Error: branch switches must be 0 or 1, got ${value}" >&2
        exit 1
    }
done
[[ "${RUN_FIXED_GT_CAM}" == 1 || "${RUN_PREDICTION_CAM}" == 1 || \
   "${RUN_PRETRAINED_BACKBONE_ACTIVATION}" == 1 ]] || {
    echo "Error: all analysis branches are disabled" >&2
    exit 1
}

echo "============================================================"
echo "RUOD CAM and activation analysis suite"
echo "============================================================"
echo "RUOD:                           ${RUOD_ROOT}"
echo "Samples / seed:                 ${SAMPLES} / ${SEED}"
echo "Layers:                         ${LAYERS}"
echo "RUN_FIXED_GT_CAM:               ${RUN_FIXED_GT_CAM}"
echo "RUN_PREDICTION_CAM:             ${RUN_PREDICTION_CAM}"
echo "RUN_PRETRAINED_BACKBONE:        ${RUN_PRETRAINED_BACKBONE_ACTIVATION}"
echo "Detector devices:               ${DETECTOR_DEVICES}"
echo "Pretrained device:              ${PRETRAINED_DEVICE}"
echo "Output:                         ${OUT_ROOT}"
echo "============================================================"

if [[ "${RUN_SAMPLE}" == 1 ]]; then
    if [[ "${RESUME}" == 1 && -s "${SAMPLE_ROOT}/manifest.jsonl" ]]; then
        echo "REUSE shared sample: ${SAMPLE_ROOT}/manifest.jsonl"
    else
        python "${SCRIPT_DIR}/sample_fixed_gt_ruod.py" \
            --annotation-file "${RUOD_ANN}" \
            --image-root "${RUOD_IMAGE_ROOT}" \
            --out-dir "${SAMPLE_ROOT}" \
            --samples "${SAMPLES}" \
            --seed "${SEED}" \
            --minimum-instances 1 \
            --materialize "${MATERIALIZE}" \
            2>&1 | tee "${LOG_ROOT}/00_sample.log"
    fi
fi
[[ -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || {
    echo "Error: shared sample manifest is missing: ${SAMPLE_ROOT}/manifest.jsonl" >&2
    exit 1
}

if [[ "${RUN_FIXED_GT_CAM}" == 1 ]]; then
    echo "===== BRANCH fixed-GT detector XGradCAM ====="
    env \
        RUOD_ROOT="${RUOD_ROOT}" \
        RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT}" \
        RUOD_ANN="${RUOD_ANN}" \
        MODELS_CONFIG="${DETECTOR_MODELS_CONFIG}" \
        SAMPLES="${SAMPLES}" \
        SEED="${SEED}" \
        SAMPLE_ROOT="${SAMPLE_ROOT}" \
        OUT_ROOT="${OUT_ROOT}/fixed_gt" \
        CAM_ROOT="${OUT_ROOT}/fixed_gt/raw" \
        RENDER_ROOT="${OUT_ROOT}/fixed_gt/rendered" \
        LOG_ROOT="${OUT_ROOT}/fixed_gt/logs" \
        RUN_SAMPLE=0 \
        RUN_EXTRACT="${RUN_FIXED_GT_EXTRACT:-1}" \
        RUN_RENDER="${RUN_FIXED_GT_RENDER:-1}" \
        SAVE_INSTANCE_VIEWS="${FIXED_GT_SAVE_INSTANCE_VIEWS:-0}" \
        RUN_PLOTS="${RUN_FIXED_GT_PLOTS:-1}" \
        RUN_IMAGE_AGGREGATE="${RUN_FIXED_GT_IMAGE_AGGREGATE:-1}" \
        IMAGE_AGGREGATE_ROOT="${OUT_ROOT}/fixed_gt/image_aggregate" \
        IMAGE_AGGREGATION="${FIXED_GT_IMAGE_AGGREGATION:-max}" \
        IMAGE_AGGREGATE_VIEW="${FIXED_GT_IMAGE_AGGREGATE_VIEW:-pure}" \
        MAX_INSTANCES_PER_IMAGE="${MAX_INSTANCES_PER_IMAGE}" \
        INSTANCE_ORDER="${INSTANCE_ORDER}" \
        LAYERS="${LAYERS}" \
        CASCADE_STAGE="${CASCADE_STAGE}" \
        REFERENCE_MODEL="${DETECTOR_REFERENCE_MODEL}" \
        LOW_PERCENTILE="${LOW_PERCENTILE}" \
        HIGH_PERCENTILE="${HIGH_PERCENTILE}" \
        DISPLAY_GAMMA="${DISPLAY_GAMMA}" \
        OVERLAY_ALPHA="${OVERLAY_ALPHA}" \
        DEVICES="${DETECTOR_DEVICES}" \
        PARALLEL_MODELS="${DETECTOR_PARALLEL_MODELS}" \
        RESUME="${RESUME}" \
        bash "${SCRIPT_DIR}/run_fixed_gt_xgradcam_analysis.sh" \
        2>&1 | tee "${LOG_ROOT}/10_fixed_gt_branch.log"
fi

if [[ "${RUN_PREDICTION_CAM}" == 1 ]]; then
    echo "===== BRANCH prediction-conditioned detector XGradCAM ====="
    env \
        RUOD_ROOT="${RUOD_ROOT}" \
        RUOD_ANN="${RUOD_ANN}" \
        MODELS_CONFIG="${DETECTOR_MODELS_CONFIG}" \
        SAMPLE_ROOT="${SAMPLE_ROOT}" \
        OUT_ROOT="${OUT_ROOT}/prediction" \
        CAM_ROOT="${OUT_ROOT}/prediction/raw" \
        RENDER_ROOT="${OUT_ROOT}/prediction/rendered" \
        LOG_ROOT="${OUT_ROOT}/prediction/logs" \
        RUN_EXTRACT="${RUN_PREDICTION_EXTRACT:-1}" \
        RUN_RENDER="${RUN_PREDICTION_RENDER:-1}" \
        RUN_LEGACY_AGGREGATE="${RUN_PREDICTION_LEGACY_AGGREGATE:-1}" \
        LEGACY_AGGREGATE_ROOT="${OUT_ROOT}/prediction/legacy_image_aggregate" \
        LEGACY_AGGREGATION="${PREDICTION_LEGACY_AGGREGATION:-sum}" \
        LEGACY_STYLES="${PREDICTION_LEGACY_STYLES:-legacy_jet,legacy_turbo_gamma05}" \
        LAYERS="${LAYERS}" \
        CASCADE_STAGE="${CASCADE_STAGE}" \
        SCORE_THRESHOLD="${PREDICTION_SCORE_THRESHOLD}" \
        MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE}" \
        MATCH_IOU_THRESHOLD="${PREDICTION_MATCH_IOU}" \
        REFERENCE_MODEL="${DETECTOR_REFERENCE_MODEL}" \
        LOW_PERCENTILE="${LOW_PERCENTILE}" \
        HIGH_PERCENTILE="${HIGH_PERCENTILE}" \
        DISPLAY_GAMMA="${DISPLAY_GAMMA}" \
        OVERLAY_ALPHA="${OVERLAY_ALPHA}" \
        DEVICES="${DETECTOR_DEVICES}" \
        PARALLEL_MODELS="${DETECTOR_PARALLEL_MODELS}" \
        RESUME="${RESUME}" \
        bash "${SCRIPT_DIR}/run_prediction_xgradcam_analysis.sh" \
        2>&1 | tee "${LOG_ROOT}/20_prediction_branch.log"
fi

if [[ "${RUN_PRETRAINED_BACKBONE_ACTIVATION}" == 1 ]]; then
    echo "===== BRANCH bare pretrained backbone activation ====="
    env \
        SAMPLE_ROOT="${SAMPLE_ROOT}" \
        PRETRAINED_MODELS_CONFIG="${PRETRAINED_MODELS_CONFIG}" \
        OUT_ROOT="${OUT_ROOT}/pretrained_backbone" \
        FEATURE_ROOT="${OUT_ROOT}/pretrained_backbone/feature_store" \
        RENDER_ROOT="${OUT_ROOT}/pretrained_backbone/rendered" \
        LOG_ROOT="${OUT_ROOT}/pretrained_backbone/logs" \
        RUN_EXTRACT="${RUN_PRETRAINED_EXTRACT:-1}" \
        RUN_RENDER="${RUN_PRETRAINED_RENDER:-1}" \
        LAYERS="${LAYERS}" \
        PRETRAINED_DEVICE="${PRETRAINED_DEVICE}" \
        PRETRAINED_REFERENCE_MODEL="${PRETRAINED_REFERENCE_MODEL}" \
        PRETRAINED_RENDER_MODEL_WORKERS="${PRETRAINED_RENDER_MODEL_WORKERS}" \
        LOW_PERCENTILE="${LOW_PERCENTILE}" \
        HIGH_PERCENTILE="${HIGH_PERCENTILE}" \
        OVERWRITE="${PRETRAINED_OVERWRITE:-0}" \
        bash "${SCRIPT_DIR}/run_pretrained_backbone_activation.sh" \
        2>&1 | tee "${LOG_ROOT}/30_pretrained_backbone_branch.log"
fi

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
RUN_NAME=${RUN_NAME}
OUT_ROOT=${OUT_ROOT}
SAMPLE_ROOT=${SAMPLE_ROOT}
SAMPLES=${SAMPLES}
SEED=${SEED}
LAYERS=${LAYERS}
RUN_FIXED_GT_CAM=${RUN_FIXED_GT_CAM}
RUN_PREDICTION_CAM=${RUN_PREDICTION_CAM}
RUN_PRETRAINED_BACKBONE_ACTIVATION=${RUN_PRETRAINED_BACKBONE_ACTIVATION}
EOF

echo "============================================================"
echo "CAM analysis suite complete"
echo "============================================================"
echo "Shared sample:       ${SAMPLE_ROOT}"
[[ "${RUN_FIXED_GT_CAM}" == 1 ]] && echo "Fixed-GT:            ${OUT_ROOT}/fixed_gt"
[[ "${RUN_PREDICTION_CAM}" == 1 ]] && echo "Prediction CAM:      ${OUT_ROOT}/prediction"
[[ "${RUN_PRETRAINED_BACKBONE_ACTIVATION}" == 1 ]] && \
    echo "Pretrained backbone: ${OUT_ROOT}/pretrained_backbone"
