#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
RUOD_ANN="${RUOD_ANN:-${RUOD_ROOT}/annotations/instances_val.json}"
MODELS_CONFIG="${MODELS_CONFIG:-${SCRIPT_DIR}/models.fixed_gt_cam.json}"
SAMPLE_ROOT="${SAMPLE_ROOT:?SAMPLE_ROOT must point to the shared sampled RUOD set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
CAM_ROOT="${CAM_ROOT:-${OUT_ROOT}/prediction_cam}"
RENDER_ROOT="${RENDER_ROOT:-${OUT_ROOT}/rendered}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

LAYERS="${LAYERS:-res2,res3,res4,res5}"
CASCADE_STAGE="${CASCADE_STAGE:--1}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.30}"
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-10}"
MINIMUM_BOX_AREA="${MINIMUM_BOX_AREA:-4}"
MATCH_IOU_THRESHOLD="${MATCH_IOU_THRESHOLD:-0.50}"
REFERENCE_MODEL="${REFERENCE_MODEL:-imagenet_dino100e_ruod_cascade}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99}"
DISPLAY_GAMMA="${DISPLAY_GAMMA:-1.0}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.48}"
PANEL_LIMIT="${PANEL_LIMIT:-0}"
DEVICES="${DEVICES:-cuda:0}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_LEGACY_AGGREGATE="${RUN_LEGACY_AGGREGATE:-1}"
LEGACY_AGGREGATE_ROOT="${LEGACY_AGGREGATE_ROOT:-${OUT_ROOT}/legacy_image_aggregate}"
LEGACY_AGGREGATION="${LEGACY_AGGREGATION:-sum}"
LEGACY_STYLES="${LEGACY_STYLES:-legacy_jet,legacy_turbo_gamma05}"
RESUME="${RESUME:-1}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
[[ -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || {
    echo "Error: shared sample manifest is missing: ${SAMPLE_ROOT}/manifest.jsonl" >&2
    exit 1
}

mapfile -t MODEL_IDS < <(
    python - "${MODELS_CONFIG}" <<'PY'
import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    value = json.load(handle)
for item in value.get('models', []):
    print(item['id'])
PY
)
[[ "${#MODEL_IDS[@]}" -gt 0 ]] || {
    echo "Error: models config has no detectors: ${MODELS_CONFIG}" >&2
    exit 1
}
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
for index in "${!DEVICE_ARRAY[@]}"; do
    DEVICE_ARRAY[$index]="$(echo "${DEVICE_ARRAY[$index]}" | xargs)"
done
MODEL_CSV="$(IFS=,; echo "${MODEL_IDS[*]}")"

echo "============================================================"
echo "Prediction-conditioned Cascade R-CNN XGradCAM"
echo "============================================================"
echo "Sample:             ${SAMPLE_ROOT}/manifest.jsonl"
echo "Models:             ${MODEL_IDS[*]}"
echo "Layers:             ${LAYERS}"
echo "Score threshold:    ${SCORE_THRESHOLD}"
echo "Maximum boxes:      ${MAX_PREDICTIONS_PER_IMAGE} per image/model"
echo "GT match IoU:       ${MATCH_IOU_THRESHOLD}"
echo "Devices:            ${DEVICE_ARRAY[*]}"
echo "Output:             ${OUT_ROOT}"
echo "============================================================"

if [[ "${RUN_EXTRACT}" == 1 ]]; then
    pids=()
    names=()
    failures=0
    running=0
    for index in "${!MODEL_IDS[@]}"; do
        model="${MODEL_IDS[$index]}"
        device="${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}"
        log="${LOG_ROOT}/extract_${model}.log"
        command=(
            python "${SCRIPT_DIR}/extract_prediction_xgradcam.py"
            --manifest "${SAMPLE_ROOT}/manifest.jsonl"
            --annotation-file "${RUOD_ANN}"
            --models-config "${MODELS_CONFIG}"
            --models "${model}"
            --out-dir "${CAM_ROOT}"
            --layers "${LAYERS}"
            --device "${device}"
            --cascade-stage "${CASCADE_STAGE}"
            --score-threshold "${SCORE_THRESHOLD}"
            --max-predictions-per-image "${MAX_PREDICTIONS_PER_IMAGE}"
            --minimum-box-area "${MINIMUM_BOX_AREA}"
            --match-iou-threshold "${MATCH_IOU_THRESHOLD}"
        )
        [[ "${RESUME}" == 1 ]] && command+=(--resume)
        echo "START prediction model=${model} device=${device} log=${log}"
        "${command[@]}" > "${log}" 2>&1 &
        pids+=("$!")
        names+=("${model}")
        running=$((running + 1))
        if [[ "${running}" -ge "${PARALLEL_MODELS}" ]]; then
            if wait "${pids[0]}"; then
                echo "DONE prediction model=${names[0]}"
            else
                echo "FAILED prediction model=${names[0]}" >&2
                failures=$((failures + 1))
            fi
            pids=("${pids[@]:1}")
            names=("${names[@]:1}")
            running=$((running - 1))
        fi
    done
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "DONE prediction model=${names[$index]}"
        else
            echo "FAILED prediction model=${names[$index]}" >&2
            failures=$((failures + 1))
        fi
    done
    [[ "${failures}" -eq 0 ]] || exit 1
fi

if [[ "${RUN_RENDER}" == 1 ]]; then
    render_args=(
        --cam-root "${CAM_ROOT}"
        --out-dir "${RENDER_ROOT}"
        --models "${MODEL_CSV}"
        --layers "${LAYERS}"
        --reference-model "${REFERENCE_MODEL}"
        --low-percentile "${LOW_PERCENTILE}"
        --high-percentile "${HIGH_PERCENTILE}"
        --gamma "${DISPLAY_GAMMA}"
        --overlay-alpha "${OVERLAY_ALPHA}"
        --panel-limit "${PANEL_LIMIT}"
    )
    [[ "${RESUME}" == 1 ]] && render_args+=(--overwrite)
    python "${SCRIPT_DIR}/render_prediction_xgradcam.py" \
        "${render_args[@]}" 2>&1 | tee "${LOG_ROOT}/render.log"
fi

if [[ "${RUN_LEGACY_AGGREGATE}" == 1 ]]; then
    legacy_args=(
        --cam-root "${CAM_ROOT}"
        --out-dir "${LEGACY_AGGREGATE_ROOT}"
        --models "${MODEL_CSV}"
        --layers "${LAYERS}"
        --aggregation "${LEGACY_AGGREGATION}"
        --styles "${LEGACY_STYLES}"
    )
    [[ "${RESUME}" == 1 ]] && legacy_args+=(--overwrite)
    python "${SCRIPT_DIR}/render_prediction_legacy_aggregate.py" \
        "${legacy_args[@]}" 2>&1 | tee "${LOG_ROOT}/render_legacy_aggregate.log"
fi

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=prediction_conditioned_xgradcam
CAM_ROOT=${CAM_ROOT}
RENDER_ROOT=${RENDER_ROOT}
LEGACY_AGGREGATE_ROOT=${LEGACY_AGGREGATE_ROOT}
LEGACY_AGGREGATION=${LEGACY_AGGREGATION}
LEGACY_STYLES=${LEGACY_STYLES}
MODELS=${MODEL_CSV}
LAYERS=${LAYERS}
EOF

echo "Prediction XGradCAM complete: ${OUT_ROOT}"
