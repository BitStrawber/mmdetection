#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:-${RUOD_ROOT}/val}"
RUOD_ANN="${RUOD_ANN:-${RUOD_ROOT}/annotations/instances_val.json}"
MODELS_CONFIG="${MODELS_CONFIG:-${SCRIPT_DIR}/models.fixed_gt_cam.json}"

SAMPLES="${SAMPLES:-50}"
SEED="${SEED:-2026}"
MATERIALIZE="${MATERIALIZE:-none}"
MAX_INSTANCES_PER_IMAGE="${MAX_INSTANCES_PER_IMAGE:-5}"
MINIMUM_BOX_AREA="${MINIMUM_BOX_AREA:-4}"
INSTANCE_ORDER="${INSTANCE_ORDER:-area-desc}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
CASCADE_STAGE="${CASCADE_STAGE:--1}"
REFERENCE_MODEL="${REFERENCE_MODEL:-imagenet_dino100e_ruod_cascade}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99}"
DISPLAY_GAMMA="${DISPLAY_GAMMA:-1.0}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.48}"
PANEL_LIMIT="${PANEL_LIMIT:-0}"

DEVICES="${DEVICES:-cuda:0}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_IMAGE_AGGREGATE="${RUN_IMAGE_AGGREGATE:-1}"
RESUME="${RESUME:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-fixed_gt_xgradcam_ruod${SAMPLES}_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/further_features/${RUN_NAME}}"
SAMPLE_ROOT="${SAMPLE_ROOT:-${OUT_ROOT}/sample}"
CAM_ROOT="${CAM_ROOT:-${OUT_ROOT}/fixed_gt_cam}"
RENDER_ROOT="${RENDER_ROOT:-${OUT_ROOT}/rendered}"
IMAGE_AGGREGATE_ROOT="${IMAGE_AGGREGATE_ROOT:-${OUT_ROOT}/image_aggregate}"
IMAGE_AGGREGATION="${IMAGE_AGGREGATION:-max}"
IMAGE_AGGREGATE_VIEW="${IMAGE_AGGREGATE_VIEW:-pure}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

require_file() {
    [[ -s "$1" ]] || {
        echo "Error: required file is missing or empty: $1" >&2
        exit 1
    }
}

require_dir() {
    [[ -d "$1" ]] || {
        echo "Error: required directory is missing: $1" >&2
        exit 1
    }
}

require_file "${RUOD_ANN}"
require_file "${MODELS_CONFIG}"
require_dir "${RUOD_IMAGE_ROOT}"

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
    echo "Error: models config has no models: ${MODELS_CONFIG}" >&2
    exit 1
}
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
for index in "${!DEVICE_ARRAY[@]}"; do
    DEVICE_ARRAY[$index]="$(echo "${DEVICE_ARRAY[$index]}" | xargs)"
done
[[ "${#DEVICE_ARRAY[@]}" -gt 0 ]] || DEVICE_ARRAY=(cuda:0)

echo "============================================================"
echo "Fixed-GT Cascade R-CNN XGradCAM analysis"
echo "============================================================"
echo "RUOD images:             ${RUOD_IMAGE_ROOT}"
echo "RUOD annotations:        ${RUOD_ANN}"
echo "Models config:           ${MODELS_CONFIG}"
echo "Models:                  ${MODEL_IDS[*]}"
echo "Samples / seed:          ${SAMPLES} / ${SEED}"
echo "Max instances per image: ${MAX_INSTANCES_PER_IMAGE}"
echo "Instance order:          ${INSTANCE_ORDER}"
echo "Layers:                  ${LAYERS}"
echo "Cascade stage:           ${CASCADE_STAGE} (-1 = last)"
echo "Reference detector:      ${REFERENCE_MODEL}"
echo "Normalization outputs:   independent + ImageNet-reference"
echo "Percentiles:             ${LOW_PERCENTILE}, ${HIGH_PERCENTILE}"
echo "Devices:                 ${DEVICE_ARRAY[*]}"
echo "Parallel models:         ${PARALLEL_MODELS}"
echo "Output:                  ${OUT_ROOT}"
echo "============================================================"

python "${SCRIPT_DIR}/check_fixed_gt_cam_inputs.py" \
    --annotation-file "${RUOD_ANN}" \
    --image-root "${RUOD_IMAGE_ROOT}" \
    --models-config "${MODELS_CONFIG}" \
    2>&1 | tee "${LOG_ROOT}/00_check_inputs.log"

if [[ "${RUN_SAMPLE}" == 1 ]]; then
    sample_args=(
        --annotation-file "${RUOD_ANN}"
        --image-root "${RUOD_IMAGE_ROOT}"
        --out-dir "${SAMPLE_ROOT}"
        --samples "${SAMPLES}"
        --seed "${SEED}"
        --minimum-instances 1
        --materialize "${MATERIALIZE}"
    )
    [[ "${RESUME}" == 1 && -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || \
        python "${SCRIPT_DIR}/sample_fixed_gt_ruod.py" \
            "${sample_args[@]}" \
            2>&1 | tee "${LOG_ROOT}/01_sample.log"
fi
require_file "${SAMPLE_ROOT}/manifest.jsonl"

if [[ "${RUN_EXTRACT}" == 1 ]]; then
    worker_pids=()
    worker_models=()
    running=0
    failures=0
    for index in "${!MODEL_IDS[@]}"; do
        model="${MODEL_IDS[$index]}"
        device="${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}"
        log="${LOG_ROOT}/02_extract_${model}.log"
        command=(
            python "${SCRIPT_DIR}/extract_fixed_gt_xgradcam.py"
            --manifest "${SAMPLE_ROOT}/manifest.jsonl"
            --annotation-file "${RUOD_ANN}"
            --models-config "${MODELS_CONFIG}"
            --models "${model}"
            --out-dir "${CAM_ROOT}"
            --layers "${LAYERS}"
            --device "${device}"
            --cascade-stage "${CASCADE_STAGE}"
            --max-instances-per-image "${MAX_INSTANCES_PER_IMAGE}"
            --minimum-box-area "${MINIMUM_BOX_AREA}"
            --instance-order "${INSTANCE_ORDER}"
        )
        [[ "${RESUME}" == 1 ]] && command+=(--resume)
        echo "START model=${model} device=${device} log=${log}"
        "${command[@]}" > "${log}" 2>&1 &
        worker_pids+=("$!")
        worker_models+=("${model}")
        running=$((running + 1))
        if [[ "${PARALLEL_MODELS}" -gt 0 && "${running}" -ge "${PARALLEL_MODELS}" ]]; then
            first_pid="${worker_pids[0]}"
            first_model="${worker_models[0]}"
            if wait "${first_pid}"; then
                echo "DONE model=${first_model}"
            else
                echo "FAILED model=${first_model}" >&2
                failures=$((failures + 1))
            fi
            worker_pids=("${worker_pids[@]:1}")
            worker_models=("${worker_models[@]:1}")
            running=$((running - 1))
        fi
    done
    for index in "${!worker_pids[@]}"; do
        if wait "${worker_pids[$index]}"; then
            echo "DONE model=${worker_models[$index]}"
        else
            echo "FAILED model=${worker_models[$index]}" >&2
            failures=$((failures + 1))
        fi
    done
    [[ "${failures}" -eq 0 ]] || {
        echo "Error: ${failures} CAM extraction workers failed" >&2
        exit 1
    }
fi

MODEL_CSV="$(IFS=,; echo "${MODEL_IDS[*]}")"
python "${SCRIPT_DIR}/index_fixed_gt_xgradcam.py" \
    --cam-root "${CAM_ROOT}" \
    --models "${MODEL_CSV}" \
    --layers "${LAYERS}" \
    --require-complete \
    2>&1 | tee "${LOG_ROOT}/03_index.log"

if [[ "${RUN_RENDER}" == 1 ]]; then
    render_args=(
        --cam-root "${CAM_ROOT}"
        --out-dir "${RENDER_ROOT}"
        --reference-model "${REFERENCE_MODEL}"
        --models "${MODEL_CSV}"
        --layers "${LAYERS}"
        --low-percentile "${LOW_PERCENTILE}"
        --high-percentile "${HIGH_PERCENTILE}"
        --gamma "${DISPLAY_GAMMA}"
        --overlay-alpha "${OVERLAY_ALPHA}"
        --panel-limit "${PANEL_LIMIT}"
    )
    [[ "${RESUME}" == 1 ]] && render_args+=(--overwrite)
    python "${SCRIPT_DIR}/render_fixed_gt_xgradcam.py" \
        "${render_args[@]}" \
        2>&1 | tee "${LOG_ROOT}/04_render.log"
fi

if [[ "${RUN_IMAGE_AGGREGATE}" == 1 ]]; then
    aggregate_args=(
        --cam-root "${CAM_ROOT}"
        --out-dir "${IMAGE_AGGREGATE_ROOT}"
        --reference-model "${REFERENCE_MODEL}"
        --models "${MODEL_CSV}"
        --layers "${LAYERS}"
        --aggregation "${IMAGE_AGGREGATION}"
        --view "${IMAGE_AGGREGATE_VIEW}"
        --low-percentile "${LOW_PERCENTILE}"
        --high-percentile "${HIGH_PERCENTILE}"
        --gamma "${DISPLAY_GAMMA}"
        --overlay-alpha "${OVERLAY_ALPHA}"
    )
    [[ "${RESUME}" == 1 ]] && aggregate_args+=(--overwrite)
    python "${SCRIPT_DIR}/render_fixed_gt_image_aggregate.py" \
        "${aggregate_args[@]}" \
        2>&1 | tee "${LOG_ROOT}/04b_render_image_aggregate.log"
fi

if [[ "${RUN_PLOTS}" == 1 ]]; then
    python "${SCRIPT_DIR}/plot_fixed_gt_cam_metrics.py" \
        --metrics-tsv "${RENDER_ROOT}/metrics/instance_layer_metrics.tsv" \
        --out-dir "${RENDER_ROOT}/figures" \
        --model-order "${MODEL_CSV}" \
        --layer-order "${LAYERS}" \
        2>&1 | tee "${LOG_ROOT}/05_plots.log"
fi

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
RUN_NAME=${RUN_NAME}
OUT_ROOT=${OUT_ROOT}
SAMPLE_ROOT=${SAMPLE_ROOT}
CAM_ROOT=${CAM_ROOT}
RENDER_ROOT=${RENDER_ROOT}
IMAGE_AGGREGATE_ROOT=${IMAGE_AGGREGATE_ROOT}
IMAGE_AGGREGATION=${IMAGE_AGGREGATION}
IMAGE_AGGREGATE_VIEW=${IMAGE_AGGREGATE_VIEW}
REFERENCE_MODEL=${REFERENCE_MODEL}
MODELS=${MODEL_CSV}
LAYERS=${LAYERS}
EOF

echo "============================================================"
echo "Fixed-GT XGradCAM analysis complete"
echo "============================================================"
echo "Raw CAM:    ${CAM_ROOT}/raw_cam"
echo "Two views:  ${RENDER_ROOT}"
echo "Metrics:    ${RENDER_ROOT}/metrics"
echo "Figures:    ${RENDER_ROOT}/figures"
echo "Panels:     ${RENDER_ROOT}/panels"
