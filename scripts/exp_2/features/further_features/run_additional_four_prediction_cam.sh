#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CAM_BASE="${CAM_BASE:-/media/HDD2/XCX/exp_2/frozen_backbone_prediction_cam_additional_20261105}"
RUNNER="${SCRIPT_DIR}/run_prediction_xgradcam_analysis.sh"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco}"
RUOD_SAMPLE="${RUOD_SAMPLE:-/media/HDD2/XCX/exp_2/frozen_backbone_prediction_cam_20261029/samples/ruod_100}"
UIIS_SAMPLE="${UIIS_SAMPLE:-/media/HDD2/XCX/exp_2/frozen_backbone_prediction_cam_20261029/samples/uiis10k_100}"

R50_LANE_GPU="${R50_LANE_GPU:-cuda:0}"
VITS_LANE_GPU="${VITS_LANE_GPU:-cuda:1}"

SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-30}"

for path in "${RUNNER}" "${RUOD_SAMPLE}/manifest.jsonl" "${UIIS_SAMPLE}/manifest.jsonl"; do
    [[ -s "${path}" ]] || {
        echo "Error: missing required file: ${path}" >&2
        exit 1
    }
done
for config in "${CAM_BASE}"/configs/models.*.json; do
    [[ -s "${config}" ]] || {
        echo "Error: missing model config: ${config}" >&2
        exit 1
    }
done

mkdir -p "${CAM_BASE}/output" "${CAM_BASE}/logs"

run_group() {
    local dataset="$1"
    local arch="$2"
    local layer="$3"
    local data_root="$4"
    local sample_root="$5"
    local reference="$6"
    local device="$7"
    local group="${dataset}_${arch}"
    local out_root="${CAM_BASE}/output/${group}"
    local log="${CAM_BASE}/logs/${group}.runner.log"

    if [[ -s "${out_root}/COMPLETE.env" ]]; then
        echo "SKIP completed group=${group}"
        return
    fi

    echo "START group=${group} device=${device} log=${log}"
    RUOD_ROOT="${data_root}" \
    RUOD_ANN="${sample_root}/annotations.coco.json" \
    SAMPLE_ROOT="${sample_root}" \
    MODELS_CONFIG="${CAM_BASE}/configs/models.${dataset}.${arch}.json" \
    OUT_ROOT="${out_root}" \
    LAYERS="${layer}" \
    CASCADE_STAGE=-1 \
    SCORE_THRESHOLD="${SCORE_THRESHOLD}" \
    INFERENCE_SCORE_THRESHOLD="${SCORE_THRESHOLD}" \
    MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE}" \
    MINIMUM_BOX_AREA=4 \
    MATCH_IOU_THRESHOLD=0.50 \
    TARGET_MODE=bbox \
    REFERENCE_MODEL="${reference}" \
    DEVICES="${device}" \
    PARALLEL_MODELS=1 \
    SAVE_PREDICTION_IMAGES=0 \
    COMPUTE_CAM_METRICS=0 \
    COLORMAP=jet \
    DRAW_PREDICTION_BOXES=0 \
    OVERLAY_ORIGINAL=0 \
    NORMALIZATION_STRATEGY=reference \
    CAM_INTERPOLATION=linear \
    CAM_SMOOTHING_SIGMA=0 \
    RUN_LEGACY_AGGREGATE=0 \
    RESUME=1 \
    bash "${RUNNER}" > "${log}" 2>&1
    echo "DONE group=${group}"
}

run_r50_lane() {
    run_group ruod r50 res4 "${RUOD_ROOT}" "${RUOD_SAMPLE}" \
        ruod_imagenet_r50 "${R50_LANE_GPU}"
    run_group uiis10k r50 res4 "${UIIS_ROOT}" "${UIIS_SAMPLE}" \
        uiis10k_imagenet_r50 "${R50_LANE_GPU}"
}

run_vits_lane() {
    run_group ruod vits block12 "${RUOD_ROOT}" "${RUOD_SAMPLE}" \
        ruod_imagenet_vits "${VITS_LANE_GPU}"
    run_group uiis10k vits block12 "${UIIS_ROOT}" "${UIIS_SAMPLE}" \
        uiis10k_imagenet_vits "${VITS_LANE_GPU}"
}

run_r50_lane > "${CAM_BASE}/logs/lane_r50.log" 2>&1 &
r50_pid=$!
run_vits_lane > "${CAM_BASE}/logs/lane_vits.log" 2>&1 &
vits_pid=$!

echo "${r50_pid}" > "${CAM_BASE}/logs/lane_r50.pid"
echo "${vits_pid}" > "${CAM_BASE}/logs/lane_vits.pid"

failed=0
wait "${r50_pid}" || failed=1
wait "${vits_pid}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
    echo "One or more CAM lanes failed" >&2
    exit 1
fi

echo "All four CAM groups complete: ${CAM_BASE}/output"
