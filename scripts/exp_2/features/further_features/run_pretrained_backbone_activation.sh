#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

SAMPLE_ROOT="${SAMPLE_ROOT:?SAMPLE_ROOT must point to the shared sampled RUOD set}"
MODELS_CONFIG="${PRETRAINED_MODELS_CONFIG:-${SCRIPT_DIR}/models.pretrained_backbone_activation.json}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
FEATURE_ROOT="${FEATURE_ROOT:-${OUT_ROOT}/feature_store}"
RENDER_ROOT="${RENDER_ROOT:-${OUT_ROOT}/rendered}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
DEVICE="${PRETRAINED_DEVICE:-cuda:0}"
REFERENCE_MODEL="${PRETRAINED_REFERENCE_MODEL:-imagenet_dino100e_backbone}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99}"
MODEL_WORKERS="${PRETRAINED_RENDER_MODEL_WORKERS:-4}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-3}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
[[ -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || {
    echo "Error: shared sample manifest is missing: ${SAMPLE_ROOT}/manifest.jsonl" >&2
    exit 1
}
[[ -s "${MODELS_CONFIG}" ]] || {
    echo "Error: pretrained model config is missing: ${MODELS_CONFIG}" >&2
    exit 1
}

mapfile -t MODEL_IDS < <(
    python - "${MODELS_CONFIG}" <<'PY'
import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    value = json.load(handle)
for item in value.get('models', []):
    if item.get('kind') != 'backbone':
        raise SystemExit('Every pretrained activation model must use kind=backbone')
    print(item['id'])
PY
)
MODEL_CSV="$(IFS=,; echo "${MODEL_IDS[*]}")"
SAMPLES="$(wc -l < "${SAMPLE_ROOT}/manifest.jsonl")"

echo "============================================================"
echo "Bare pretrained backbone activation"
echo "============================================================"
echo "Sample:          ${SAMPLE_ROOT}/manifest.jsonl (${SAMPLES} images)"
echo "Models:          ${MODEL_IDS[*]}"
echo "Layers:          ${LAYERS}"
echo "Device:          ${DEVICE}"
echo "Reference model: ${REFERENCE_MODEL}"
echo "Output:          ${OUT_ROOT}"
echo "============================================================"

if [[ "${RUN_EXTRACT}" == 1 ]]; then
    command=(
        python -m tools.exp_2.backbone_analysis.extract_backbone_features
        --manifest "${SAMPLE_ROOT}/manifest.jsonl"
        --models-config "${MODELS_CONFIG}"
        --out-dir "${FEATURE_ROOT}"
        --models "${MODEL_CSV}"
        --variants clean
        --layers "${LAYERS}"
        --device "${DEVICE}"
        --pooling avg
        --save-spatial
        --spatial-samples "${SAMPLES}"
        --spatial-dtype float16
    )
    [[ "${OVERWRITE}" == 1 ]] && command+=(--overwrite)
    "${command[@]}" 2>&1 | tee "${LOG_ROOT}/extract.log"
fi

if [[ "${RUN_RENDER}" == 1 ]]; then
    for mode in reference-dataset-per-layer reference-per-sample-layer; do
        name="imagenet_reference_dataset_p1_p99"
        [[ "${mode}" == reference-per-sample-layer ]] && \
            name="imagenet_reference_per_sample_p1_p99"
        command=(
            python -m tools.exp_2.backbone_analysis.render_feature_activation
            --feature-root "${FEATURE_ROOT}"
            --manifest "${SAMPLE_ROOT}/manifest.jsonl"
            --models "${MODEL_CSV}"
            --layers "${LAYERS}"
            --variant clean
            --normalization-mode "${mode}"
            --normalization-reference-model "${REFERENCE_MODEL}"
            --low-percentile "${LOW_PERCENTILE}"
            --high-percentile "${HIGH_PERCENTILE}"
            --model-workers "${MODEL_WORKERS}"
            --skip-raw-activation
            --png-compress-level "${PNG_COMPRESS_LEVEL}"
            --out-dir "${RENDER_ROOT}/${name}"
        )
        [[ "${OVERWRITE}" == 1 ]] && command+=(--overwrite)
        "${command[@]}" 2>&1 | tee "${LOG_ROOT}/render_${name}.log"
    done
fi

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=bare_pretrained_backbone_activation
FEATURE_AGGREGATION=mean_abs_channel
FEATURE_ROOT=${FEATURE_ROOT}
RENDER_ROOT=${RENDER_ROOT}
MODELS=${MODEL_CSV}
LAYERS=${LAYERS}
REFERENCE_MODEL=${REFERENCE_MODEL}
EOF

echo "Pretrained backbone activation complete: ${OUT_ROOT}"
