#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

FIXED_ROOT="${FIXED_ROOT:-/media/HDD2/XCX/exp_2/backbone_activation_fixed_sets/seed2026_v1}"
SAMPLES="${SAMPLES:-5}"
DATASETS="${DATASETS:-ruod uiis10k}"
R50_DEVICE="${R50_DEVICE:-cuda:4}"
VITS_DEVICE="${VITS_DEVICE:-cuda:5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/backbone_eigen_5plus5_${STAMP}}"

R50_MODELS="${R50_MODELS:-imagenet_dino100e_backbone,realuw_dino100e_backbone,synthetic5_dino100e_backbone}"
VITS_MODELS="${VITS_MODELS:-imagenet_vits_backbone,realuw_vits_backbone,synthetic5_vits_backbone}"
R50_CONFIG="${R50_CONFIG:-scripts/exp_2/features/further_features/models.pretrained_backbone_activation.json}"
VITS_CONFIG="${VITS_CONFIG:-scripts/exp_2/features/further_features/models.pretrained_backbone_activation.vits.json}"

mkdir -p "${OUT_ROOT}/logs"

run_arch() {
    local dataset="$1"
    local manifest="$2"
    local backbone="$3"
    local config="$4"
    local models="$5"
    local layer="$6"
    local device="$7"
    local reference="$8"
    local root="${OUT_ROOT}/${dataset}/${backbone}"
    local features="${root}/features"
    local rendered="${root}/rendered_eigen"
    local log="${OUT_ROOT}/logs/${dataset}_${backbone}.log"

    echo "============================================================"
    echo "Dataset:   ${dataset}"
    echo "Backbone:  ${backbone}"
    echo "Models:    ${models}"
    echo "Layer:     ${layer}"
    echo "Device:    ${device}"
    echo "Output:    ${root}"
    echo "============================================================"

    {
        python -m tools.exp_2.backbone_analysis.extract_backbone_features \
            --manifest "${manifest}" \
            --models-config "${config}" \
            --models "${models}" \
            --out-dir "${features}" \
            --layers "${layer}" \
            --device "${device}" \
            --batch-size 1 \
            --pooling avg \
            --save-spatial \
            --spatial-samples 0 \
            --spatial-dtype float32 \
            --overwrite

        python -m tools.exp_2.backbone_analysis.render_feature_activation \
            --feature-root "${features}" \
            --manifest "${manifest}" \
            --models "${models}" \
            --layers "${layer}" \
            --variant clean \
            --aggregation eigen \
            --normalization-mode reference-dataset-per-layer \
            --normalization-reference-model "${reference}" \
            --low-percentile 1 \
            --high-percentile 99 \
            --model-workers 1 \
            --skip-raw-activation \
            --out-dir "${rendered}" \
            --overwrite
    } 2>&1 | tee "${log}"
}

for dataset in ${DATASETS}; do
    source_manifest="${FIXED_ROOT}/${dataset}100/manifest.jsonl"
    if [[ ! -s "${source_manifest}" ]]; then
        echo "Missing source manifest: ${source_manifest}" >&2
        exit 1
    fi
    dataset_root="${OUT_ROOT}/${dataset}"
    mkdir -p "${dataset_root}"
    manifest="${dataset_root}/manifest_${SAMPLES}.jsonl"
    head -n "${SAMPLES}" "${source_manifest}" > "${manifest}"

    run_arch \
        "${dataset}" "${manifest}" "r50" \
        "${R50_CONFIG}" "${R50_MODELS}" "res4" \
        "${R50_DEVICE}" "imagenet_dino100e_backbone"
    run_arch \
        "${dataset}" "${manifest}" "vits" \
        "${VITS_CONFIG}" "${VITS_MODELS}" "block12" \
        "${VITS_DEVICE}" "imagenet_vits_backbone"
done

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=EigenCAM
SAMPLES_PER_DATASET=${SAMPLES}
DATASETS=${DATASETS}
R50_LAYER=res4
VITS_LAYER=block12
R50_FEATURE_AGGREGATION=eigen
VITS_FEATURE_AGGREGATION=eigen
EOF

echo
echo "Complete EigenCAM outputs: ${OUT_ROOT}"
for dataset in ${DATASETS}; do
    printf "%-10s R50 panels: " "${dataset}"
    find "${OUT_ROOT}/${dataset}/r50/rendered_eigen/panels/res4" \
        -type f -name '*.png' 2>/dev/null | wc -l
    printf "%-10s ViT panels: " "${dataset}"
    find "${OUT_ROOT}/${dataset}/vits/rendered_eigen/panels/block12" \
        -type f -name '*.png' 2>/dev/null | wc -l
done
