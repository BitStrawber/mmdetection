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
SMOOTH_SIGMA="${SMOOTH_SIGMA:-0}"
RESIZE_INTERPOLATION="${RESIZE_INTERPOLATION:-bilinear}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/backbone_eigen_reference_5plus5_${STAMP}}"

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
    local layers="$6"
    local device="$7"
    local reference="$8"
    local root="${OUT_ROOT}/${dataset}/${backbone}"
    local log="${OUT_ROOT}/logs/${dataset}_${backbone}.log"

    echo "============================================================"
    echo "Dataset:   ${dataset}"
    echo "Backbone:  ${backbone}"
    echo "Layers:    ${layers}"
    echo "Device:    ${device}"
    echo "Output:    ${root}"
    echo "============================================================"

    {
        python -m tools.exp_2.backbone_analysis.extract_backbone_features \
            --manifest "${manifest}" \
            --models-config "${config}" \
            --models "${models}" \
            --out-dir "${root}/features" \
            --layers "${layers}" \
            --device "${device}" \
            --batch-size 1 \
            --pooling avg \
            --save-spatial \
            --spatial-samples 0 \
            --spatial-dtype float32 \
            --overwrite

        python -m tools.exp_2.backbone_analysis.render_feature_activation \
            --feature-root "${root}/features" \
            --manifest "${manifest}" \
            --models "${models}" \
            --layers "${layers}" \
            --variant clean \
            --aggregation eigen \
            --colormap jet \
            --smooth-sigma "${SMOOTH_SIGMA}" \
            --resize-interpolation "${RESIZE_INTERPOLATION}" \
            --normalization-mode reference-dataset-per-layer \
            --normalization-reference-model "${reference}" \
            --low-percentile 1 \
            --high-percentile 99 \
            --model-workers 1 \
            --skip-raw-activation \
            --out-dir "${root}/rendered_eigen_reference" \
            --overwrite
    } 2>&1 | tee "${log}"
}

for dataset in ${DATASETS}; do
    source_manifest="${FIXED_ROOT}/${dataset}100/manifest.jsonl"
    [[ -s "${source_manifest}" ]] || {
        echo "Missing source manifest: ${source_manifest}" >&2
        exit 1
    }
    dataset_root="${OUT_ROOT}/${dataset}"
    mkdir -p "${dataset_root}"
    manifest="${dataset_root}/manifest_${SAMPLES}.jsonl"
    head -n "${SAMPLES}" "${source_manifest}" > "${manifest}"

    run_arch \
        "${dataset}" "${manifest}" "r50" \
        "${R50_CONFIG}" "${R50_MODELS}" "res2,res3,res4,res5" \
        "${R50_DEVICE}" "imagenet_dino100e_backbone"

    run_arch \
        "${dataset}" "${manifest}" "vits" \
        "${VITS_CONFIG}" "${VITS_MODELS}" "block1,block4,block8,block12" \
        "${VITS_DEVICE}" "imagenet_vits_backbone"
done

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=EigenCAM
SAMPLES_PER_DATASET=${SAMPLES}
DATASETS=${DATASETS}
R50_LAYERS=res2,res3,res4,res5
VITS_LAYERS=block1,block4,block8,block12
COLORMAP=jet
SMOOTH_SIGMA=${SMOOTH_SIGMA}
RESIZE_INTERPOLATION=${RESIZE_INTERPOLATION}
EOF

echo
echo "Complete reference-style EigenCAM outputs: ${OUT_ROOT}"
for dataset in ${DATASETS}; do
    printf "%-10s R50 progression panels: " "${dataset}"
    find "${OUT_ROOT}/${dataset}/r50/rendered_eigen_reference/panels_5x4" \
        -type f -name '*.png' 2>/dev/null | wc -l
    printf "%-10s ViT progression panels: " "${dataset}"
    find "${OUT_ROOT}/${dataset}/vits/rendered_eigen_reference/panels_5x4" \
        -type f -name '*.png' 2>/dev/null | wc -l
done
