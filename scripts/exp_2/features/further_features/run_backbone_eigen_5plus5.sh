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
R50_BATCH_SIZE="${R50_BATCH_SIZE:-1}"
VITS_BATCH_SIZE="${VITS_BATCH_SIZE:-1}"
RENDER_MODEL_WORKERS="${RENDER_MODEL_WORKERS:-3}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-1}"
PARALLEL_ARCH="${PARALLEL_ARCH:-1}"
ACTIVATION_AGGREGATION="${ACTIVATION_AGGREGATION:-eigen}"
EIGEN_COMPONENTS="${EIGEN_COMPONENTS:-3}"
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
    local batch_size="$9"
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
            --batch-size "${batch_size}" \
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
            --aggregation "${ACTIVATION_AGGREGATION}" \
            --eigen-components "${EIGEN_COMPONENTS}" \
            --colormap jet \
            --smooth-sigma 0 \
            --resize-interpolation bilinear \
            --normalization-mode reference-dataset-per-layer \
            --normalization-reference-model "${reference}" \
            --low-percentile 1 \
            --high-percentile 99 \
            --model-workers "${RENDER_MODEL_WORKERS}" \
            --png-compress-level "${PNG_COMPRESS_LEVEL}" \
            --skip-5x4-panels \
            --skip-layer-evolution-panels \
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

    if [[ "${PARALLEL_ARCH}" == 1 ]]; then
        run_arch \
            "${dataset}" "${manifest}" "r50" \
            "${R50_CONFIG}" "${R50_MODELS}" "res4" \
            "${R50_DEVICE}" "imagenet_dino100e_backbone" \
            "${R50_BATCH_SIZE}" &
        r50_pid=$!
        run_arch \
            "${dataset}" "${manifest}" "vits" \
            "${VITS_CONFIG}" "${VITS_MODELS}" "block12" \
            "${VITS_DEVICE}" "imagenet_vits_backbone" \
            "${VITS_BATCH_SIZE}" &
        vit_pid=$!
        failure=0
        wait "${r50_pid}" || failure=1
        wait "${vit_pid}" || failure=1
        [[ "${failure}" == 0 ]] || exit 1
    else
        run_arch \
            "${dataset}" "${manifest}" "r50" \
            "${R50_CONFIG}" "${R50_MODELS}" "res4" \
            "${R50_DEVICE}" "imagenet_dino100e_backbone" \
            "${R50_BATCH_SIZE}"
        run_arch \
            "${dataset}" "${manifest}" "vits" \
            "${VITS_CONFIG}" "${VITS_MODELS}" "block12" \
            "${VITS_DEVICE}" "imagenet_vits_backbone" \
            "${VITS_BATCH_SIZE}"
    fi
done

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=EigenCAM
SAMPLES_PER_DATASET=${SAMPLES}
DATASETS=${DATASETS}
R50_LAYER=res4
VITS_LAYER=block12
R50_FEATURE_AGGREGATION=${ACTIVATION_AGGREGATION}
VITS_FEATURE_AGGREGATION=${ACTIVATION_AGGREGATION}
EIGEN_COMPONENTS=${EIGEN_COMPONENTS}
COLORMAP=jet
SMOOTH_SIGMA=0
RESIZE_INTERPOLATION=bilinear
R50_BATCH_SIZE=${R50_BATCH_SIZE}
VITS_BATCH_SIZE=${VITS_BATCH_SIZE}
RENDER_MODEL_WORKERS=${RENDER_MODEL_WORKERS}
PNG_COMPRESS_LEVEL=${PNG_COMPRESS_LEVEL}
PARALLEL_ARCH=${PARALLEL_ARCH}
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
