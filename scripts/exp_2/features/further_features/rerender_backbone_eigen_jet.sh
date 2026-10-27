#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/backbone_eigen_ruod5_uiis5_20261027_184846}"
R50_MODELS="${R50_MODELS:-imagenet_dino100e_backbone,realuw_dino100e_backbone,synthetic5_dino100e_backbone}"
VITS_MODELS="${VITS_MODELS:-imagenet_vits_backbone,realuw_vits_backbone,synthetic5_vits_backbone}"

render_group() {
    local group_root="$1"
    local manifest="$2"
    local models="$3"
    local layer="$4"
    local reference="$5"

    [[ -s "${manifest}" ]] || {
        echo "Missing manifest: ${manifest}" >&2
        exit 1
    }
    [[ -d "${group_root}/features" ]] || {
        echo "Missing feature root: ${group_root}/features" >&2
        exit 1
    }

    python -m tools.exp_2.backbone_analysis.render_feature_activation \
        --feature-root "${group_root}/features" \
        --manifest "${manifest}" \
        --models "${models}" \
        --layers "${layer}" \
        --variant clean \
        --aggregation eigen \
        --colormap jet \
        --normalization-mode reference-dataset-per-layer \
        --normalization-reference-model "${reference}" \
        --low-percentile 1 \
        --high-percentile 99 \
        --model-workers 1 \
        --skip-raw-activation \
        --out-dir "${group_root}/rendered_eigen_jet" \
        --overwrite
}

render_group \
    "${OUT_ROOT}/ruod/r50" \
    "${OUT_ROOT}/ruod/manifest_5.jsonl" \
    "${R50_MODELS}" res4 imagenet_dino100e_backbone

render_group \
    "${OUT_ROOT}/ruod/vits" \
    "${OUT_ROOT}/ruod/manifest_5.jsonl" \
    "${VITS_MODELS}" block12 imagenet_vits_backbone

render_group \
    "${OUT_ROOT}/uiis10k/r50" \
    "${OUT_ROOT}/uiis10k/manifest_5.jsonl" \
    "${R50_MODELS}" res4 imagenet_dino100e_backbone

render_group \
    "${OUT_ROOT}/uiis10k/vits" \
    "${OUT_ROOT}/uiis10k/manifest_5.jsonl" \
    "${VITS_MODELS}" block12 imagenet_vits_backbone

count="$(
    find \
        "${OUT_ROOT}/ruod/r50/rendered_eigen_jet/panels/res4" \
        "${OUT_ROOT}/ruod/vits/rendered_eigen_jet/panels/block12" \
        "${OUT_ROOT}/uiis10k/r50/rendered_eigen_jet/panels/res4" \
        "${OUT_ROOT}/uiis10k/vits/rendered_eigen_jet/panels/block12" \
        -type f -name '*.png' 2>/dev/null | wc -l
)"

echo "JET panels: ${count}"
[[ "${count}" -eq 20 ]] || {
    echo "Expected 20 JET panels, got ${count}" >&2
    exit 1
}
echo "JET rerender complete: ${OUT_ROOT}"
