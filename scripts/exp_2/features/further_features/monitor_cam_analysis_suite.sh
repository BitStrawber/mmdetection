#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
INTERVAL="${INTERVAL:-15}"
LAYERS_COUNT="${LAYERS_COUNT:-4}"

while true; do
    clear || true
    echo "============================================================"
    echo "RUOD CAM analysis suite monitor"
    echo "============================================================"
    date
    echo
    echo "===== processes ====="
    pgrep -af \
        '[r]un_cam_analysis_suite|[e]xtract_fixed_gt_xgradcam|[e]xtract_prediction_xgradcam|[e]xtract_backbone_features|[r]ender_.*xgradcam|[r]ender_feature_activation' \
        || true

    sample_count=0
    [[ -s "${OUT_ROOT}/sample/manifest.jsonl" ]] && \
        sample_count="$(wc -l < "${OUT_ROOT}/sample/manifest.jsonl")"
    echo
    echo "===== shared sample ====="
    echo "images: ${sample_count}"

    echo
    echo "===== fixed-GT CAM ====="
    fixed_meta="$(find "${OUT_ROOT}/fixed_gt/raw/raw_cam" -type f -name instance.json 2>/dev/null | wc -l)"
    fixed_cam="$(find "${OUT_ROOT}/fixed_gt/raw/raw_cam" -type f -name '*.npz' 2>/dev/null | wc -l)"
    fixed_panels="$(find "${OUT_ROOT}/fixed_gt/rendered/panels" -type f -name '*.png' 2>/dev/null | wc -l)"
    echo "instances=${fixed_meta} raw_cam=${fixed_cam} panels=${fixed_panels}"

    echo
    echo "===== prediction-conditioned CAM ====="
    pred_meta="$(find "${OUT_ROOT}/prediction/raw/raw_cam" -type f -name prediction.json 2>/dev/null | wc -l)"
    pred_cam="$(find "${OUT_ROOT}/prediction/raw/raw_cam" -type f -name '*.npz' 2>/dev/null | wc -l)"
    pred_panels="$(find "${OUT_ROOT}/prediction/rendered/panels" -type f -name '*.png' 2>/dev/null | wc -l)"
    echo "predictions=${pred_meta} raw_cam=${pred_cam} panels=${pred_panels}"

    echo
    echo "===== pretrained backbone activation ====="
    spatial="$(find "${OUT_ROOT}/pretrained_backbone/feature_store/spatial" -type f -name '*.npz' 2>/dev/null | wc -l)"
    rendered="$(find "${OUT_ROOT}/pretrained_backbone/rendered" -type f -name '*.png' 2>/dev/null | wc -l)"
    echo "spatial_features=${spatial} rendered_png=${rendered}"

    echo
    echo "===== completion markers ====="
    find "${OUT_ROOT}" -maxdepth 3 \
        \( -name COMPLETE.env -o -name COMPLETE.json \) -print 2>/dev/null || true
    echo
    du -sh "${OUT_ROOT}" 2>/dev/null || true
    echo
    echo "Next refresh in ${INTERVAL}s. Ctrl+C stops monitoring only."
    sleep "${INTERVAL}"
done
