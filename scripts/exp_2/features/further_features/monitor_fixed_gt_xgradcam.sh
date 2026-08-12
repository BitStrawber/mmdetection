#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the run directory}"
INTERVAL="${INTERVAL:-15}"
CAM_ROOT="${CAM_ROOT:-${OUT_ROOT}/fixed_gt_cam}"
RENDER_ROOT="${RENDER_ROOT:-${OUT_ROOT}/rendered}"

while true; do
    clear || true
    echo "============================================================"
    echo "Fixed-GT XGradCAM monitor"
    echo "============================================================"
    date
    echo
    echo "===== active processes ====="
    pgrep -af \
        '[e]xtract_fixed_gt_xgradcam|[r]ender_fixed_gt_xgradcam|[p]lot_fixed_gt_cam_metrics|[r]un_fixed_gt_xgradcam_analysis' \
        || true
    echo
    echo "===== raw CAM ====="
    metadata=$(find "${CAM_ROOT}/raw_cam" -type f -name instance.json 2>/dev/null | wc -l)
    arrays=$(find "${CAM_ROOT}/raw_cam" -type f -name '*.npz' 2>/dev/null | wc -l)
    printf 'model-instance metadata: %d\n' "${metadata}"
    printf 'raw layer CAM arrays:    %d\n' "${arrays}"
    find "${CAM_ROOT}/raw_cam" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r root; do
        model="$(basename "${root}")"
        instances=$(find "${root}" -type f -name instance.json | wc -l)
        layers=$(find "${root}" -type f -name '*.npz' | wc -l)
        printf '%-42s instances=%5d arrays=%6d\n' "${model}" "${instances}" "${layers}"
    done
    echo
    echo "===== rendered output ====="
    if [[ -d "${RENDER_ROOT}" ]]; then
        find "${RENDER_ROOT}" -mindepth 1 -maxdepth 1 -type d | while read -r root; do
            files=$(find "${root}" -type f -name '*.png' | wc -l)
            printf '%-42s PNG=%7d\n' "$(basename "${root}")" "${files}"
        done
    else
        echo "NOT STARTED"
    fi
    echo
    echo "===== latest extraction lines ====="
    find "${OUT_ROOT}/logs" -maxdepth 1 -type f -name '02_extract_*.log' -print0 2>/dev/null |
        xargs -0 -r tail -n 1 || true
    echo
    echo "===== errors ====="
    grep -RInaE \
        'Traceback|FAILED|Error:|RuntimeError|ValueError|CUDA out of memory' \
        "${OUT_ROOT}/logs" 2>/dev/null | tail -n 20 || true
    echo
    if [[ -s "${OUT_ROOT}/COMPLETE.env" ]]; then
        echo "PIPELINE COMPLETE"
        cat "${OUT_ROOT}/COMPLETE.env"
        exit 0
    fi
    echo "Next refresh in ${INTERVAL}s. Ctrl+C stops monitoring only."
    sleep "${INTERVAL}"
done
