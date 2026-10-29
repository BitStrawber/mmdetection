#!/usr/bin/env bash
set -euo pipefail

CAM_BASE="${1:-${CAM_BASE:-}}"
[[ -n "${CAM_BASE}" ]] || {
    echo "Usage: $0 CAM_BASE" >&2
    exit 2
}

date
echo "CAM_BASE=${CAM_BASE}"
echo

for name in ruod_r50 ruod_vits uiis10k_r50 uiis10k_vits; do
    out="${CAM_BASE}/output/${name}"
    log="${CAM_BASE}/logs/${name}.runner.log"
    render_log="${out}/logs/render.log"
    pid_file="${CAM_BASE}/logs/${name}.pid"
    pid=""
    if [[ -s "${pid_file}" ]]; then
        pid="$(cat "${pid_file}")"
    fi

    state="WAITING"
    if [[ -s "${out}/COMPLETE.env" ]]; then
        state="COMPLETE"
    elif [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        state="RUNNING"
    elif [[ -s "${log}" ]]; then
        state="STOPPED_OR_FAILED"
    fi

    meta=0
    unions=0
    panels=0
    if [[ -d "${out}/prediction_cam/raw_cam" ]]; then
        meta="$(
            find "${out}/prediction_cam/raw_cam" \
                -name prediction.json -type f | wc -l
        )"
    fi
    if [[ -d "${out}/rendered" ]]; then
        unions="$(
            find "${out}/rendered" \
                -path '*/image_union/*.png' -type f | wc -l
        )"
        panels="$(
            find "${out}/rendered" \
                -path '*/panels_5x4/*.png' -type f | wc -l
        )"
    fi
    printf '%-16s %-18s predictions=%-6s unions=%-5s panels=%s\n' \
        "${name}" "${state}" "${meta}" "${unions}" "${panels}"

    if [[ -s "${render_log}" ]]; then
        printf '  render: %s\n' "$(tail -n 1 "${render_log}")"
    else
        echo "  render: no render.log yet"
    fi
done

echo
echo "===== latest logs ====="
for log in "${CAM_BASE}"/logs/*.runner.log; do
    [[ -s "${log}" ]] || continue
    echo "--- $(basename "${log}") ---"
    tail -n 5 "${log}"
done

echo
echo "===== GPUs 4-7 ====="
nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader |
awk -F', ' '$1 >= 4 && $1 <= 7'
