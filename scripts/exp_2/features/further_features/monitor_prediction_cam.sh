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
    panels=0
    if [[ -d "${out}/raw" ]]; then
        meta="$(find "${out}/raw" -name prediction.json -type f | wc -l)"
    fi
    if [[ -d "${out}/rendered" ]]; then
        panels="$(find "${out}/rendered" -name '*.png' -type f | wc -l)"
    fi
    printf '%-16s %-18s predictions=%-6s panels=%s\n' \
        "${name}" "${state}" "${meta}" "${panels}"
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
