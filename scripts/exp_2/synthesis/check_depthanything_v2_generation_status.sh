#!/usr/bin/env bash
set -euo pipefail

# Check Depth Anything V2 generation status for synthetic ImageNet splits.
#
# It reports:
#   1. source image count
#   2. generated depth PNG count
#   3. missing count by simple count difference
#   4. source/depth size alignment using tools/check_source_depth_size_alignment.py
#
# Usage:
#   bash scripts/exp_2/synthesis/check_depthanything_v2_generation_status.sh
#
# Common overrides:
#   METHODS="syreanet uwnr watergan uwdf" SPLITS="train val" WORKERS=16 \
#     bash scripts/exp_2/synthesis/check_depthanything_v2_generation_status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
METHODS="${METHODS:-syreanet uwnr watergan uwdf}"
SPLITS="${SPLITS:-train val}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/logs/depthanything_v2_status}"
WORKERS="${WORKERS:-16}"
CHUNKSIZE="${CHUNKSIZE:-128}"
LIMIT="${LIMIT:-0}"
RUN_ALIGNMENT="${RUN_ALIGNMENT:-1}"

mkdir -p "${OUT_DIR}"

SUMMARY_TSV="${OUT_DIR}/depth_generation_status.tsv"
SUMMARY_JSON="${OUT_DIR}/depth_generation_status.json"
: > "${SUMMARY_TSV}"
printf "method\tsplit\tsource_count\tdepth_count\tcount_missing\talignment_status\tmatch\tproblem_total\tmissing_depth\tswapped_hw\tother_mismatch\tread_error\tjson\n" >> "${SUMMARY_TSV}"

echo "========================================="
echo "Depth Anything V2 generation status check"
echo "========================================="
echo "SOURCE_ROOT:   ${SOURCE_ROOT}"
echo "DEPTH_ROOT:    ${DEPTH_ROOT}"
echo "METHODS:       ${METHODS}"
echo "SPLITS:        ${SPLITS}"
echo "OUT_DIR:       ${OUT_DIR}"
echo "WORKERS:       ${WORKERS}"
echo "CHUNKSIZE:     ${CHUNKSIZE}"
echo "LIMIT:         ${LIMIT}"
echo "RUN_ALIGNMENT: ${RUN_ALIGNMENT}"
echo "========================================="

count_source_images() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" \
    -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.JPEG' -o -iname '*.bmp' -o -iname '*.webp' \) \
    2>/dev/null | wc -l
}

count_depth_images() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" -type f -name '*.png' 2>/dev/null | wc -l
}

json_files=()

for method in ${METHODS}; do
  for split in ${SPLITS}; do
    source_dir="${SOURCE_ROOT}/${method}/source/${split}"
    depth_dir="${DEPTH_ROOT}/${method}/${split}"
    prefix="${OUT_DIR}/${method}_${split}_source_depth_size_alignment"

    echo
    echo "-----------------------------------------"
    echo "${method}/${split}"
    echo "source_dir: ${source_dir}"
    echo "depth_dir:  ${depth_dir}"
    echo "-----------------------------------------"

    source_count="$(count_source_images "${source_dir}")"
    depth_count="$(count_depth_images "${depth_dir}")"
    count_missing=$((source_count - depth_count))
    echo "source_count: ${source_count}"
    echo "depth_count:  ${depth_count}"
    echo "missing:      ${count_missing}"

    alignment_status="skipped"
    match=0
    problem_total=0
    missing_depth=0
    swapped_hw=0
    other_mismatch=0
    read_error=0
    json_path="${prefix}.json"

    if [[ "${RUN_ALIGNMENT}" == "1" ]]; then
      if [[ "${source_count}" == "0" ]]; then
        alignment_status="no_source"
        echo "Skip alignment: source count is 0"
      elif [[ "${depth_count}" == "0" ]]; then
        alignment_status="no_depth"
        echo "Skip alignment: depth count is 0"
      else
        alignment_status="running"
        limit_args=()
        if [[ "${LIMIT}" != "0" ]]; then
          limit_args+=(--limit "${LIMIT}")
        fi
        python tools/check_source_depth_size_alignment.py \
          --source-root "${source_dir}" \
          --depth-root "${depth_dir}" \
          --out-prefix "${prefix}" \
          --workers "${WORKERS}" \
          --chunksize "${CHUNKSIZE}" \
          "${limit_args[@]}" \
          2>&1 | tee "${prefix}.log"
        alignment_status="done"
        json_files+=("${json_path}")

        read -r match problem_total missing_depth swapped_hw other_mismatch read_error < <(
          JSON_PATH="${json_path}" python - <<'PY'
import json
import os

path = os.environ["JSON_PATH"]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(
    data.get("match", 0),
    data.get("problem_total", 0),
    data.get("missing_depth", 0),
    data.get("swapped_hw", 0),
    data.get("other_mismatch", 0),
    data.get("read_error", 0),
)
PY
        )
      fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${method}" "${split}" "${source_count}" "${depth_count}" "${count_missing}" \
      "${alignment_status}" "${match}" "${problem_total}" "${missing_depth}" \
      "${swapped_hw}" "${other_mismatch}" "${read_error}" "${json_path}" \
      >> "${SUMMARY_TSV}"
  done
done

SUMMARY_TSV="${SUMMARY_TSV}" SUMMARY_JSON="${SUMMARY_JSON}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

summary_tsv = Path(os.environ["SUMMARY_TSV"])
summary_json = Path(os.environ["SUMMARY_JSON"])

records = []
with summary_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        for key in [
            "source_count",
            "depth_count",
            "count_missing",
            "match",
            "problem_total",
            "missing_depth",
            "swapped_hw",
            "other_mismatch",
            "read_error",
        ]:
            row[key] = int(row[key])
        records.append(row)

totals = {
    "source_count": sum(r["source_count"] for r in records),
    "depth_count": sum(r["depth_count"] for r in records),
    "count_missing": sum(r["count_missing"] for r in records),
    "problem_total": sum(r["problem_total"] for r in records),
    "missing_depth": sum(r["missing_depth"] for r in records),
    "swapped_hw": sum(r["swapped_hw"] for r in records),
    "other_mismatch": sum(r["other_mismatch"] for r in records),
    "read_error": sum(r["read_error"] for r in records),
}

payload = {
    "summary_tsv": str(summary_tsv),
    "records": records,
    "totals": totals,
}
summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo
echo "========================================="
echo "Depth status check done"
echo "========================================="
echo "summary tsv:  ${SUMMARY_TSV}"
echo "summary json: ${SUMMARY_JSON}"
echo "details dir:  ${OUT_DIR}"
echo "========================================="
