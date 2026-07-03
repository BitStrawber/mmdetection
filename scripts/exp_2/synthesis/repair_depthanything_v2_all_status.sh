#!/usr/bin/env bash
set -euo pipefail

# Full Depth Anything V2 audit + repair workflow.
#
# It performs:
#   1. Check all configured method/split source-depth counts and size alignment.
#   2. Delete problematic depth maps reported by the checker.
#   3. Re-run Depth Anything V2 generation serially over all configured tasks.
#   4. Run a final check and print the remaining problem summary.
#
# Problematic depth maps include:
#   - swapped_hw
#   - other_mismatch
#   - read_error
#
# Missing depth maps are not deleted; the generation step will fill them because
# OVERWRITE defaults to 0 and missing files are processed normally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps}"
DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR:-/home/fcp/xcx/exp_2/syn/Depth-Anything-V2}"
ENCODER="${ENCODER:-vitb}"
CHECKPOINT="${CHECKPOINT:-${DEPTHANYTHING_DIR}/checkpoints/depth_anything_v2_${ENCODER}.pth}"
METHODS="${METHODS:-syreanet uwnr watergan uwdf}"
SPLITS="${SPLITS:-train val}"
GPU_IDS="${GPU_IDS:-2,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
INPUT_SIZE="${INPUT_SIZE:-518}"
WORKERS="${WORKERS:-32}"
CHUNKSIZE="${CHUNKSIZE:-256}"
OVERWRITE="${OVERWRITE:-0}"
INVERT="${INVERT:-0}"
DELETE_STATUSES="${DELETE_STATUSES:-swapped_hw other_mismatch read_error}"
RUN_REGEN="${RUN_REGEN:-1}"
RUN_FINAL_CHECK="${RUN_FINAL_CHECK:-1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/depthanything_v2_repair_all}"

PRECHECK_DIR="${LOG_ROOT}/precheck"
POSTCHECK_DIR="${LOG_ROOT}/postcheck"
DELETE_LOG="${LOG_ROOT}/deleted_problem_depths.tsv"
SERIAL_LOG_DIR="${LOG_ROOT}/serial"

mkdir -p "${LOG_ROOT}" "${PRECHECK_DIR}" "${POSTCHECK_DIR}" "${SERIAL_LOG_DIR}"

echo "========================================="
echo "Depth Anything V2 full audit + repair"
echo "========================================="
echo "SOURCE_ROOT:       ${SOURCE_ROOT}"
echo "DEPTH_ROOT:        ${DEPTH_ROOT}"
echo "DEPTHANYTHING_DIR: ${DEPTHANYTHING_DIR}"
echo "CHECKPOINT:        ${CHECKPOINT}"
echo "ENCODER:           ${ENCODER}"
echo "METHODS:           ${METHODS}"
echo "SPLITS:            ${SPLITS}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "PROCS_PER_GPU:     ${PROCS_PER_GPU}"
echo "INPUT_SIZE:        ${INPUT_SIZE}"
echo "WORKERS:           ${WORKERS}"
echo "CHUNKSIZE:         ${CHUNKSIZE}"
echo "OVERWRITE:         ${OVERWRITE}"
echo "INVERT:            ${INVERT}"
echo "DELETE_STATUSES:   ${DELETE_STATUSES}"
echo "RUN_REGEN:         ${RUN_REGEN}"
echo "RUN_FINAL_CHECK:   ${RUN_FINAL_CHECK}"
echo "LOG_ROOT:          ${LOG_ROOT}"
echo "========================================="

echo
echo "Step 1/4: Pre-check all source/depth pairs"
METHODS="${METHODS}" \
SPLITS="${SPLITS}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
OUT_DIR="${PRECHECK_DIR}" \
WORKERS="${WORKERS}" \
CHUNKSIZE="${CHUNKSIZE}" \
bash scripts/exp_2/synthesis/check_depthanything_v2_generation_status.sh \
  2>&1 | tee "${LOG_ROOT}/precheck.log"

echo
echo "Step 2/4: Delete problematic depth maps from pre-check results"
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
METHODS="${METHODS}" \
SPLITS="${SPLITS}" \
PRECHECK_DIR="${PRECHECK_DIR}" \
DELETE_LOG="${DELETE_LOG}" \
DELETE_STATUSES="${DELETE_STATUSES}" \
python - <<'PY'
import csv
import os
from pathlib import Path

depth_root = Path(os.environ["DEPTH_ROOT"])
methods = os.environ["METHODS"].split()
splits = os.environ["SPLITS"].split()
precheck_dir = Path(os.environ["PRECHECK_DIR"])
delete_log = Path(os.environ["DELETE_LOG"])
delete_statuses = set(os.environ["DELETE_STATUSES"].split())

delete_log.parent.mkdir(parents=True, exist_ok=True)
deleted = 0
already_missing = 0
missing_csv = []
kept = 0

with delete_log.open("w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out, delimiter="\t")
    writer.writerow(["method", "split", "status", "relative", "depth", "action"])

    for method in methods:
        for split in splits:
            csv_path = precheck_dir / f"{method}_{split}_source_depth_size_alignment.csv"
            if not csv_path.exists():
                missing_csv.append(str(csv_path))
                continue

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "")
                    if status not in delete_statuses:
                        kept += 1
                        continue

                    rel = Path(row["relative"])
                    depth_path = (depth_root / method / split / rel).with_suffix(".png")
                    if depth_path.exists():
                        depth_path.unlink()
                        deleted += 1
                        writer.writerow([method, split, status, row["relative"], str(depth_path), "deleted"])
                    else:
                        already_missing += 1
                        writer.writerow([method, split, status, row["relative"], str(depth_path), "already_missing"])

print(f"deleted_problem_depths: {deleted}")
print(f"already_missing_problem_depths: {already_missing}")
print(f"kept_records: {kept}")
print(f"delete_log: {delete_log}")
if missing_csv:
    print("missing alignment csv files:")
    for p in missing_csv:
        print(f"  {p}")
PY

if [[ "${RUN_REGEN}" == "1" ]]; then
  echo
  echo "Step 3/4: Regenerate missing/deleted depth maps serially"
  task_list=""
  for method in ${METHODS}; do
    for split in ${SPLITS}; do
      task_list="${task_list} ${method}:${split}"
    done
  done

  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  TASKS="${task_list}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  OUT_ROOT="${DEPTH_ROOT}" \
  DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR}" \
  ENCODER="${ENCODER}" \
  CHECKPOINT="${CHECKPOINT}" \
  GPU_IDS="${GPU_IDS}" \
  PROCS_PER_GPU="${PROCS_PER_GPU}" \
  INPUT_SIZE="${INPUT_SIZE}" \
  OVERWRITE="${OVERWRITE}" \
  INVERT="${INVERT}" \
  LOG_DIR="${SERIAL_LOG_DIR}" \
  bash scripts/exp_2/synthesis/run_depthanything_v2_remaining_serial_continue.sh \
    2>&1 | tee "${LOG_ROOT}/regenerate.log"
else
  echo
  echo "Step 3/4: Skip regeneration because RUN_REGEN=${RUN_REGEN}"
fi

if [[ "${RUN_FINAL_CHECK}" == "1" ]]; then
  echo
  echo "Step 4/4: Final source/depth check"
  METHODS="${METHODS}" \
  SPLITS="${SPLITS}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  DEPTH_ROOT="${DEPTH_ROOT}" \
  OUT_DIR="${POSTCHECK_DIR}" \
  WORKERS="${WORKERS}" \
  CHUNKSIZE="${CHUNKSIZE}" \
  bash scripts/exp_2/synthesis/check_depthanything_v2_generation_status.sh \
    2>&1 | tee "${LOG_ROOT}/postcheck.log"

  echo
  echo "Final compact summary:"
  SUMMARY_JSON="${POSTCHECK_DIR}/depth_generation_status.json" python - <<'PY'
import json
import os

with open(os.environ["SUMMARY_JSON"], "r", encoding="utf-8") as f:
    data = json.load(f)

for r in data["records"]:
    print(
        f"{r['method']}/{r['split']}: "
        f"source={r['source_count']}, depth={r['depth_count']}, "
        f"missing={r['missing_depth']}, swapped_hw={r['swapped_hw']}, "
        f"other_mismatch={r['other_mismatch']}, read_error={r['read_error']}, "
        f"problem_total={r['problem_total']}"
    )
print("totals:", data["totals"])
PY
else
  echo
  echo "Step 4/4: Skip final check because RUN_FINAL_CHECK=${RUN_FINAL_CHECK}"
fi

echo
echo "========================================="
echo "Depth Anything V2 full repair done"
echo "========================================="
echo "precheck:  ${PRECHECK_DIR}"
echo "delete log: ${DELETE_LOG}"
echo "serial:    ${SERIAL_LOG_DIR}"
echo "postcheck: ${POSTCHECK_DIR}"
echo "========================================="
