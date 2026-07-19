#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BASE_SHARD_ROOT="${BASE_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_resume_shards/train}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_parallel/train}"
RESUME_SHARD_ROOT="${RESUME_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_retry_shards/train}"
RESUME_RESULTS_ROOT="${RESUME_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_resume/train}"
RETRY_RESULTS_ROOT="${RETRY_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_retry/train}"
VAL_SHARD_ROOT="${VAL_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_shards/val}"
VAL_RESULTS_ROOT="${VAL_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_parallel/val}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated}"
WORK_ROOT="${WORK_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/.layered_restore_work}"
PUBLISH_ROOT="${PUBLISH_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/.layered_restore_publish}"
RESTORE_WORKERS="${RESTORE_WORKERS:-8}"
RESET_WORK="${RESET_WORK:-1}"
REPLACE_FINAL="${REPLACE_FINAL:-0}"
CLEAN_WORK="${CLEAN_WORK:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

RESTORE_TOOL="${REPO_ROOT}/tools/restore_watergan_fake.py"
JOBS_FILE="${WORK_ROOT}/restore_jobs.tsv"

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

count_fake() {
  find "$1" -maxdepth 1 -type f -name 'fake_*.png' 2>/dev/null \
    | wc -l | tr -d ' '
}

manifest_count() {
  wc -l < "$1" | tr -d ' '
}

if [[ "${RESET_WORK}" == 1 ]]; then
  rm -rf "${WORK_ROOT}" "${PUBLISH_ROOT}"
fi
[[ ! -e "${WORK_ROOT}" && ! -e "${PUBLISH_ROOT}" ]] || {
  echo "Error: restore workspace already exists; use RESET_WORK=1 after checking it." >&2
  exit 1
}
mkdir -p "${WORK_ROOT}/train" "${WORK_ROOT}/val" "${PUBLISH_ROOT}"
: > "${JOBS_FILE}"

add_job() {
  local label="$1" manifest="$2" results="$3" out="$4"
  local records outputs
  [[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 1; }
  [[ -d "${results}" ]] || { echo "Missing results: ${results}" >&2; exit 1; }
  records="$(manifest_count "${manifest}")"
  outputs="$(count_fake "${results}")"
  [[ "${outputs}" -ge "${records}" ]] || {
    echo "Incomplete ${label}: records=${records}, fake=${outputs}" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\n' "${label}" "${manifest}" "${results}" "${out}" >> "${JOBS_FILE}"
  echo "${label}: records=${records}, fake=${outputs}"
}

shopt -s nullglob

train_total=0
for manifest in "${BASE_SHARD_ROOT}/completed_manifests"/shard*of4.jsonl; do
  name="$(basename "${manifest}" .jsonl)"
  count="$(manifest_count "${manifest}")"; train_total=$((train_total + count))
  add_job "train_base_${name}" "${manifest}" "${BASE_RESULTS_ROOT}/${name}" \
    "${WORK_ROOT}/train/base_${name}"
done

for manifest in "${RESUME_SHARD_ROOT}/completed_manifests"/shard*of12.jsonl; do
  name="$(basename "${manifest}" .jsonl)"
  count="$(manifest_count "${manifest}")"; train_total=$((train_total + count))
  add_job "train_resume_${name}" "${manifest}" "${RESUME_RESULTS_ROOT}/${name}" \
    "${WORK_ROOT}/train/resume_${name}"
done

for shard in "${RESUME_SHARD_ROOT}"/shard*of2; do
  name="$(basename "${shard}")"
  manifest="${shard}/watergan_air_manifest.jsonl"
  count="$(manifest_count "${manifest}")"; train_total=$((train_total + count))
  add_job "train_retry_${name}" "${manifest}" "${RETRY_RESULTS_ROOT}/${name}" \
    "${WORK_ROOT}/train/retry_${name}"
done

[[ "${train_total}" -eq 250000 ]] || {
  echo "Error: layered train manifests total=${train_total}, expected=250000" >&2
  exit 1
}

val_total=0
for shard in "${VAL_SHARD_ROOT}"/shard*of*; do
  [[ -d "${shard}" ]] || continue
  name="$(basename "${shard}")"
  manifest="${shard}/watergan_air_manifest.jsonl"
  count="$(manifest_count "${manifest}")"; val_total=$((val_total + count))
  add_job "val_${name}" "${manifest}" "${VAL_RESULTS_ROOT}/${name}" \
    "${WORK_ROOT}/val/${name}"
done

[[ "${val_total}" -eq 10000 ]] || {
  echo "Error: val manifests total=${val_total}, expected=10000" >&2
  exit 1
}

echo "Restore jobs: $(wc -l < "${JOBS_FILE}")"
export RESTORE_TOOL BATCH_SIZE
xargs -P "${RESTORE_WORKERS}" -n 4 bash -c '
  label="$1"; manifest="$2"; results="$3"; out="$4"
  echo "START ${label}"
  python "${RESTORE_TOOL}" \
    --manifest "${manifest}" --results-dir "${results}" --out-dir "${out}" \
    --batch-size "${BATCH_SIZE}" --overwrite > "${out}.log" 2>&1
  echo "DONE ${label}"
' _ < "${JOBS_FILE}"

python - "${WORK_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = sorted(root.rglob('restore_watergan_fake_summary.json'))
if not summaries:
    raise SystemExit('No restore summaries found')
bad = []
for path in summaries:
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('missing') or data.get('bad_names') or data.get('duplicate_indices'):
        bad.append((str(path), data))
print('restore summaries: {}'.format(len(summaries)))
if bad:
    for path, data in bad:
        print('BAD {}: {}'.format(path, data))
    raise SystemExit(1)
PY

merge_split() {
  local split="$1" expected="$2" source_root="${WORK_ROOT}/$1"
  local publish="${PUBLISH_ROOT}/${split}"
  mkdir -p "${publish}"
  while IFS= read -r -d '' class_dir; do
    class_name="$(basename "${class_dir}")"
    mkdir -p "${publish}/${class_name}"
    cp -al "${class_dir}/." "${publish}/${class_name}/"
  done < <(find "${source_root}" -type d -name 'n[0-9]*' -print0)

  local images classes existing
  images="$(count_images "${publish}")"
  classes="$(find "${publish}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  [[ "${images}" -eq "${expected}" && "${classes}" -eq 1000 ]] || {
    echo "Error: staged ${split}: images=${images}/${expected}, classes=${classes}/1000" >&2
    exit 1
  }

  existing="$(count_images "${FINAL_ROOT}/${split}")"
  if [[ "${existing}" -eq "${expected}" ]]; then
    echo "Reuse complete final ${split}: ${FINAL_ROOT}/${split}"
    rm -rf "${publish}"
  else
    [[ "${existing}" -eq 0 || "${REPLACE_FINAL}" == 1 ]] || {
      echo "Error: incomplete final ${split} exists (${existing}); set REPLACE_FINAL=1." >&2
      exit 1
    }
    rm -rf "${FINAL_ROOT}/${split}"
    mkdir -p "${FINAL_ROOT}"
    mv "${publish}" "${FINAL_ROOT}/${split}"
    echo "Published ${split}: ${FINAL_ROOT}/${split}"
  fi
}

merge_split train 250000
merge_split val 10000

echo "train: $(count_images "${FINAL_ROOT}/train")"
echo "val:   $(count_images "${FINAL_ROOT}/val")"

if [[ "${CLEAN_WORK}" == 1 ]]; then
  rm -rf "${WORK_ROOT}" "${PUBLISH_ROOT}"
fi

echo "WaterGAN layered restore complete."
