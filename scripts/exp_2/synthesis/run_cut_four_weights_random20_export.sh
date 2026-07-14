#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/exp_2/synthesis/lib/experiment_common.sh"

CUT_DIR="${CUT_DIR:-/home/fcp/xcx/exp_2/syn/CUT}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/cut/source/train}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/cut_four_weights_random20}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export.tar.gz}"
RCLONE_DEST="${RCLONE_DEST:-gd:exp_2/cut_four_weights_random20_export}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU="${GPU:-2}"
LOAD_SIZE="${LOAD_SIZE:-256}"
CROP_SIZE="${CROP_SIZE:-256}"
NUM_THREADS="${NUM_THREADS:-4}"
UPLOAD="${UPLOAD:-1}"
PACKAGE_EXPORT="${PACKAGE_EXPORT:-1}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"

CHECKPOINT_SEARCH_ROOTS="${CHECKPOINT_SEARCH_ROOTS:-/media/SSD1/XCX/exp_2/synthesis_work /media/HDD1/XCX/exp_2 /home/fcp/xcx/exp_2}"
MODEL_NAMES="${MODEL_NAMES:-imagenet_ruod_cut_full_bs2_1epoch_gpu2 imagenet_ruod_cut_full_bs2_2epoch_gpu3 imagenet_ruod_cut_full_bs2_3epoch_gpu4 imagenet_ruod_cut_full_bs2_5epoch_gpu5}"

LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
SELECT_DIR="${WORK_ROOT}/selected"
TEST_ROOT="${WORK_ROOT}/cut_test_dataroot"
STATUS_FILE="${WORK_ROOT}/status.tsv"

find_checkpoint_dir() {
  local model_name="$1"
  local root
  local found

  for root in ${CHECKPOINT_SEARCH_ROOTS}; do
    if [ ! -d "${root}" ]; then
      continue
    fi
    found="$(find "${root}" -type d -name "${model_name}" 2>/dev/null | head -n 1 || true)"
    if [ -n "${found}" ] && [ -f "${found}/latest_net_G.pth" ]; then
      echo "${found}"
      return 0
    fi
  done

  return 1
}

select_sources() {
  python - "$SOURCE_ROOT" "$SELECT_DIR" "$NUM" "$SEED" <<'PY'
import json
import random
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
select_dir = Path(sys.argv[2])
num = int(sys.argv[3])
seed = int(sys.argv[4])
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

images = sorted(
    p for p in source_root.rglob('*')
    if p.is_file() and p.suffix.lower() in image_exts
)
if len(images) < num:
    raise SystemExit(f'not enough source images: need {num}, found {len(images)}')

random.Random(seed).shuffle(images)
selected = images[:num]

input_dir = select_dir / 'testA'
source_export = select_dir / 'source'
input_dir.mkdir(parents=True, exist_ok=True)
source_export.mkdir(parents=True, exist_ok=True)

records = []
used_names = set()
for index, src in enumerate(selected):
    rel = src.relative_to(source_root)
    stem = f'{index:03d}_{rel.parent.as_posix().replace("/", "_")}_{src.stem}'
    stem = ''.join(c if c.isalnum() or c in '._-' else '_' for c in stem).strip('_')
    if not stem:
        stem = f'{index:03d}'
    name = f'{stem}{src.suffix.lower()}'
    while name in used_names:
        stem = f'{stem}_dup'
        name = f'{stem}{src.suffix.lower()}'
    used_names.add(name)

    dst_input = input_dir / name
    dst_source = source_export / name
    shutil.copy2(src, dst_input)
    shutil.copy2(src, dst_source)
    records.append({
        'index': index,
        'source': str(src),
        'relative_source': rel.as_posix(),
        'test_name': name,
        'testA': str(dst_input),
    })

manifest = {
    'source_root': str(source_root),
    'num': num,
    'seed': seed,
    'records': records,
}
(select_dir / 'selection_manifest.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding='utf-8',
)
for record in records:
    print(f"{record['index']:03d}\t{record['relative_source']}\t{record['test_name']}")
PY
}

copy_cut_results() {
  local model_name="$1"
  local result_image_dir="$2"
  local export_dir="$3"

  mkdir -p "${export_dir}"

  if [ ! -d "${result_image_dir}" ]; then
    exp_die "CUT result image dir not found: ${result_image_dir}" || return 1
  fi

  find "${result_image_dir}" -maxdepth 1 -type f \( -name '*_fake_B.png' -o -name '*_fake.png' -o -name '*.png' -o -name '*.jpg' \) \
    | sort \
    | while read -r image_path; do
        local base
        base="$(basename "${image_path}")"
        case "${base}" in
          *_fake_B.png) base="${base%_fake_B.png}.png" ;;
          *_fake.png) base="${base%_fake.png}.png" ;;
        esac
        cp -f "${image_path}" "${export_dir}/${base}"
      done

  local count
  count="$(find "${export_dir}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)"
  if [ "${count}" -lt "${NUM}" ]; then
    exp_warn "${model_name}: expected ${NUM} generated images, found ${count}"
  fi
}

run_model() {
  local model_name="$1"
  local checkpoint_dir="$2"
  local checkpoint_parent
  local log_path
  local result_dir
  local result_image_dir
  local export_dir

  checkpoint_parent="$(dirname "${checkpoint_dir}")"
  log_path="${LOG_DIR}/${model_name}.log"
  result_dir="${WORK_ROOT}/cut_results/${model_name}"
  result_image_dir="${result_dir}/test_latest/images"
  export_dir="${OUT_ROOT}/generated/${model_name}"

  rm -rf "${result_dir}" "${export_dir}"
  mkdir -p "$(dirname "${result_dir}")" "${export_dir}" "${LOG_DIR}"

  echo "Run CUT model: ${model_name}"
  echo "  checkpoint_dir: ${checkpoint_dir}"
  echo "  log:            ${log_path}"

  (
    cd "${CUT_DIR}" || exit 1
    CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
      --dataroot "${TEST_ROOT}" \
      --name "${model_name}" \
      --checkpoints_dir "${checkpoint_parent}" \
      --results_dir "${WORK_ROOT}/cut_results" \
      --phase test \
      --num_test "${NUM}" \
      --batch_size 1 \
      --num_threads "${NUM_THREADS}" \
      --serial_batches \
      --no_flip \
      --load_size "${LOAD_SIZE}" \
      --crop_size "${CROP_SIZE}" \
      --preprocess resize_and_crop \
      --model cut \
      --no_dropout
  ) 2>&1 | tee "${log_path}"

  local rc="${PIPESTATUS[0]}"
  if [ "${rc}" -ne 0 ]; then
    exp_record_status "${STATUS_FILE}" "${model_name}" "FAILED" "${GPU}" "${log_path}"
    exp_warn "${model_name}: CUT inference failed with exit code ${rc}"
    return 1
  fi

  copy_cut_results "${model_name}" "${result_image_dir}" "${export_dir}" || {
    exp_record_status "${STATUS_FILE}" "${model_name}" "FAILED_COPY" "${GPU}" "${log_path}"
    return 1
  }

  exp_record_status "${STATUS_FILE}" "${model_name}" "OK" "${GPU}" "${log_path}"
  return 0
}

main() {
  exp_section "CUT four-checkpoint random20 export"
  echo "CUT_DIR:                 ${CUT_DIR}"
  echo "SOURCE_ROOT:             ${SOURCE_ROOT}"
  echo "WORK_ROOT:               ${WORK_ROOT}"
  echo "OUT_ROOT:                ${OUT_ROOT}"
  echo "ARCHIVE_PATH:            ${ARCHIVE_PATH}"
  echo "RCLONE_DEST:             ${RCLONE_DEST}"
  echo "NUM:                     ${NUM}"
  echo "SEED:                    ${SEED}"
  echo "GPU:                     ${GPU}"
  echo "MODEL_NAMES:             ${MODEL_NAMES}"
  echo "CHECKPOINT_SEARCH_ROOTS: ${CHECKPOINT_SEARCH_ROOTS}"
  echo "UPLOAD:                  ${UPLOAD}"
  echo "PACKAGE_EXPORT:          ${PACKAGE_EXPORT}"
  echo "RESET_OUTPUTS:           ${RESET_OUTPUTS}"

  exp_require_paths "${CUT_DIR}" "${SOURCE_ROOT}" || exit 1

  if [ "${RESET_OUTPUTS}" = "1" ]; then
    rm -rf "${WORK_ROOT}" "${OUT_ROOT}"
  fi
  mkdir -p "${WORK_ROOT}" "${OUT_ROOT}" "${LOG_DIR}"
  : > "${STATUS_FILE}"

  exp_section "Select source images"
  mkdir -p "${SELECT_DIR}"
  select_sources | tee "${OUT_ROOT}/selected_sources.tsv"
  cp -a "${SELECT_DIR}/source" "${OUT_ROOT}/source"
  cp -f "${SELECT_DIR}/selection_manifest.json" "${OUT_ROOT}/selection_manifest.json"
  rm -rf "${TEST_ROOT}"
  mkdir -p "${TEST_ROOT}"
  cp -a "${SELECT_DIR}/testA" "${TEST_ROOT}/testA"

  exp_section "Run CUT inference"
  local model_name
  local checkpoint_dir
  local failed=0
  for model_name in ${MODEL_NAMES}; do
    checkpoint_dir="$(find_checkpoint_dir "${model_name}" || true)"
    if [ -z "${checkpoint_dir}" ]; then
      exp_warn "checkpoint not found for ${model_name}"
      exp_record_status "${STATUS_FILE}" "${model_name}" "MISSING_CHECKPOINT" "${GPU}" ""
      failed=1
      continue
    fi
    run_model "${model_name}" "${checkpoint_dir}" || failed=1
  done

  cp -f "${STATUS_FILE}" "${OUT_ROOT}/status.tsv"
  cp -a "${LOG_DIR}" "${OUT_ROOT}/logs"

  exp_section "Package and upload"
  exp_package_and_upload "${OUT_ROOT}" "${ARCHIVE_PATH}" "${UPLOAD}" "${PACKAGE_EXPORT}" "${RCLONE_DEST}" || failed=1

  exp_section "Done"
  if [ "${failed}" -ne 0 ]; then
    echo "Finished with some failed or missing models. Check ${OUT_ROOT}/status.tsv"
    exit 1
  fi
  echo "All CUT random20 exports completed."
}

main "$@"
