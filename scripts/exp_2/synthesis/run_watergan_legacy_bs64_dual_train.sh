#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-/media/HDD2/XCX/exp_2/watergan_legacy_bs64_dual_train}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_legacy_bs64_dual_train}"

GPU_10E="${GPU_10E:-0}"
GPU_5E="${GPU_5E:-1}"
EPOCHS_10E="${EPOCHS_10E:-10}"
EPOCHS_5E="${EPOCHS_5E:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
SAVE_EPOCH="${SAVE_EPOCH:-1}"
MAX_TO_KEEP="${MAX_TO_KEEP:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
BETA1="${BETA1:-0.5}"
RESET_RUNS="${RESET_RUNS:-0}"

INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
INPUT_WATER_HEIGHT="${INPUT_WATER_HEIGHT:-1024}"
INPUT_WATER_WIDTH="${INPUT_WATER_WIDTH:-1360}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"

count_entries() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null |
    wc -l |
    tr -d ' '
}

require_path() {
  [[ -e "$1" ]] || {
    echo "Error: required path not found: $1" >&2
    exit 1
  }
}

safe_reset() {
  local path="$1"
  local allowed_prefix="$2"
  local resolved

  resolved="$(readlink -m "${path}")"
  case "${resolved}" in
    "${allowed_prefix}"*) rm -rf -- "${resolved}" ;;
    *)
      echo "Error: refusing to reset unexpected path: ${resolved}" >&2
      exit 1
      ;;
  esac
}

require_path "${WATERGAN_DIR}/mainmhl.py"
for name in air_images air_depth water_images; do
  require_path "${DATA_ROOT}/${name}"
  count="$(count_entries "${DATA_ROOT}/${name}")"
  [[ "${count}" -eq "${TRAIN_SIZE}" ]] || {
    echo "Error: ${name} has ${count} entries; expected ${TRAIN_SIZE}" >&2
    exit 1
  }
done

[[ "${GPU_10E}" != "${GPU_5E}" ]] || {
  echo "Error: GPU_10E and GPU_5E must be different" >&2
  exit 1
}
[[ "${BATCH_SIZE}" -eq 64 ]] || {
  echo "Error: this official-style launcher requires BATCH_SIZE=64" >&2
  exit 1
}

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" "${WATERGAN_DIR}/data"

WATERGAN_DIR="${WATERGAN_DIR}" \
  bash scripts/exp_2/synthesis/patch_watergan_depth_safety.sh

# Keep the initial model-2 checkpoint and every epoch checkpoint. Legacy
# copies may contain the original Saver(), a fixed max_to_keep value, or the
# environment-controlled compatibility patch.
python - "${WATERGAN_DIR}/modelmhl.py" "${WATERGAN_DIR}/modeljamaica.py" "${MAX_TO_KEEP}" <<'PY'
from __future__ import print_function

import io
import re
import sys

replacement = (
    "tf.train.Saver(max_to_keep=int("
    "os.environ.get('WATERGAN_MAX_TO_KEEP', '{}')))"
).format(sys.argv[-1])
fixed_saver = re.compile(r"tf\.train\.Saver\(max_to_keep=\d+\)")

for filename in sys.argv[1:-1]:
    with io.open(filename, "r", encoding="utf-8") as handle:
        text = handle.read()

    if "WATERGAN_MAX_TO_KEEP" in text:
        print("Saver retention already uses WATERGAN_MAX_TO_KEEP: {}".format(filename))
        continue

    if "tf.train.Saver()" in text:
        updated = text.replace("tf.train.Saver()", replacement)
    else:
        updated, count = fixed_saver.subn(replacement, text)
        if count != 1:
            raise RuntimeError(
                "expected one WaterGAN saver in {}, found {}".format(
                    filename, count
                )
            )

    with io.open(filename, "w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    print("Saver retention patched: {}".format(filename))
PY

python -m py_compile \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py"

ln -sfn "${DATA_ROOT}/air_images" \
  "${WATERGAN_DIR}/data/${DATA_NAME}_air_images"
ln -sfn "${DATA_ROOT}/air_depth" \
  "${WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" \
  "${WATERGAN_DIR}/data/${DATA_NAME}_water_images"

cat <<EOF
============================================================
WaterGAN legacy official-style dual training
============================================================
WATERGAN_DIR:  ${WATERGAN_DIR}
DATA_ROOT:     ${DATA_ROOT}
TRAIN_SIZE:   ${TRAIN_SIZE}
BATCH_SIZE:   ${BATCH_SIZE}
10e run:      GPU ${GPU_10E}, ${EPOCHS_10E} epochs
5e run:       GPU ${GPU_5E}, ${EPOCHS_5E} epochs
SAVE_EPOCH:   ${SAVE_EPOCH}
MAX_TO_KEEP:  ${MAX_TO_KEEP}
RESULT_ROOT:  ${RESULT_ROOT}
LOG_ROOT:     ${LOG_ROOT}
RESET_RUNS:   ${RESET_RUNS}
============================================================
EOF

run_experiment() {
  local label="$1"
  local gpu="$2"
  local epochs="$3"
  local checkpoint_dir="checkpoint_legacy_official_bs64_${label}"
  local sample_dir="samples_legacy_official_bs64_${label}"
  local checkpoint_root="${WATERGAN_DIR}/${checkpoint_dir}"
  local sample_root="${WATERGAN_DIR}/${sample_dir}"
  local result_dir="${RESULT_ROOT}/${label}"
  local log_dir="${LOG_ROOT}/${label}"
  local log_file="${log_dir}/train.log"

  if [[ -e "${checkpoint_root}" || -e "${sample_root}" || \
        -e "${result_dir}" || -e "${log_dir}" ]]; then
    if [[ "${RESET_RUNS}" != 1 ]]; then
      echo "Error: ${label} output already exists; set RESET_RUNS=1 to restart" >&2
      return 1
    fi
    [[ -e "${checkpoint_root}" ]] && safe_reset \
      "${checkpoint_root}" "${WATERGAN_DIR}/checkpoint_legacy_official_bs64_"
    [[ -e "${sample_root}" ]] && safe_reset \
      "${sample_root}" "${WATERGAN_DIR}/samples_legacy_official_bs64_"
    [[ -e "${result_dir}" ]] && safe_reset \
      "${result_dir}" "${RESULT_ROOT}/"
    [[ -e "${log_dir}" ]] && safe_reset \
      "${log_dir}" "${LOG_ROOT}/"
  fi

  mkdir -p "${result_dir}" "${log_dir}"

  cat > "${log_dir}/configuration.txt" <<EOF
label=${label}
gpu=${gpu}
epochs=${epochs}
batch_size=${BATCH_SIZE}
train_size=${TRAIN_SIZE}
save_epoch=${SAVE_EPOCH}
max_to_keep=${MAX_TO_KEEP}
checkpoint_dir=${checkpoint_dir}
sample_dir=${sample_dir}
result_dir=${result_dir}
EOF

  echo "START ${label}: GPU=${gpu}, epochs=${epochs}"
  (
    cd "${WATERGAN_DIR}"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      TF_FORCE_GPU_ALLOW_GROWTH=true \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      WATERGAN_MAX_TO_KEEP="${MAX_TO_KEEP}" \
      python mainmhl.py \
        --water_dataset "${DATA_NAME}_water_images" \
        --air_dataset "${DATA_NAME}_air_images" \
        --depth_dataset "${DATA_NAME}_air_depth" \
        --epoch "${epochs}" \
        --train_size "${TRAIN_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --num_samples "${NUM_SAMPLES}" \
        --learning_rate "${LEARNING_RATE}" \
        --beta1 "${BETA1}" \
        --input_height "${INPUT_HEIGHT}" \
        --input_width "${INPUT_WIDTH}" \
        --input_water_height "${INPUT_WATER_HEIGHT}" \
        --input_water_width "${INPUT_WATER_WIDTH}" \
        --output_height "${OUTPUT_HEIGHT}" \
        --output_width "${OUTPUT_WIDTH}" \
        --save_epoch "${SAVE_EPOCH}" \
        --checkpoint_dir "${checkpoint_dir}" \
        --sample_dir "${sample_dir}" \
        --results_dir "${result_dir}"
  ) 2>&1 | tee "${log_file}"

  local model_dir
  model_dir="${checkpoint_root}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
  require_path "${model_dir}"
  find "${model_dir}" -maxdepth 1 -type f -name 'DCGAN.model-*.index' \
    -printf '%f\n' |
    sort -V > "${log_dir}/checkpoint_steps.txt"

  local checkpoint_count
  checkpoint_count="$(wc -l < "${log_dir}/checkpoint_steps.txt" | tr -d ' ')"
  echo "DONE ${label}: checkpoints=${checkpoint_count}"
  if (( checkpoint_count < epochs )); then
    echo "Warning: ${label} saved ${checkpoint_count} checkpoints for ${epochs} epochs" >&2
  fi
}

run_experiment "e10_gpu${GPU_10E}" "${GPU_10E}" "${EPOCHS_10E}" &
pid_10e="$!"
run_experiment "e5_gpu${GPU_5E}" "${GPU_5E}" "${EPOCHS_5E}" &
pid_5e="$!"

failed=0
wait "${pid_10e}" || failed=1
wait "${pid_5e}" || failed=1

echo
echo "============================================================"
echo "Training summary"
echo "============================================================"
for label in "e10_gpu${GPU_10E}" "e5_gpu${GPU_5E}"; do
  echo
  echo "[${label}]"
  cat "${LOG_ROOT}/${label}/checkpoint_steps.txt" 2>/dev/null || true
  grep -aE '^Epoch:' "${LOG_ROOT}/${label}/train.log" 2>/dev/null | tail -n 1 || true
  grep -aEi 'Traceback|Nan in summary|InvalidArgumentError|CUBLAS|GEMM' \
    "${LOG_ROOT}/${label}/train.log" 2>/dev/null | tail -n 10 || true
done

exit "${failed}"
