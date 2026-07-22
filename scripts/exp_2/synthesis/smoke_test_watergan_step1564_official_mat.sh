#!/usr/bin/env bash
set -euo pipefail

# Validate the complete model-1564 official-MAT inference contract on exactly
# one batch. This script never creates full shards or starts full generation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
SOURCE_CHECKPOINT_NAME="${SOURCE_CHECKPOINT_NAME:-checkpoint_legacy_bs64_cumulative_epoch10_keepstep_v4}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_watergan_legacy_bs64_step1564_smoke}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1564}"
DATA_ROOT="${DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_train_full250k_ssd}"
MAT_ROOT="${MAT_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/smoke_step1564_official_mat64}"
RESULT_ROOT="${RESULT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_smoke64}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_official_mat_smoke64}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
MAT_WORKERS="${MAT_WORKERS:-16}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-2}"
RESET="${RESET:-1}"
KEEP_MAT="${KEEP_MAT:-1}"

MODEL_SUBDIR="${DATA_NAME}_water_images_${BATCH_SIZE}_48_64"
SOURCE_MODEL_DIR="${WATERGAN_DIR}/${SOURCE_CHECKPOINT_NAME}/${MODEL_SUBDIR}"
CHECKPOINT_ROOT="${WATERGAN_DIR}/${CHECKPOINT_NAME}"
MODEL_DIR="${CHECKPOINT_ROOT}/${MODEL_SUBDIR}"
ALIAS="${DATA_NAME}_official_mat_smoke64"
ALIAS_MODEL="${CHECKPOINT_ROOT}/${ALIAS}_water_images_${BATCH_SIZE}_48_64"
INFER_LOG="${LOG_ROOT}/inference_gpu${GPU}.log"
PASS_MARKER="${LOG_ROOT}/smoke_passed.env"

require_file() {
  [[ -s "$1" ]] || { echo "Error: required file missing: $1" >&2; exit 1; }
}

safe_reset() {
  local path="$1" expected="$2"
  [[ "${path}" == "${expected}" ]] || {
    echo "Error: refusing to reset unexpected path: ${path}" >&2
    exit 1
  }
  rm -rf "${path}"
}

[[ "${BATCH_SIZE}" -eq 64 && "${NUM_SAMPLES}" -eq 64 ]] || {
  echo "Error: this smoke test requires BATCH_SIZE=64 and NUM_SAMPLES=64" >&2
  exit 1
}
[[ "${MAT_WORKERS}" -gt 0 ]] || { echo "Error: MAT_WORKERS must be positive" >&2; exit 1; }
require_file "${DATA_ROOT}/watergan_air_manifest.jsonl"
for suffix in index meta data-00000-of-00001; do
  require_file "${SOURCE_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}"
done

if [[ "${RESET}" == 1 ]]; then
  safe_reset "${MAT_ROOT}" "/media/SSD1/XCX/exp_2/synthesis_work/watergan/smoke_step1564_official_mat64"
  safe_reset "${RESULT_ROOT}" "/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_smoke64"
  safe_reset "${LOG_ROOT}" "${REPO_ROOT}/logs/synthesis_full/watergan_step1564_official_mat_smoke64"
fi
mkdir -p "${MODEL_DIR}" "${MAT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}" "${WATERGAN_DIR}/data"

for suffix in index meta data-00000-of-00001; do
  source_file="${SOURCE_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}"
  frozen_file="${MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}"
  if [[ -e "${frozen_file}" ]]; then
    cmp -s "${source_file}" "${frozen_file}" || {
      echo "Error: frozen checkpoint differs from source: ${frozen_file}" >&2
      exit 1
    }
  else
    cp -a "${source_file}" "${frozen_file}"
  fi
done
printf '%s\n' \
  "model_checkpoint_path: \"DCGAN.model-${CHECKPOINT_STEP}\"" \
  "all_model_checkpoint_paths: \"DCGAN.model-${CHECKPOINT_STEP}\"" \
  > "${MODEL_DIR}/checkpoint"

WATERGAN_DIR="${WATERGAN_DIR}" bash "${SCRIPT_DIR}/patch_watergan_gpu_selection.sh"
WATERGAN_DIR="${WATERGAN_DIR}" bash "${SCRIPT_DIR}/patch_watergan_inference_aux_outputs.sh"

cat <<EOF
============================================================
WaterGAN model-1564 official-MAT isolated smoke test
============================================================
CHECKPOINT:   ${MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}
DATA ROOT:    ${DATA_ROOT}
GPU:          ${GPU}
SAMPLES:      ${NUM_SAMPLES}
BATCH SIZE:   ${BATCH_SIZE}
MAT WORKERS:  ${MAT_WORKERS}
MAT ROOT:     ${MAT_ROOT}
RESULT ROOT:  ${RESULT_ROOT}
INFER LOG:    ${INFER_LOG}
RESET:        ${RESET}
KEEP MAT:     ${KEEP_MAT}
FULL RUN:     disabled by design
============================================================
EOF

python tools/materialize_watergan_official_mat_shard.py \
  --source-shard "${DATA_ROOT}" \
  --out-dir "${MAT_ROOT}" \
  --workers "${MAT_WORKERS}" \
  --limit "${NUM_SAMPLES}" \
  --reset

mat_count="$(find "${MAT_ROOT}/air_depth" -maxdepth 1 -type f -name '*.mat' | wc -l | tr -d ' ')"
air_count="$(find "${MAT_ROOT}/air_images" -maxdepth 1 \( -type f -o -type l \) | wc -l | tr -d ' ')"
[[ "${mat_count}" -eq 64 && "${air_count}" -eq 64 ]] || {
  echo "Error: materialized counts air=${air_count}/64 depth=${mat_count}/64" >&2
  exit 1
}

[[ ! -e "${ALIAS_MODEL}" || -L "${ALIAS_MODEL}" ]] || {
  echo "Error: checkpoint alias is not a symlink: ${ALIAS_MODEL}" >&2
  exit 1
}
ln -sfn "${MODEL_DIR}" "${ALIAS_MODEL}"
ln -sfn "${MAT_ROOT}/air_images" "${WATERGAN_DIR}/data/${ALIAS}_air_images"
ln -sfn "${MAT_ROOT}/air_depth" "${WATERGAN_DIR}/data/${ALIAS}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" "${WATERGAN_DIR}/data/${ALIAS}_water_images"

(
  cd "${WATERGAN_DIR}"
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    WATERGAN_SAVE_AUX_OUTPUTS=0 \
    WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python mainmhl.py \
      --is_train=False \
      --water_dataset "${ALIAS}_water_images" \
      --air_dataset "${ALIAS}_air_images" \
      --depth_dataset "${ALIAS}_air_depth" \
      --checkpoint_dir "${CHECKPOINT_ROOT}" \
      --sample_dir "samples_${ALIAS}" \
      --results_dir "${RESULT_ROOT}" \
      --epoch 1 --num_samples 64 --train_size 64 --batch_size 64 \
      --input_height 480 --input_width 640 \
      --input_water_height 1024 --input_water_width 1360 \
      --output_height 48 --output_width 64
) > "${INFER_LOG}" 2>&1 || {
  echo "Error: inference failed. Last 120 log lines:" >&2
  tail -n 120 "${INFER_LOG}" >&2
  exit 1
}

grep -aq "Success to read DCGAN.model-${CHECKPOINT_STEP}" "${INFER_LOG}" || {
  echo "Error: log does not confirm loading DCGAN.model-${CHECKPOINT_STEP}" >&2
  grep -aE 'Reading checkpoints|Restoring parameters|Success to read|Load failed|Traceback' "${INFER_LOG}" >&2 || true
  exit 1
}
grep -aq 'Load SUCCESS' "${INFER_LOG}" || {
  echo "Error: log does not contain Load SUCCESS" >&2
  exit 1
}

fake_count="$(find "${RESULT_ROOT}" -maxdepth 1 -type f -name 'fake_*.png' | wc -l | tr -d ' ')"
[[ "${fake_count}" -eq 64 ]] || {
  echo "Error: generated fake outputs=${fake_count}/64" >&2
  exit 1
}

python - "${RESULT_ROOT}" <<'PY'
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
files = sorted(root.glob('fake_*.png'))
bad = []
sizes = {}
for path in files:
    try:
        with Image.open(str(path)) as image:
            image.load()
            sizes[image.size] = sizes.get(image.size, 0) + 1
    except (OSError, TypeError, ValueError) as error:
        bad.append((str(path), repr(error)))
print('decoded:', len(files) - len(bad))
print('bad:', len(bad))
print('sizes:', sizes)
if bad:
    for path, error in bad[:10]:
        print('BAD:', path, error)
    raise SystemExit('generated PNG decode validation failed')
PY

if [[ "${KEEP_MAT}" != 1 ]]; then
  safe_reset "${MAT_ROOT}" "/media/SSD1/XCX/exp_2/synthesis_work/watergan/smoke_step1564_official_mat64"
fi

cat > "${PASS_MARKER}" <<EOF
checkpoint_step=${CHECKPOINT_STEP}
checkpoint=${MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}
data_root=${DATA_ROOT}
batch_size=${BATCH_SIZE}
num_samples=${NUM_SAMPLES}
fake_count=${fake_count}
passed_at=$(date --iso-8601=seconds)
EOF

cat <<EOF
============================================================
SMOKE TEST PASSED
============================================================
checkpoint: DCGAN.model-${CHECKPOINT_STEP}
MAT depth:  ${mat_count}/64
fake PNG:   ${fake_count}/64
result:     ${RESULT_ROOT}
log:        ${INFER_LOG}
marker:     ${PASS_MARKER}
full run:   NOT STARTED
============================================================
EOF
