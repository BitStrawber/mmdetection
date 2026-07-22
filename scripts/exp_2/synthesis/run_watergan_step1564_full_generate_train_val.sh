#!/usr/bin/env bash
set -euo pipefail

# Freeze the selected legacy WaterGAN model-1564 checkpoint and generate the
# complete ImageNet train/val sets with 48 batch-aligned inference shards.
# Outputs are staged separately so an existing published WaterGAN dataset is
# never mixed with or deleted by an interrupted run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"

SOURCE_CHECKPOINT_NAME="${SOURCE_CHECKPOINT_NAME:-checkpoint_legacy_bs64_cumulative_epoch10_keepstep_v4}"
SOURCE_CHECKPOINT_STEP="${SOURCE_CHECKPOINT_STEP:-1564}"
FROZEN_CHECKPOINT_NAME="${FROZEN_CHECKPOINT_NAME:-checkpoint_watergan_legacy_bs64_step1564_final}"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_train_full250k_ssd}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_val_full10k_infer_ssd}"
SHARD_DATA_ROOT="${SHARD_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_step1564_48shards}"
FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_flat_results_48shards}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated_step1564}"
RESTORE_SHARD_ROOT="${RESTORE_SHARD_ROOT:-${FINAL_ROOT}/.parallel_shards}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_full_48shards}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS="${NUM_SHARDS:-48}"
GENERATE_WORKERS="${GENERATE_WORKERS:-16}"
RESTORE_WORKERS="${RESTORE_WORKERS:-16}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-4}"
RESET_SHARDS="${RESET_SHARDS:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"
CLEAN_RESTORE_SHARDS="${CLEAN_RESTORE_SHARDS:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

SOURCE_CHECKPOINT_ROOT="${WATERGAN_DIR}/${SOURCE_CHECKPOINT_NAME}"
FROZEN_CHECKPOINT_ROOT="${WATERGAN_DIR}/${FROZEN_CHECKPOINT_NAME}"
MODEL_SUBDIR="${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
SOURCE_MODEL_DIR="${SOURCE_CHECKPOINT_ROOT}/${MODEL_SUBDIR}"
FROZEN_MODEL_DIR="${FROZEN_CHECKPOINT_ROOT}/${MODEL_SUBDIR}"

require_file() {
  [[ -s "$1" ]] || {
    echo "Error: required file missing or empty: $1" >&2
    exit 1
  }
}

require_dir() {
  [[ -d "$1" ]] || {
    echo "Error: required directory not found: $1" >&2
    exit 1
  }
}

count_images() {
  if [[ ! -d "$1" ]]; then
    echo 0
    return
  fi
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

[[ "${SOURCE_CHECKPOINT_STEP}" =~ ^[0-9]+$ ]] || {
  echo "Error: SOURCE_CHECKPOINT_STEP must be an integer" >&2
  exit 1
}
[[ "${NUM_SHARDS}" -eq 48 ]] || {
  echo "Error: this final run requires NUM_SHARDS=48; got ${NUM_SHARDS}" >&2
  exit 1
}

require_dir "${WATERGAN_DIR}"
require_dir "${SOURCE_MODEL_DIR}"
require_dir "${TRAIN_DATA_ROOT}"
require_dir "${VAL_DATA_ROOT}"
require_file "${TRAIN_DATA_ROOT}/watergan_air_manifest.jsonl"
require_file "${VAL_DATA_ROOT}/watergan_air_manifest.jsonl"

for suffix in index meta data-00000-of-00001; do
  require_file "${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}"
done

train_count="$(wc -l < "${TRAIN_DATA_ROOT}/watergan_air_manifest.jsonl" | tr -d ' ')"
val_count="$(wc -l < "${VAL_DATA_ROOT}/watergan_air_manifest.jsonl" | tr -d ' ')"
[[ "${train_count}" -eq 250000 ]] || {
  echo "Error: train manifest=${train_count}, expected=250000" >&2
  exit 1
}
[[ "${val_count}" -eq 10000 ]] || {
  echo "Error: val manifest=${val_count}, expected=10000" >&2
  exit 1
}

cat <<EOF
============================================================
WaterGAN model-1564 full train+val generation
============================================================
SOURCE_CHECKPOINT:  ${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}
FROZEN_CHECKPOINT:  ${FROZEN_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}
GPUS:               ${GPUS}
NUM_SHARDS:         ${NUM_SHARDS}
GENERATE_WORKERS:   ${GENERATE_WORKERS}
RESTORE_WORKERS:    ${RESTORE_WORKERS}
TRAIN_DATA_ROOT:    ${TRAIN_DATA_ROOT} (${train_count})
VAL_DATA_ROOT:      ${VAL_DATA_ROOT} (${val_count})
SHARD_DATA_ROOT:    ${SHARD_DATA_ROOT}
FLAT_ROOT:          ${FLAT_ROOT}
FINAL_ROOT:         ${FINAL_ROOT}
RESET_SHARDS:       ${RESET_SHARDS}
RESET_OUTPUTS:      ${RESET_OUTPUTS}
CHECK_ONLY:         ${CHECK_ONLY}
============================================================
EOF

if [[ "${CHECK_ONLY}" == 1 ]]; then
  echo "Preflight check complete; generation was not started."
  exit 0
fi

# The source trajectory currently points at a later checkpoint. Give model-1564
# an isolated checkpoint state so TensorFlow restores the selected files.
mkdir -p "${FROZEN_MODEL_DIR}" "${LOG_ROOT}"
for suffix in index meta data-00000-of-00001; do
  source_file="${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}"
  frozen_file="${FROZEN_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}"
  if [[ -e "${frozen_file}" ]]; then
    cmp -s "${source_file}" "${frozen_file}" || {
      echo "Error: frozen checkpoint differs from source: ${frozen_file}" >&2
      exit 1
    }
  else
    cp -a "${source_file}" "${frozen_file}"
  fi
done
cat > "${FROZEN_MODEL_DIR}/checkpoint" <<EOF
model_checkpoint_path: "DCGAN.model-${SOURCE_CHECKPOINT_STEP}"
all_model_checkpoint_paths: "DCGAN.model-${SOURCE_CHECKPOINT_STEP}"
EOF

WATERGAN_DIR="${WATERGAN_DIR}" \
  bash "${SCRIPT_DIR}/patch_watergan_inference_aux_outputs.sh"

WATERGAN_DIR="${WATERGAN_DIR}" \
  bash "${SCRIPT_DIR}/patch_watergan_gpu_selection.sh"

WATERGAN_DIR="${WATERGAN_DIR}" \
TRAIN_DATA_NAME="${DATA_NAME}" \
CHECKPOINT_DIR="${FROZEN_CHECKPOINT_NAME}" \
CHECKPOINT_STEP="${SOURCE_CHECKPOINT_STEP}" \
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT}" \
VAL_DATA_ROOT="${VAL_DATA_ROOT}" \
SHARD_DATA_ROOT="${SHARD_DATA_ROOT}" \
FLAT_ROOT="${FLAT_ROOT}" \
FINAL_ROOT="${FINAL_ROOT}" \
RESTORE_SHARD_ROOT="${RESTORE_SHARD_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
GPUS="${GPUS}" \
SPLITS="train val" \
NUM_SHARDS="${NUM_SHARDS}" \
GENERATE_WORKERS="${GENERATE_WORKERS}" \
RESTORE_WORKERS="${RESTORE_WORKERS}" \
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
OUTPUT_HEIGHT="${OUTPUT_HEIGHT}" \
OUTPUT_WIDTH="${OUTPUT_WIDTH}" \
PAD_TO_BATCH=1 \
RESET_SHARDS="${RESET_SHARDS}" \
RESET_OUTPUTS="${RESET_OUTPUTS}" \
CLEAN_RESTORE_SHARDS="${CLEAN_RESTORE_SHARDS}" \
bash "${SCRIPT_DIR}/run_watergan_parallel_generate_train_val.sh"

final_train="$(count_images "${FINAL_ROOT}/train")"
final_val="$(count_images "${FINAL_ROOT}/val")"
[[ "${final_train}" -eq 250000 && "${final_val}" -eq 10000 ]] || {
  echo "Error: final counts train=${final_train}/250000 val=${final_val}/10000" >&2
  exit 1
}

cat <<EOF
============================================================
WaterGAN model-1564 generation complete
============================================================
train: ${final_train}/250000
val:   ${final_val}/10000
root:  ${FINAL_ROOT}
============================================================
EOF
