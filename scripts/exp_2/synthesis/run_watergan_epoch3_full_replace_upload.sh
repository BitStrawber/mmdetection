#!/usr/bin/env bash
set -euo pipefail

# End-to-end WaterGAN replacement pipeline:
#   1. Resume the legacy batch-64 trajectory from cumulative epoch 2.
#   2. Preserve global steps, train through epoch 10, and freeze epoch 5.
#   3. Generate train/val with 48 shards on 8 GPUs (16 concurrent workers).
#   4. Restore the ImageNet class layout, atomically replace old outputs.
#   5. Package train+val and upload the replacement archive to Hugging Face.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/${DATA_NAME}}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-${WATERGAN_DIR}/checkpoint_legacy_official_bs64_e10_gpu0}"
SOURCE_CHECKPOINT_STEP="${SOURCE_CHECKPOINT_STEP:-1564}"

RESUME_CHECKPOINT_NAME="${RESUME_CHECKPOINT_NAME:-checkpoint_legacy_bs64_cumulative_epoch10_pipeline}"
RESUME_CHECKPOINT_ROOT="${WATERGAN_DIR}/${RESUME_CHECKPOINT_NAME}"
FINAL_CHECKPOINT_NAME="${FINAL_CHECKPOINT_NAME:-checkpoint_watergan_final_legacy_bs64_cumulative_epoch5}"
FINAL_CHECKPOINT_ROOT="${WATERGAN_DIR}/${FINAL_CHECKPOINT_NAME}"

TRAIN_GPU="${TRAIN_GPU:-0}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS="${NUM_SHARDS:-48}"
GENERATE_WORKERS="${GENERATE_WORKERS:-16}"
RESTORE_WORKERS="${RESTORE_WORKERS:-16}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-4}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_train_full250k_ssd}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_val_full10k_infer_ssd}"
SHARD_DATA_ROOT="${SHARD_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_epoch5_48shards}"
FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_epoch5_flat_results_48shards}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
FINAL_GENERATED_ROOT="${FINAL_GENERATED_ROOT:-${SYN_ROOT}/watergan/generated}"
STAGING_GENERATED_ROOT="${STAGING_GENERATED_ROOT:-${SYN_ROOT}/watergan/generated_epoch5_staging}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/media/HDD2/XCX/exp_2/transfer_archives}"
UPLOAD_STAGE="${UPLOAD_STAGE:-/media/HDD2/XCX/exp_2/hf_watergan_epoch5_upload_stage}"
HF_REPO_ID="${HF_REPO_ID:-BitStrawber/transfer}"
HF_BIN="${HF_BIN:-/media/SSD1/conda_envs/hf_transfer/bin/hf}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_epoch5_full_replace}"
TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-/media/HDD2/XCX/exp_2/watergan_epoch5_resume_training}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_PUBLISH="${RUN_PUBLISH:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
RUN_UPLOAD="${RUN_UPLOAD:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RESET_TRAIN="${RESET_TRAIN:-0}"
RESET_SHARDS="${RESET_SHARDS:-0}"
RESET_GENERATION_OUTPUTS="${RESET_GENERATION_OUTPUTS:-0}"
REPLACE_FINAL="${REPLACE_FINAL:-1}"
DELETE_REPLACED_FINAL="${DELETE_REPLACED_FINAL:-1}"
REPLACE_ARCHIVE="${REPLACE_ARCHIVE:-1}"
VERIFY_ARCHIVE="${VERIFY_ARCHIVE:-0}"

SOURCE_MODEL_DIR="${SOURCE_CHECKPOINT_ROOT}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
RESUME_MODEL_DIR="${RESUME_CHECKPOINT_ROOT}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
FINAL_MODEL_DIR="${FINAL_CHECKPOINT_ROOT}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
TRAIN_LOG="${LOG_ROOT}/training/resume_to_epoch10.log"
TRAIN_PID_FILE="${LOG_ROOT}/training/resume_to_epoch10.pid"
TRAIN_COMPLETE_MARKER="${RESUME_CHECKPOINT_ROOT}/.cumulative_epoch10_complete"
FINAL_CHECKPOINT_MARKER="${FINAL_CHECKPOINT_ROOT}/.cumulative_epoch5_frozen"
training_monitor_pid=""
train_pid=""

mkdir -p "${LOG_ROOT}" "${LOG_ROOT}/training"

count_images() {
  if [[ ! -d "$1" ]]; then
    echo 0
    return
  fi
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

require_file() {
  [[ -s "$1" ]] || { echo "Error: required file missing or empty: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Error: required directory not found: $1" >&2; exit 1; }
}

checkpoint_complete() {
  local root="$1" step="$2" suffix
  for suffix in index meta data-00000-of-00001; do
    [[ -s "${root}/DCGAN.model-${step}.${suffix}" ]] || return 1
  done
}

safe_remove_tree() {
  local target="$1" prefix="$2"
  case "${target}" in
    "${prefix}"*) ;;
    *) echo "Error: refusing to remove unexpected path: ${target}" >&2; exit 1 ;;
  esac
  [[ "${target}" != "${prefix}" ]] || {
    echo "Error: refusing to remove path prefix itself: ${target}" >&2; exit 1;
  }
  rm -rf -- "${target}"
}

cat <<EOF
============================================================
WaterGAN cumulative epoch-5 full replacement pipeline (train through epoch 10)
============================================================
WATERGAN_DIR:             ${WATERGAN_DIR}
SOURCE_CHECKPOINT:        ${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}
RESUME_CHECKPOINT_ROOT:   ${RESUME_CHECKPOINT_ROOT}
FINAL_CHECKPOINT_ROOT:    ${FINAL_CHECKPOINT_ROOT}
TRAIN_GPU:                ${TRAIN_GPU}
GPUS:                     ${GPUS}
NUM_SHARDS:               ${NUM_SHARDS}
GENERATE_WORKERS:         ${GENERATE_WORKERS}
RESTORE_WORKERS:          ${RESTORE_WORKERS}
UPLOAD_WORKERS:           ${UPLOAD_WORKERS}
TRAIN_DATA_ROOT:          ${TRAIN_DATA_ROOT}
VAL_DATA_ROOT:            ${VAL_DATA_ROOT}
STAGING_GENERATED_ROOT:   ${STAGING_GENERATED_ROOT}
FINAL_GENERATED_ROOT:     ${FINAL_GENERATED_ROOT}
ARCHIVE_ROOT:             ${ARCHIVE_ROOT}
HF_REPO_ID:               ${HF_REPO_ID}
Stages: train=${RUN_TRAIN}, generate=${RUN_GENERATE}, publish=${RUN_PUBLISH}, package=${RUN_PACKAGE}, upload=${RUN_UPLOAD}
CHECK_ONLY:                ${CHECK_ONLY}
============================================================
EOF

require_dir "${WATERGAN_DIR}"
require_dir "${TRAINING_DATA_ROOT}"
require_dir "${TRAIN_DATA_ROOT}"
require_dir "${VAL_DATA_ROOT}"

[[ "${SOURCE_CHECKPOINT_STEP}" =~ ^[0-9]+$ ]] && \
  (( SOURCE_CHECKPOINT_STEP > 0 )) || {
  echo "Error: SOURCE_CHECKPOINT_STEP must be a positive integer; got ${SOURCE_CHECKPOINT_STEP}" >&2
  exit 1
}

for suffix in index meta data-00000-of-00001; do
  require_file "${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}"
done

for data_root in "${TRAIN_DATA_ROOT}" "${VAL_DATA_ROOT}"; do
  require_file "${data_root}/watergan_air_manifest.jsonl"
  require_dir "${data_root}/air_images"
  require_dir "${data_root}/air_depth"
  require_dir "${data_root}/water_images"
done

if [[ "${CHECK_ONLY}" == 1 ]]; then
  echo
  echo "Preflight check complete. No training, generation, replacement, packaging, or upload was started."
  exit 0
fi

if [[ "${RUN_TRAIN}" == 1 ]]; then
  echo
  echo "===== Stage 1: resume through cumulative epoch 10 ====="

  WATERGAN_DIR="${WATERGAN_DIR}" \
    bash "${SCRIPT_DIR}/patch_watergan_gpu_selection.sh"

  if [[ -f "${TRAIN_COMPLETE_MARKER}" && -f "${FINAL_CHECKPOINT_MARKER}" ]] && \
     checkpoint_complete "${RESUME_MODEL_DIR}" 7812 && \
     checkpoint_complete "${FINAL_MODEL_DIR}" 3907; then
    echo "Reuse complete cumulative epoch-3..10 trajectory and frozen epoch-5 checkpoint."
  else
    if [[ "${RESET_TRAIN}" == 1 ]]; then
      safe_remove_tree "${RESUME_CHECKPOINT_ROOT}" "${WATERGAN_DIR}/checkpoint_legacy_bs64_"
      safe_remove_tree "${FINAL_CHECKPOINT_ROOT}" "${WATERGAN_DIR}/checkpoint_watergan_final_"
    fi
    [[ ! -e "${RESUME_CHECKPOINT_ROOT}" ]] || {
      echo "Error: resume checkpoint root already exists without a completion marker:" >&2
      echo "  ${RESUME_CHECKPOINT_ROOT}" >&2
      echo "Inspect it, then use RESET_TRAIN=1 to restart this stage." >&2
      exit 1
    }
    [[ ! -e "${FINAL_CHECKPOINT_ROOT}" ]] || {
      echo "Error: final checkpoint root already exists without a completion marker: ${FINAL_CHECKPOINT_ROOT}" >&2
      exit 1
    }

    mkdir -p "${RESUME_MODEL_DIR}" "${TRAIN_RESULT_ROOT}"
    for suffix in index meta data-00000-of-00001; do
      cp -a \
        "${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}" \
        "${RESUME_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}.${suffix}"
    done
    cat > "${RESUME_MODEL_DIR}/checkpoint" <<EOF
model_checkpoint_path: "DCGAN.model-${SOURCE_CHECKPOINT_STEP}"
all_model_checkpoint_paths: "DCGAN.model-${SOURCE_CHECKPOINT_STEP}"
EOF

    mkdir -p "${WATERGAN_DIR}/data"
    ln -sfn "${TRAINING_DATA_ROOT}/air_images" "${WATERGAN_DIR}/data/${DATA_NAME}_air_images"
    ln -sfn "${TRAINING_DATA_ROOT}/air_depth" "${WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
    ln -sfn "${TRAINING_DATA_ROOT}/water_images" "${WATERGAN_DIR}/data/${DATA_NAME}_water_images"

    WATERGAN_DIR="${WATERGAN_DIR}" \
      bash "${SCRIPT_DIR}/patch_watergan_resume_counter.sh"

    # Keep the loaded checkpoint named model-${SOURCE_CHECKPOINT_STEP}. The
    # runtime value is one lower only because this legacy training loop
    # increments counter immediately before writing the next checkpoint.
    resume_counter_start=$((SOURCE_CHECKPOINT_STEP - 1))

    # Nine loop epochs are required because legacy WaterGAN saves the
    # completed epoch when the next loop epoch starts.
    (
      cd "${WATERGAN_DIR}"
      env -u LD_LIBRARY_PATH -u LD_PRELOAD \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        PYTHONUNBUFFERED=1 \
        WATERGAN_COUNTER_START="${resume_counter_start}" \
        WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
        WATERGAN_MAX_TO_KEEP=20 \
        python mainmhl.py \
          --is_train=True \
          --water_dataset "${DATA_NAME}_water_images" \
          --air_dataset "${DATA_NAME}_air_images" \
          --depth_dataset "${DATA_NAME}_air_depth" \
          --epoch 9 --train_size 50000 --batch_size "${BATCH_SIZE}" \
          --num_samples 64 --learning_rate 0.0002 --beta1 0.5 \
          --input_height 480 --input_width 640 \
          --input_water_height 1024 --input_water_width 1360 \
          --output_height "${OUTPUT_HEIGHT}" --output_width "${OUTPUT_WIDTH}" \
          --save_epoch 1 \
          --checkpoint_dir "${RESUME_CHECKPOINT_NAME}" \
          --sample_dir samples_legacy_bs64_cumulative_epoch10_pipeline \
          --results_dir "${TRAIN_RESULT_ROOT}"
    ) > "${TRAIN_LOG}" 2>&1 &
    train_pid="$!"
    echo "${train_pid}" > "${TRAIN_PID_FILE}"
    echo "Training PID=${train_pid}; log=${TRAIN_LOG}"

    load_deadline=$((SECONDS + 600))
    while ! grep -aqE 'Success to read|Load failed|Failed to find a checkpoint|Traceback' "${TRAIN_LOG}" 2>/dev/null; do
      kill -0 "${train_pid}" 2>/dev/null || {
        echo "Error: training stopped before checkpoint load confirmation" >&2
        tail -n 100 "${TRAIN_LOG}" >&2
        exit 1
      }
      (( SECONDS < load_deadline )) || { echo "Error: checkpoint load confirmation timed out" >&2; exit 1; }
      sleep 5
    done
    grep -aq "Success to read DCGAN.model-${SOURCE_CHECKPOINT_STEP}" "${TRAIN_LOG}" || {
      echo "Error: training did not load DCGAN.model-${SOURCE_CHECKPOINT_STEP}" >&2
      grep -aE 'Reading checkpoints|Restoring parameters|Success to read|Load failed|Traceback' "${TRAIN_LOG}" >&2 || true
      kill -TERM "${train_pid}" 2>/dev/null || true
      exit 1
    }
    echo "Checkpoint seed loaded successfully. Waiting for cumulative epoch 5."

    # A later save may advance the TensorFlow checkpoint pointer before this
    # polling loop wakes up. The immutable model-3907 file triple is the
    # authoritative readiness condition; the pointer does not need to remain
    # on model-3907 after those files are complete.
    while ! checkpoint_complete "${RESUME_MODEL_DIR}" 3907; do
      if ! kill -0 "${train_pid}" 2>/dev/null; then
        echo "Error: training stopped before cumulative epoch 5 was saved" >&2
        tail -n 120 "${TRAIN_LOG}" >&2
        exit 1
      fi
      grep -aE '^Epoch:' "${TRAIN_LOG}" 2>/dev/null | tail -n 1 || true
      sleep 15
    done

    epoch_steps=(2345 3126 3907)
    epoch_number=3
    for epoch_step in "${epoch_steps[@]}"; do
      checkpoint_complete "${RESUME_MODEL_DIR}" "${epoch_step}" || {
        echo "Error: cumulative epoch-${epoch_number} checkpoint model-${epoch_step} is incomplete" >&2
        exit 1
      }
      echo "Verified cumulative epoch ${epoch_number}: DCGAN.model-${epoch_step}"
      epoch_number=$((epoch_number + 1))
    done
    mkdir -p "${FINAL_MODEL_DIR}"
    for suffix in index meta data-00000-of-00001; do
      cp -a \
        "${RESUME_MODEL_DIR}/DCGAN.model-3907.${suffix}" \
        "${FINAL_MODEL_DIR}/DCGAN.model-3907.${suffix}"
    done
    cat > "${FINAL_MODEL_DIR}/checkpoint" <<'EOF'
model_checkpoint_path: "DCGAN.model-3907"
all_model_checkpoint_paths: "DCGAN.model-3907"
EOF
    cat > "${FINAL_CHECKPOINT_MARKER}" <<EOF
source=${SOURCE_MODEL_DIR}/DCGAN.model-${SOURCE_CHECKPOINT_STEP}
trajectory=cumulative_epoch5
target=DCGAN.model-3907
created=$(date --iso-8601=seconds)
EOF
    echo "Frozen cumulative epoch-5 checkpoint: ${FINAL_MODEL_DIR}/DCGAN.model-3907"

    # Continue training in parallel with generation. This monitor owns the
    # epoch-10 completion check while the main process proceeds to inference,
    # restoration, packaging, and upload with the frozen epoch-5 checkpoint.
    (
      while ! checkpoint_complete "${RESUME_MODEL_DIR}" 7812; do
        if ! kill -0 "${train_pid}" 2>/dev/null; then
          echo "Error: training stopped before cumulative epoch 10 was saved" >&2
          tail -n 120 "${TRAIN_LOG}" >&2
          exit 1
        fi
        sleep 15
      done

      epoch_steps=(2345 3126 3907 4688 5469 6250 7031 7812)
      epoch_number=3
      for epoch_step in "${epoch_steps[@]}"; do
        checkpoint_complete "${RESUME_MODEL_DIR}" "${epoch_step}" || {
          echo "Error: cumulative epoch-${epoch_number} checkpoint model-${epoch_step} is incomplete" >&2
          exit 1
        }
        echo "Verified cumulative epoch ${epoch_number}: DCGAN.model-${epoch_step}"
        epoch_number=$((epoch_number + 1))
      done

      echo "Cumulative epoch 10 saved; stop the unnecessary final loop epoch."
      kill -TERM "${train_pid}" 2>/dev/null || true
      date > "${TRAIN_COMPLETE_MARKER}"
    ) &
    training_monitor_pid="$!"
    echo "Training continues through epoch 10 in parallel; monitor PID=${training_monitor_pid}"
  fi
fi

checkpoint_complete "${FINAL_MODEL_DIR}" 3907 || {
  echo "Error: final cumulative epoch-5 checkpoint is unavailable: ${FINAL_MODEL_DIR}" >&2
  exit 1
}

if [[ "${RUN_GENERATE}" == 1 ]]; then
  echo
  echo "===== Stage 2: 48-shard train/val generation ====="
  staged_train="$(count_images "${STAGING_GENERATED_ROOT}/train")"
  staged_val="$(count_images "${STAGING_GENERATED_ROOT}/val")"
  if [[ "${staged_train}" -eq 250000 && "${staged_val}" -eq 10000 ]]; then
    echo "Reuse complete staged generation: ${STAGING_GENERATED_ROOT}"
  else
    CHECKPOINT_DIR="${FINAL_CHECKPOINT_NAME}" \
    CHECKPOINT_STEP=3907 \
    TRAIN_DATA_NAME="${DATA_NAME}" \
    WATERGAN_DIR="${WATERGAN_DIR}" \
    TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT}" \
    VAL_DATA_ROOT="${VAL_DATA_ROOT}" \
    SHARD_DATA_ROOT="${SHARD_DATA_ROOT}" \
    FLAT_ROOT="${FLAT_ROOT}" \
    FINAL_ROOT="${STAGING_GENERATED_ROOT}" \
    RESTORE_SHARD_ROOT="${STAGING_GENERATED_ROOT}/.parallel_shards" \
    LOG_ROOT="${LOG_ROOT}/generation" \
    GPUS="${GPUS}" SPLITS="train val" NUM_SHARDS="${NUM_SHARDS}" \
    GENERATE_WORKERS="${GENERATE_WORKERS}" RESTORE_WORKERS="${RESTORE_WORKERS}" \
    WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
    BATCH_SIZE="${BATCH_SIZE}" PAD_TO_BATCH=1 \
    RESET_SHARDS="${RESET_SHARDS}" RESET_OUTPUTS="${RESET_GENERATION_OUTPUTS}" \
    CLEAN_RESTORE_SHARDS=1 \
      bash "${SCRIPT_DIR}/run_watergan_parallel_generate_train_val.sh"
  fi
fi

if [[ "${RUN_PUBLISH}" == 1 ]]; then
  echo
  echo "===== Stage 3: replace standard generated/train and generated/val ====="
  train_count="$(count_images "${STAGING_GENERATED_ROOT}/train")"
  val_count="$(count_images "${STAGING_GENERATED_ROOT}/val")"
  [[ "${train_count}" -eq 250000 && "${val_count}" -eq 10000 ]] || {
    echo "Error: staged generated counts are incomplete: train=${train_count}, val=${val_count}" >&2
    exit 1
  }
  existing_train="$(count_images "${FINAL_GENERATED_ROOT}/train")"
  existing_val="$(count_images "${FINAL_GENERATED_ROOT}/val")"
  if [[ "${existing_train}" -eq 250000 && "${existing_val}" -eq 10000 ]]; then
    [[ "${REPLACE_FINAL}" == 1 ]] || {
      echo "Error: complete old WaterGAN output exists; set REPLACE_FINAL=1" >&2; exit 1;
    }
    backup="${FINAL_GENERATED_ROOT}.replaced.$(date +%Y%m%d_%H%M%S)"
    mv "${FINAL_GENERATED_ROOT}" "${backup}"
    mv "${STAGING_GENERATED_ROOT}" "${FINAL_GENERATED_ROOT}"
    if [[ "${DELETE_REPLACED_FINAL}" == 1 ]]; then
      safe_remove_tree "${backup}" "${SYN_ROOT}/watergan/generated.replaced."
    else
      echo "Preserved replaced output: ${backup}"
    fi
  elif [[ "${existing_train}" -eq 0 && "${existing_val}" -eq 0 ]]; then
    mkdir -p "$(dirname "${FINAL_GENERATED_ROOT}")"
    rm -rf "${FINAL_GENERATED_ROOT}"
    mv "${STAGING_GENERATED_ROOT}" "${FINAL_GENERATED_ROOT}"
  else
    echo "Error: existing final output is partial: train=${existing_train}, val=${existing_val}" >&2
    exit 1
  fi
  echo "Published train=$(count_images "${FINAL_GENERATED_ROOT}/train")"
  echo "Published val=$(count_images "${FINAL_GENERATED_ROOT}/val")"
fi

if [[ "${RUN_PACKAGE}" == 1 ]]; then
  echo
  echo "===== Stage 4: replace WaterGAN transfer archive ====="
  mkdir -p "${ARCHIVE_ROOT}"
  archive="${ARCHIVE_ROOT}/watergan_train_val.tar"
  if [[ -e "${archive}" || -e "${archive}.sha256" || -e "${archive}.partial" ]]; then
    [[ "${REPLACE_ARCHIVE}" == 1 ]] || {
      echo "Error: old WaterGAN archive exists; set REPLACE_ARCHIVE=1" >&2; exit 1;
    }
    if command -v fuser >/dev/null 2>&1; then
      for artifact in "${archive}" "${archive}.sha256" "${archive}.partial"; do
        if [[ -e "${artifact}" ]] && fuser "${artifact}" >/dev/null 2>&1; then
          echo "Error: archive artifact is held by another process: ${artifact}" >&2
          exit 1
        fi
      done
    fi
    rm -f -- "${archive}" "${archive}.sha256" "${archive}.partial"
  fi
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    ARCHIVE_ROOT="${ARCHIVE_ROOT}" SYN_ROOT="${SYN_ROOT}" METHODS=watergan \
    RESET_PARTIAL=1 VERIFY_ARCHIVE="${VERIFY_ARCHIVE}" \
    TRAIN_EXPECTED=250000 VAL_EXPECTED=10000 \
      /bin/bash "${SCRIPT_DIR}/package_completed_synthesis_train_val.sh"
fi

archive="${ARCHIVE_ROOT}/watergan_train_val.tar"
require_file "${archive}"
require_file "${archive}.sha256"

if [[ "${RUN_UPLOAD}" == 1 ]]; then
  echo
  echo "===== Stage 5: Hugging Face replacement upload ====="
  [[ -x "${HF_BIN}" ]] || {
    HF_BIN="$(command -v hf || true)"
    [[ -n "${HF_BIN}" ]] || { echo "Error: hf CLI not found" >&2; exit 1; }
  }
  "${HF_BIN}" auth whoami
  safe_remove_tree "${UPLOAD_STAGE}" "/media/HDD2/XCX/exp_2/hf_watergan_"
  mkdir -p "${UPLOAD_STAGE}/archives"
  ln "${archive}" "${UPLOAD_STAGE}/archives/watergan_train_val.tar"
  cp -a "${ARCHIVE_ROOT}/SHA256SUMS.txt" "${UPLOAD_STAGE}/archives/SHA256SUMS.txt"

  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HUB_ETAG_TIMEOUT=60
  export HF_HUB_UPLOAD_TIMEOUT=3600
  "${HF_BIN}" upload-large-folder \
    "${HF_REPO_ID}" "${UPLOAD_STAGE}" \
    --repo-type dataset \
    --num-workers "${UPLOAD_WORKERS}"
fi

if [[ -n "${training_monitor_pid}" ]]; then
  echo
  echo "===== Final stage: wait for cumulative epoch 10 training ====="
  wait "${training_monitor_pid}"
  wait "${train_pid}" 2>/dev/null || true
  checkpoint_complete "${RESUME_MODEL_DIR}" 7812 || {
    echo "Error: cumulative epoch-10 checkpoint is incomplete after monitoring" >&2
    exit 1
  }
  echo "Training and downstream epoch-5 generation pipeline are both complete."
fi

echo
echo "============================================================"
echo "WaterGAN epoch-5 replacement pipeline complete"
echo "============================================================"
echo "Checkpoint: ${FINAL_MODEL_DIR}/DCGAN.model-3907"
echo "Train:      ${FINAL_GENERATED_ROOT}/train ($(count_images "${FINAL_GENERATED_ROOT}/train"))"
echo "Val:        ${FINAL_GENERATED_ROOT}/val ($(count_images "${FINAL_GENERATED_ROOT}/val"))"
echo "Archive:    ${archive}"
echo "HF repo:    https://huggingface.co/datasets/${HF_REPO_ID}"
