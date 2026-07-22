#!/usr/bin/env bash
set -euo pipefail

# Generate the full WaterGAN train/val set with the official MAT depth input
# contract. PNG depth is converted one shard at a time and removed after a
# successful inference, so the complete 260k MAT set is never stored at once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
SOURCE_CHECKPOINT_NAME="${SOURCE_CHECKPOINT_NAME:-checkpoint_legacy_bs64_cumulative_epoch10_keepstep_v4}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_watergan_legacy_bs64_step1564_final}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1564}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_train_full250k_ssd}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_val_full10k_infer_ssd}"
BASE_SHARD_ROOT="${BASE_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_step1564_official_base_48shards}"
MAT_SHARD_ROOT="${MAT_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_step1564_official_mat_active}"
FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_flat_48shards}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated_step1564_official_mat}"
RESTORE_ROOT="${RESTORE_ROOT:-${FINAL_ROOT}/.restore_shards}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_official_mat_48shards}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS="${NUM_SHARDS:-48}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-4}"
MAT_WORKERS_PER_PROCESS="${MAT_WORKERS_PER_PROCESS:-${MAT_WORKERS_PER_GPU:-2}}"
RESTORE_WORKERS="${RESTORE_WORKERS:-16}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-2}"
RESET_BASE_SHARDS="${RESET_BASE_SHARDS:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"
KEEP_FAILED_MAT="${KEEP_FAILED_MAT:-1}"
RUN_SMOKE="${RUN_SMOKE:-0}"
REQUIRE_SMOKE_PASS="${REQUIRE_SMOKE_PASS:-1}"
SMOKE_PASS_MARKER="${SMOKE_PASS_MARKER:-${REPO_ROOT}/logs/synthesis_full/watergan_step1564_official_mat_smoke64/smoke_passed.env}"

read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
MAX_CONCURRENT=$((NUM_GPUS * PROCESSES_PER_GPU))
MODEL_SUBDIR="${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
CHECKPOINT_ROOT="${WATERGAN_DIR}/${CHECKPOINT_NAME}"
MODEL_DIR="${CHECKPOINT_ROOT}/${MODEL_SUBDIR}"
SOURCE_MODEL_DIR="${WATERGAN_DIR}/${SOURCE_CHECKPOINT_NAME}/${MODEL_SUBDIR}"

export PYTHONUNBUFFERED=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export WATERGAN_SAVE_AUX_OUTPUTS=0 WATERGAN_IO_WORKERS
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

count_fake() {
  find "$1" -maxdepth 1 -type f -name 'fake_*.png' 2>/dev/null | wc -l | tr -d ' '
}

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    2>/dev/null | wc -l | tr -d ' '
}

require_file() {
  [[ -s "$1" ]] || { echo "Error: required file missing: $1" >&2; exit 1; }
}

safe_clear_dir() {
  local path="$1" prefix="$2"
  case "$path" in
    "${prefix}"*) rm -rf "$path" ;;
    *) echo "Error: refusing to remove unexpected path: $path" >&2; exit 1 ;;
  esac
}

[[ "${NUM_GPUS}" -eq 8 ]] || { echo "Error: exactly 8 GPUs are required; got ${NUM_GPUS}" >&2; exit 1; }
[[ "${NUM_SHARDS}" -eq 48 ]] || { echo "Error: NUM_SHARDS must be 48" >&2; exit 1; }
[[ "${PROCESSES_PER_GPU}" -gt 0 ]] || { echo "Error: PROCESSES_PER_GPU must be positive" >&2; exit 1; }
[[ "${MAX_CONCURRENT}" -le "${NUM_SHARDS}" ]] || {
  echo "Error: concurrent process slots ${MAX_CONCURRENT} exceed NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
}
[[ "${MAT_WORKERS_PER_PROCESS}" -gt 0 ]] || {
  echo "Error: MAT_WORKERS_PER_PROCESS must be positive" >&2
  exit 1
}
for suffix in index meta data-00000-of-00001; do
  require_file "${SOURCE_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}"
done
for item in "${TRAIN_DATA_ROOT}" "${VAL_DATA_ROOT}"; do
  require_file "${item}/watergan_air_manifest.jsonl"
done
[[ "$(wc -l < "${TRAIN_DATA_ROOT}/watergan_air_manifest.jsonl")" -eq 250000 ]] || {
  echo "Error: train manifest is not 250000" >&2; exit 1;
}
[[ "$(wc -l < "${VAL_DATA_ROOT}/watergan_air_manifest.jsonl")" -eq 10000 ]] || {
  echo "Error: val manifest is not 10000" >&2; exit 1;
}

mkdir -p "${LOG_ROOT}" "${BASE_SHARD_ROOT}" "${MAT_SHARD_ROOT}" \
  "${FLAT_ROOT}" "${RESTORE_ROOT}" "${WATERGAN_DIR}/data" "${MODEL_DIR}"

# Freeze the selected trajectory checkpoint under an isolated TensorFlow
# checkpoint state. This prevents a later checkpoint pointer from silently
# changing the model used by inference.
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
WaterGAN model-1564 official-MAT full generation
============================================================
CHECKPOINT:          ${MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}
GPUS:                ${GPUS}
SHARDS:              ${NUM_SHARDS}
PROCESSES PER GPU:   ${PROCESSES_PER_GPU}
CONCURRENT INFER:    ${MAX_CONCURRENT}
MAT WORKERS/PROCESS: ${MAT_WORKERS_PER_PROCESS}
MAX MAT WORKERS:     $((MAX_CONCURRENT * MAT_WORKERS_PER_PROCESS))
RESTORE WORKERS:     ${RESTORE_WORKERS}
TRAIN DATA:          ${TRAIN_DATA_ROOT}
VAL DATA:            ${VAL_DATA_ROOT}
TEMP MAT ROOT:       ${MAT_SHARD_ROOT}
FLAT ROOT:           ${FLAT_ROOT}
FINAL ROOT:          ${FINAL_ROOT}
RUN SMOKE:           ${RUN_SMOKE}
RESET OUTPUTS:       ${RESET_OUTPUTS}
REQUIRE SMOKE PASS:  ${REQUIRE_SMOKE_PASS}
SMOKE PASS MARKER:   ${SMOKE_PASS_MARKER}
============================================================
EOF

if [[ "${REQUIRE_SMOKE_PASS}" == 1 && "${RUN_SMOKE}" != 1 ]]; then
  require_file "${SMOKE_PASS_MARKER}"
  grep -qx "checkpoint_step=${CHECKPOINT_STEP}" "${SMOKE_PASS_MARKER}" || {
    echo "Error: smoke marker checkpoint does not match model-${CHECKPOINT_STEP}" >&2
    exit 1
  }
  grep -qx "data_root=${TRAIN_DATA_ROOT}" "${SMOKE_PASS_MARKER}" || {
    echo "Error: smoke marker data root does not match TRAIN_DATA_ROOT" >&2
    exit 1
  }
  grep -qx 'fake_count=64' "${SMOKE_PASS_MARKER}" || {
    echo "Error: smoke marker does not confirm 64 generated outputs" >&2
    exit 1
  }
  echo "Verified isolated smoke-test marker: ${SMOKE_PASS_MARKER}"
fi

prepare_base_split() {
  local split="$1" data_root="$2" args=()
  args=(
    --data-root "${data_root}"
    --out-root "${BASE_SHARD_ROOT}/${split}"
    --num-shards "${NUM_SHARDS}"
    --batch-size "${BATCH_SIZE}"
    --pad-to-batch
  )
  [[ "${RESET_BASE_SHARDS}" == 1 ]] && args+=(--reset)
  python tools/prepare_watergan_inference_shards.py "${args[@]}" \
    | tee "${LOG_ROOT}/prepare_${split}_base_shards.log"
}

prepare_base_split train "${TRAIN_DATA_ROOT}"
prepare_base_split val "${VAL_DATA_ROOT}"

if [[ "${RESET_OUTPUTS}" == 1 ]]; then
  safe_clear_dir "${FLAT_ROOT}" "/media/SSD2/XCX/exp_2/watergan_step1564_official_mat_"
  safe_clear_dir "${FINAL_ROOT}" "/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated_step1564_official_mat"
  mkdir -p "${FLAT_ROOT}" "${FINAL_ROOT}" "${RESTORE_ROOT}"
fi

run_inference() {
  local split="$1" shard_index="$2" gpu="$3" limit="${4:-0}"
  local tag alias base_shard mat_shard results expected alias_model log
  local -a materialize_args
  if [[ "${limit}" -gt 0 ]]; then
    tag="smoke_${limit}"
    alias="${DATA_NAME}_${tag}"
    base_shard="${BASE_SHARD_ROOT}/train/shard0of${NUM_SHARDS}"
    mat_shard="${MAT_SHARD_ROOT}/${tag}"
    results="${FLAT_ROOT}/${tag}"
    expected="${limit}"
    log="${LOG_ROOT}/smoke_gpu${gpu}.log"
  else
    tag="shard${shard_index}of${NUM_SHARDS}"
    alias="${DATA_NAME}_${split}_${tag}_official_mat"
    base_shard="${BASE_SHARD_ROOT}/${split}/${tag}"
    mat_shard="${MAT_SHARD_ROOT}/${split}/${tag}"
    results="${FLAT_ROOT}/${split}/${tag}"
    expected="$(wc -l < "${base_shard}/watergan_air_manifest.jsonl" | tr -d ' ')"
    log="${LOG_ROOT}/${split}/generate_${tag}_gpu${gpu}.log"
  fi
  mkdir -p "$(dirname "${log}")" "${results}"
  if [[ "$(count_fake "${results}")" -eq "${expected}" ]]; then
    echo "reuse ${split}/${tag}: ${expected} outputs"
    return 0
  fi
  rm -f "${results}"/fake_*.png "${results}"/air_*.png "${results}"/depth_*.mat
  safe_clear_dir "${mat_shard}" "${MAT_SHARD_ROOT}/"
  materialize_args=(
    --source-shard "${base_shard}"
    --out-dir "${mat_shard}"
    --workers "${MAT_WORKERS_PER_PROCESS}"
    --reset
  )
  [[ "${limit}" -gt 0 ]] && materialize_args+=(--limit "${limit}")
  python tools/materialize_watergan_official_mat_shard.py "${materialize_args[@]}"

  alias_model="${CHECKPOINT_ROOT}/${alias}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
  [[ ! -e "${alias_model}" || -L "${alias_model}" ]] || {
    echo "Error: checkpoint alias is not a symlink: ${alias_model}" >&2; return 1;
  }
  ln -sfn "${MODEL_DIR}" "${alias_model}"
  ln -sfn "${mat_shard}/air_images" "${WATERGAN_DIR}/data/${alias}_air_images"
  ln -sfn "${mat_shard}/air_depth" "${WATERGAN_DIR}/data/${alias}_air_depth"
  ln -sfn "${base_shard}/water_images" "${WATERGAN_DIR}/data/${alias}_water_images"

  (
    cd "${WATERGAN_DIR}"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
      TF_FORCE_GPU_ALLOW_GROWTH=true WATERGAN_SAVE_AUX_OUTPUTS=0 \
      WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" PYTHONUNBUFFERED=1 \
      python mainmhl.py \
        --is_train=False \
        --water_dataset "${alias}_water_images" \
        --air_dataset "${alias}_air_images" \
        --depth_dataset "${alias}_air_depth" \
        --checkpoint_dir "${CHECKPOINT_ROOT}" \
        --sample_dir "samples_${alias}" \
        --results_dir "${results}" \
        --epoch 1 --num_samples "${expected}" --train_size "${expected}" \
        --batch_size "${BATCH_SIZE}" \
        --input_height 480 --input_width 640 \
        --input_water_height 1024 --input_water_width 1360 \
        --output_height "${OUTPUT_HEIGHT}" --output_width "${OUTPUT_WIDTH}"
  ) > "${log}" 2>&1 || {
    echo "FAILED ${split}/${tag} gpu=${gpu}; log=${log}" >&2
    [[ "${KEEP_FAILED_MAT}" == 1 ]] || safe_clear_dir "${mat_shard}" "${MAT_SHARD_ROOT}/"
    return 1
  }
  local actual
  actual="$(count_fake "${results}")"
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "Error: ${split}/${tag} generated ${actual}/${expected}; log=${log}" >&2
    return 1
  }
  safe_clear_dir "${mat_shard}" "${MAT_SHARD_ROOT}/"
  echo "finished ${split}/${tag} gpu=${gpu}: ${actual}/${expected}"
}

if [[ "${RUN_SMOKE}" == 1 ]]; then
  echo "Warning: embedded smoke mode is retained for compatibility."
  echo "Prefer running smoke_test_watergan_step1564_official_mat.sh first."
  echo "===== Official-MAT smoke test: 64 images on GPU ${GPU_LIST[0]} ====="
  run_inference smoke 0 "${GPU_LIST[0]}" 64
  echo "Smoke test passed. Starting full generation."
fi

run_split() {
  local split="$1" expected_total="$2" slot pid failed wait_index
  local failure_marker="${LOG_ROOT}/${split}/generation_failed.marker"
  local pids=() labels=()
  mkdir -p "${LOG_ROOT}/${split}"
  rm -f "${failure_marker}"

  run_worker_slot() {
    local worker_slot="$1" gpu index
    gpu="${GPU_LIST[$((worker_slot % NUM_GPUS))]}"
    for ((index=worker_slot; index<NUM_SHARDS; index+=MAX_CONCURRENT)); do
      if [[ -e "${failure_marker}" ]]; then
        echo "slot${worker_slot}: stop before shard${index} because another slot failed"
        return 1
      fi
      echo "dispatch ${split}/shard${index}of${NUM_SHARDS}:gpu${gpu}:slot${worker_slot}"
      if run_inference "${split}" "${index}" "${gpu}" 0; then
        echo "completed ${split}/shard${index}of${NUM_SHARDS}:gpu${gpu}:slot${worker_slot}"
      else
        printf 'split=%s slot=%s gpu=%s shard=%s\n' \
          "${split}" "${worker_slot}" "${gpu}" "${index}" > "${failure_marker}"
        echo "FAILED ${split}/shard${index}of${NUM_SHARDS}:gpu${gpu}:slot${worker_slot}" >&2
        return 1
      fi
    done
  }

  echo "===== Generate ${split}: ${MAX_CONCURRENT} persistent worker slots ====="
  for ((slot=0; slot<MAX_CONCURRENT && slot<NUM_SHARDS; slot++)); do
    run_worker_slot "${slot}" &
    pid="$!"
    pids+=("${pid}")
    labels+=("${split}:slot${slot}:gpu${GPU_LIST[$((slot % NUM_GPUS))]}")
    echo "started worker ${labels[-1]} pid=${pid}"
  done

  failed=0
  for wait_index in "${!pids[@]}"; do
    if wait "${pids[$wait_index]}"; then
      echo "finished worker ${labels[$wait_index]}"
    else
      echo "FAILED worker ${labels[$wait_index]}" >&2
      failed=1
    fi
  done
  [[ "${failed}" == 0 && ! -e "${failure_marker}" ]] || {
    echo "Error: one or more ${split} worker slots failed" >&2
    [[ -s "${failure_marker}" ]] && cat "${failure_marker}" >&2
    return 1
  }

  local jobs="${LOG_ROOT}/${split}/restore_jobs.tsv" shard manifest results out log
  mkdir -p "${LOG_ROOT}/${split}" "${RESTORE_ROOT}/${split}"
  : > "${jobs}"
  for ((index=0; index<NUM_SHARDS; index++)); do
    shard="shard${index}of${NUM_SHARDS}"
    manifest="${BASE_SHARD_ROOT}/${split}/${shard}/watergan_air_manifest.jsonl"
    results="${FLAT_ROOT}/${split}/${shard}"
    out="${RESTORE_ROOT}/${split}/${shard}"
    log="${LOG_ROOT}/${split}/restore_${shard}.log"
    [[ "$(count_fake "${results}")" -eq "$(wc -l < "${manifest}")" ]] || {
      echo "Error: incomplete flat output: ${results}" >&2; return 1;
    }
    printf '%s\t%s\t%s\t%s\n' "${manifest}" "${results}" "${out}" "${log}" >> "${jobs}"
  done
  export BATCH_SIZE
  xargs -P "${RESTORE_WORKERS}" -n 4 bash -c '
    python tools/restore_watergan_fake.py --manifest "$1" --results-dir "$2" \
      --out-dir "$3" --batch-size "${BATCH_SIZE}" --overwrite > "$4" 2>&1
  ' _ < "${jobs}"

  mkdir -p "${FINAL_ROOT}/${split}"
  for ((index=0; index<NUM_SHARDS; index++)); do
    cp -al "${RESTORE_ROOT}/${split}/shard${index}of${NUM_SHARDS}/." "${FINAL_ROOT}/${split}/"
  done
  rm -f "${FINAL_ROOT}/${split}/restore_watergan_fake_summary.json"
  local final_count classes
  final_count="$(count_images "${FINAL_ROOT}/${split}")"
  classes="$(find "${FINAL_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  [[ "${final_count}" -eq "${expected_total}" && "${classes}" -eq 1000 ]] || {
    echo "Error: restored ${split}: ${final_count}/${expected_total}, classes=${classes}/1000" >&2
    return 1
  }
  echo "restored ${split}: ${final_count}/${expected_total}, classes=${classes}/1000"
}

run_split train 250000
run_split val 10000

cat <<EOF
============================================================
WaterGAN official-MAT generation complete
============================================================
train: $(count_images "${FINAL_ROOT}/train")/250000
val:   $(count_images "${FINAL_ROOT}/val")/10000
root:  ${FINAL_ROOT}
============================================================
EOF
