#!/usr/bin/env bash
set -euo pipefail

# Run independent WaterGAN inference shards on multiple GPUs, then restore and
# merge them into the standard ImageNet class directory layout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
TRAIN_DATA_NAME="${TRAIN_DATA_NAME:-imagenet_ruod_watergan_train_balanced50_mat_ssd}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_imagenet_ruod_watergan_train_balanced50_mat_ssd_bs8_e26_nv_gpu0}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_train_full250k_ssd}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/datasets/imagenet_ruod_watergan_val_full10k_infer_ssd}"
SHARD_DATA_ROOT="${SHARD_DATA_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_shards}"
FLAT_ROOT="${FLAT_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_parallel}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated}"
RESTORE_SHARD_ROOT="${RESTORE_SHARD_ROOT:-${FINAL_ROOT}/.parallel_shards}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_parallel_infer}"

GPUS="${GPUS:-0 1 6 7}"
SPLITS="${SPLITS:-train val}"
NUM_SHARDS="${NUM_SHARDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
GENERATE_EPOCHS="${GENERATE_EPOCHS:-1}"
RESET_SHARDS="${RESET_SHARDS:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"
CLEAN_RESTORE_SHARDS="${CLEAN_RESTORE_SHARDS:-1}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-4}"
RESTORE_WORKERS="${RESTORE_WORKERS:-8}"
GENERATE_WORKERS="${GENERATE_WORKERS:-0}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-156252}"
PAD_TO_BATCH="${PAD_TO_BATCH:-0}"
export WATERGAN_IO_WORKERS WATERGAN_SAVE_AUX_OUTPUTS=0 PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_SHARDS}" -eq 0 ]]; then
  NUM_SHARDS="${NUM_GPUS}"
fi
if [[ "${GENERATE_WORKERS}" -eq 0 || "${GENERATE_WORKERS}" -gt "${NUM_SHARDS}" ]]; then
  GENERATE_WORKERS="${NUM_SHARDS}"
fi
CHECKPOINT_ROOT="${WATERGAN_DIR}/${CHECKPOINT_DIR}"
TRAIN_MODEL_DIR="${CHECKPOINT_ROOT}/${TRAIN_DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' '
}

require_count() {
  local root="$1" expected="$2" label="$3"
  local manifest_count
  [[ -d "${root}/air_images" && -d "${root}/air_depth" && -d "${root}/water_images" ]] || {
    echo "Error: incomplete ${label} data root: ${root}" >&2; exit 1;
  }
  manifest_count="$(wc -l < "${root}/watergan_air_manifest.jsonl")"
  [[ "${manifest_count}" -eq "${expected}" ]] || {
    echo "Error: ${label} manifest=${manifest_count}, expected=${expected}" >&2; exit 1;
  }
}

[[ "${NUM_GPUS}" -gt 0 ]] || { echo 'Error: GPUS is empty' >&2; exit 1; }
[[ "${NUM_SHARDS}" -gt 0 ]] || { echo 'Error: NUM_SHARDS must be positive' >&2; exit 1; }
[[ -d "${TRAIN_MODEL_DIR}" ]] || { echo "Error: trained model dir not found: ${TRAIN_MODEL_DIR}" >&2; exit 1; }
grep -q "model_checkpoint_path: \"DCGAN.model-${CHECKPOINT_STEP}\"" "${TRAIN_MODEL_DIR}/checkpoint" || {
  echo "Error: checkpoint state does not point to DCGAN.model-${CHECKPOINT_STEP}" >&2; exit 1;
}
for suffix in index meta data-00000-of-00001; do
  [[ -s "${TRAIN_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}" ]] || {
    echo "Error: checkpoint file missing: ${TRAIN_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}.${suffix}" >&2
    exit 1
  }
done
grep -q 'WATERGAN_SAVE_AUX_OUTPUTS' "${WATERGAN_DIR}/modelmhl.py" || {
  echo 'Error: auxiliary-output guard is missing; rerun patch_watergan_tf15_compat.sh' >&2; exit 1;
}
if [[ " ${SPLITS} " == *" train "* ]]; then
  require_count "${TRAIN_DATA_ROOT}" 250000 train
fi
if [[ " ${SPLITS} " == *" val "* ]]; then
  require_count "${VAL_DATA_ROOT}" 10000 val
fi

cat <<EOF
============================================================
WaterGAN parallel train+val inference
============================================================
GPUS:             ${GPUS}
SPLITS:           ${SPLITS}
CHECKPOINT:       ${TRAIN_MODEL_DIR}/DCGAN.model-${CHECKPOINT_STEP}
TRAIN_DATA_ROOT:  ${TRAIN_DATA_ROOT}
VAL_DATA_ROOT:    ${VAL_DATA_ROOT}
SHARD_DATA_ROOT:  ${SHARD_DATA_ROOT}
FLAT_ROOT:        ${FLAT_ROOT}
FINAL_ROOT:       ${FINAL_ROOT}
BATCH_SIZE:       ${BATCH_SIZE}
NUM_SHARDS:       ${NUM_SHARDS} (${NUM_GPUS} GPUs)
CHECKPOINT_STEP:  ${CHECKPOINT_STEP}
RESTORE_WORKERS:  ${RESTORE_WORKERS}
GENERATE_WORKERS: ${GENERATE_WORKERS}
PAD_TO_BATCH:     ${PAD_TO_BATCH}
RESET_SHARDS:     ${RESET_SHARDS}
RESET_OUTPUTS:    ${RESET_OUTPUTS}
============================================================
EOF

mkdir -p "${WATERGAN_DIR}/data" "${LOG_ROOT}" "${SHARD_DATA_ROOT}" "${FLAT_ROOT}" "${RESTORE_SHARD_ROOT}"

run_split() {
  local split="$1" data_root expected shard_root split_log
  if [[ "${split}" == train ]]; then
    data_root="${TRAIN_DATA_ROOT}"; expected=250000
  elif [[ "${split}" == val ]]; then
    data_root="${VAL_DATA_ROOT}"; expected=10000
  else
    echo "Error: unsupported split: ${split}" >&2; return 1
  fi
  shard_root="${SHARD_DATA_ROOT}/${split}"
  split_log="${LOG_ROOT}/${split}"
  mkdir -p "${split_log}"

  local -a shard_args=(--data-root "${data_root}" --out-root "${shard_root}" --num-shards "${NUM_SHARDS}" --batch-size "${BATCH_SIZE}")
  [[ "${PAD_TO_BATCH}" == 1 ]] && shard_args+=(--pad-to-batch)
  [[ "${RESET_SHARDS}" == 1 ]] && shard_args+=(--reset)
  python tools/prepare_watergan_inference_shards.py "${shard_args[@]}" | tee "${split_log}/prepare_shards.log"

  if [[ "${RESET_OUTPUTS}" == 1 ]]; then
    rm -rf "${FLAT_ROOT:?}/${split}" "${RESTORE_SHARD_ROOT:?}/${split}" "${FINAL_ROOT:?}/${split}"
  fi
  mkdir -p "${FLAT_ROOT}/${split}" "${RESTORE_SHARD_ROOT}/${split}"

  local pids=() names=() index gpu alias shard_data shard_count alias_model results_dir pid output_count
  local generation_failed=0 wait_index
  for ((index = 0; index < NUM_SHARDS; index++)); do
    gpu="${GPU_LIST[$((index % NUM_GPUS))]}"
    alias="${TRAIN_DATA_NAME}_${split}_shard${index}of${NUM_SHARDS}"
    shard_data="${shard_root}/shard${index}of${NUM_SHARDS}"
    shard_count="$(wc -l < "${shard_data}/watergan_air_manifest.jsonl")"
    alias_model="${CHECKPOINT_ROOT}/${alias}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
    if [[ -e "${alias_model}" && ! -L "${alias_model}" ]]; then
      echo "Error: checkpoint alias exists and is not a symlink: ${alias_model}" >&2; return 1
    fi
    rm -f "${alias_model}"
    ln -s "${TRAIN_MODEL_DIR}" "${alias_model}"
    ln -sfn "${shard_data}/air_images" "${WATERGAN_DIR}/data/${alias}_air_images"
    ln -sfn "${shard_data}/air_depth" "${WATERGAN_DIR}/data/${alias}_air_depth"
    ln -sfn "${shard_data}/water_images" "${WATERGAN_DIR}/data/${alias}_water_images"
    results_dir="${FLAT_ROOT}/${split}/shard${index}of${NUM_SHARDS}"
    mkdir -p "${results_dir}"
    output_count="$(find "${results_dir}" -maxdepth 1 -type f -name 'fake_*.png' | wc -l | tr -d ' ')"
    if [[ "${output_count}" -eq "${shard_count}" ]]; then
      echo "reuse shard${index}of${NUM_SHARDS}:gpu${gpu}:${shard_count}"
      continue
    fi
    if [[ "${output_count}" -gt 0 ]]; then
      echo "reset incomplete shard${index}of${NUM_SHARDS}: ${output_count}/${shard_count}"
      rm -f "${results_dir}"/fake_*.png "${results_dir}"/air_*.png "${results_dir}"/depth_*.mat
    fi
    (
      cd "${WATERGAN_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu}" python mainmhl.py \
        --is_train=False \
        --water_dataset "${alias}_water_images" \
        --air_dataset "${alias}_air_images" \
        --depth_dataset "${alias}_air_depth" \
        --checkpoint_dir "${CHECKPOINT_ROOT}" \
        --sample_dir "samples_${alias}" \
        --results_dir "${results_dir}" \
        --epoch "${GENERATE_EPOCHS}" \
        --num_samples "${shard_count}" \
        --train_size "${shard_count}" \
        --batch_size "${BATCH_SIZE}" \
        --input_height 480 --input_width 640 \
        --input_water_height 1024 --input_water_width 1360 \
        --output_height "${OUTPUT_HEIGHT}" --output_width "${OUTPUT_WIDTH}"
    ) > "${split_log}/generate_shard${index}of${NUM_SHARDS}_gpu${gpu}.log" 2>&1 &
    pid="$!"
    pids+=("${pid}"); names+=("shard${index}of${NUM_SHARDS}:gpu${gpu}:${shard_count}")
    echo "started shard${index}of${NUM_SHARDS}:gpu${gpu}:${shard_count} pid=${pid}"
    if [[ "${#pids[@]}" -ge "${GENERATE_WORKERS}" ]]; then
      for wait_index in "${!pids[@]}"; do
        if wait "${pids[$wait_index]}"; then
          echo "finished ${names[$wait_index]}"
        else
          echo "FAILED ${names[$wait_index]}" >&2
          generation_failed=1
        fi
      done
      pids=(); names=()
    fi
  done

  for wait_index in "${!pids[@]}"; do
    if wait "${pids[$wait_index]}"; then
      echo "finished ${names[$wait_index]}"
    else
      echo "FAILED ${names[$wait_index]}" >&2
      generation_failed=1
    fi
  done
  [[ "${generation_failed}" == 0 ]] || return 1

  local restore_jobs="${split_log}/restore_jobs.tsv"
  : > "${restore_jobs}"
  for ((index = 0; index < NUM_SHARDS; index++)); do
    shard_data="${shard_root}/shard${index}of${NUM_SHARDS}"
    shard_count="$(wc -l < "${shard_data}/watergan_air_manifest.jsonl")"
    results_dir="${FLAT_ROOT}/${split}/shard${index}of${NUM_SHARDS}"
    [[ "$(find "${results_dir}" -maxdepth 1 -type f -name 'fake_*.png' | wc -l)" -eq "${shard_count}" ]] || {
      echo "Error: generated count mismatch for ${results_dir}" >&2; return 1;
    }
    printf '%s\t%s\t%s\t%s\n' \
      "${shard_data}/watergan_air_manifest.jsonl" \
      "${results_dir}" \
      "${RESTORE_SHARD_ROOT}/${split}/shard${index}of${NUM_SHARDS}" \
      "${split_log}/restore_shard${index}of${NUM_SHARDS}.log" \
      >> "${restore_jobs}"
  done
  export BATCH_SIZE
  xargs -P "${RESTORE_WORKERS}" -n 4 bash -c '
    python tools/restore_watergan_fake.py \
      --manifest "$1" --results-dir "$2" --out-dir "$3" \
      --batch-size "${BATCH_SIZE}" --overwrite > "$4" 2>&1
  ' _ < "${restore_jobs}"

  mkdir -p "${FINAL_ROOT}/${split}"
  for ((index = 0; index < NUM_SHARDS; index++)); do
    cp -al "${RESTORE_SHARD_ROOT}/${split}/shard${index}of${NUM_SHARDS}/." "${FINAL_ROOT}/${split}/"
  done
  rm -f "${FINAL_ROOT}/${split}/restore_watergan_fake_summary.json"
  local final_count final_classes
  final_count="$(count_images "${FINAL_ROOT}/${split}")"
  final_classes="$(find "${FINAL_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  [[ "${final_count}" -eq "${expected}" && "${final_classes}" -eq 1000 ]] || {
    echo "Error: final ${split}: images=${final_count}/${expected}, classes=${final_classes}/1000" >&2
    return 1
  }
  echo "${split}: ${final_count}/${expected}, classes=${final_classes}/1000 complete -> ${FINAL_ROOT}/${split}"
  if [[ "${CLEAN_RESTORE_SHARDS}" == 1 ]]; then
    rm -rf "${RESTORE_SHARD_ROOT:?}/${split}"
  fi
}

for split in ${SPLITS}; do run_split "${split}"; done

echo '============================================================'
echo 'WaterGAN parallel generation complete'
for split in ${SPLITS}; do echo "${split}: $(count_images "${FINAL_ROOT}/${split}")"; done
echo '============================================================'
