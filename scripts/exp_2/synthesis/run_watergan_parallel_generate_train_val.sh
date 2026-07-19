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
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
GENERATE_EPOCHS="${GENERATE_EPOCHS:-1}"
RESET_SHARDS="${RESET_SHARDS:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"
CLEAN_RESTORE_SHARDS="${CLEAN_RESTORE_SHARDS:-1}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-4}"
export WATERGAN_IO_WORKERS WATERGAN_SAVE_AUX_OUTPUTS=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS="${#GPU_LIST[@]}"
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

[[ "${NUM_SHARDS}" -gt 0 ]] || { echo 'Error: GPUS is empty' >&2; exit 1; }
[[ -d "${TRAIN_MODEL_DIR}" ]] || { echo "Error: trained model dir not found: ${TRAIN_MODEL_DIR}" >&2; exit 1; }
grep -q 'model_checkpoint_path: "DCGAN.model-156252"' "${TRAIN_MODEL_DIR}/checkpoint" || {
  echo "Error: checkpoint state does not point to DCGAN.model-156252" >&2; exit 1;
}
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
CHECKPOINT:       ${TRAIN_MODEL_DIR}/DCGAN.model-156252
TRAIN_DATA_ROOT:  ${TRAIN_DATA_ROOT}
VAL_DATA_ROOT:    ${VAL_DATA_ROOT}
SHARD_DATA_ROOT:  ${SHARD_DATA_ROOT}
FLAT_ROOT:        ${FLAT_ROOT}
FINAL_ROOT:       ${FINAL_ROOT}
BATCH_SIZE:       ${BATCH_SIZE}
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
  [[ "${RESET_SHARDS}" == 1 ]] && shard_args+=(--reset)
  python tools/prepare_watergan_inference_shards.py "${shard_args[@]}" | tee "${split_log}/prepare_shards.log"

  if [[ "${RESET_OUTPUTS}" == 1 ]]; then
    rm -rf "${FLAT_ROOT:?}/${split}" "${RESTORE_SHARD_ROOT:?}/${split}" "${FINAL_ROOT:?}/${split}"
  fi
  mkdir -p "${FLAT_ROOT}/${split}" "${RESTORE_SHARD_ROOT}/${split}"

  local pids=() names=() index gpu alias shard_data shard_count alias_model results_dir pid
  for index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$index]}"
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
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then echo "finished ${names[$index]}"; else echo "FAILED ${names[$index]}" >&2; failed=1; fi
  done
  [[ "${failed}" == 0 ]] || return 1

  local restore_pids=()
  for index in "${!GPU_LIST[@]}"; do
    shard_data="${shard_root}/shard${index}of${NUM_SHARDS}"
    shard_count="$(wc -l < "${shard_data}/watergan_air_manifest.jsonl")"
    results_dir="${FLAT_ROOT}/${split}/shard${index}of${NUM_SHARDS}"
    [[ "$(find "${results_dir}" -maxdepth 1 -type f -name 'fake_*.png' | wc -l)" -eq "${shard_count}" ]] || {
      echo "Error: generated count mismatch for ${results_dir}" >&2; return 1;
    }
    python tools/restore_watergan_fake.py \
      --manifest "${shard_data}/watergan_air_manifest.jsonl" \
      --results-dir "${results_dir}" \
      --out-dir "${RESTORE_SHARD_ROOT}/${split}/shard${index}of${NUM_SHARDS}" \
      --batch-size "${BATCH_SIZE}" --overwrite \
      > "${split_log}/restore_shard${index}of${NUM_SHARDS}.log" 2>&1 &
    restore_pids+=("$!")
  done
  for index in "${!restore_pids[@]}"; do wait "${restore_pids[$index]}"; done

  mkdir -p "${FINAL_ROOT}/${split}"
  for index in "${!GPU_LIST[@]}"; do
    cp -al "${RESTORE_SHARD_ROOT}/${split}/shard${index}of${NUM_SHARDS}/." "${FINAL_ROOT}/${split}/"
  done
  rm -f "${FINAL_ROOT}/${split}/restore_watergan_fake_summary.json"
  local final_count
  final_count="$(count_images "${FINAL_ROOT}/${split}")"
  [[ "${final_count}" -eq "${expected}" ]] || {
    echo "Error: final ${split} count=${final_count}, expected=${expected}" >&2; return 1;
  }
  echo "${split}: ${final_count}/${expected} complete -> ${FINAL_ROOT}/${split}"
  if [[ "${CLEAN_RESTORE_SHARDS}" == 1 ]]; then
    rm -rf "${RESTORE_SHARD_ROOT:?}/${split}"
  fi
}

for split in ${SPLITS}; do run_split "${split}"; done

echo '============================================================'
echo 'WaterGAN parallel generation complete'
for split in ${SPLITS}; do echo "${split}: $(count_images "${FINAL_ROOT}/${split}")"; done
echo '============================================================'
