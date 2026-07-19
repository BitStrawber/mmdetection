#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
TRAIN_DATA_NAME="${TRAIN_DATA_NAME:-imagenet_ruod_watergan_train_balanced50_mat_ssd}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_imagenet_ruod_watergan_train_balanced50_mat_ssd_bs8_e26_nv_gpu0}"
BASE_SHARD_ROOT="${BASE_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_shards/train}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_parallel/train}"
RESUME_SHARD_ROOT="${RESUME_SHARD_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/watergan/inference_resume_shards/train}"
RESUME_RESULTS_ROOT="${RESUME_RESULTS_ROOT:-/media/SSD2/XCX/exp_2/watergan_flat_results_resume/train}"
FINAL_ROOT="${FINAL_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/generated}"
RESTORE_ROOT="${RESTORE_ROOT:-${FINAL_ROOT}/.resume_restore_train}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/watergan_resume_multigpu/train}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
BASE_SHARDS="${BASE_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
RESET_RESUME_SHARDS="${RESET_RESUME_SHARDS:-1}"
RESET_RESUME_RESULTS="${RESET_RESUME_RESULTS:-1}"
RESET_FINAL="${RESET_FINAL:-1}"
RUN_VAL_AFTER="${RUN_VAL_AFTER:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-2}"
export WATERGAN_IO_WORKERS WATERGAN_SAVE_AUX_OUTPUTS=0 PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

read -r -a GPU_LIST <<< "${GPUS}"
RESUME_SHARDS="${#GPU_LIST[@]}"
CHECKPOINT_ROOT="${WATERGAN_DIR}/${CHECKPOINT_DIR}"
TRAIN_MODEL_DIR="${CHECKPOINT_ROOT}/${TRAIN_DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"

if pgrep -f '[m]ainmhl.py.*train_shard' >/dev/null; then
  echo 'Error: old WaterGAN train shard processes are still running.' >&2
  echo 'Stop them before creating a stable resume plan.' >&2
  pgrep -af '[m]ainmhl.py.*train_shard' >&2
  exit 1
fi
[[ -f "${TRAIN_MODEL_DIR}/checkpoint" ]] || { echo "Missing checkpoint: ${TRAIN_MODEL_DIR}" >&2; exit 1; }
mkdir -p "${LOG_ROOT}" "${WATERGAN_DIR}/data"

prepare_args=(
  --base-shard-root "${BASE_SHARD_ROOT}"
  --base-results-root "${BASE_RESULTS_ROOT}"
  --out-root "${RESUME_SHARD_ROOT}"
  --base-shards "${BASE_SHARDS}"
  --resume-shards "${RESUME_SHARDS}"
  --batch-size "${BATCH_SIZE}"
)
[[ "${RESET_RESUME_SHARDS}" == 1 ]] && prepare_args+=(--reset)
python tools/prepare_watergan_resume_shards.py "${prepare_args[@]}" | tee "${LOG_ROOT}/resume_plan.log"

echo
echo "Resume plan: ${RESUME_SHARD_ROOT}/resume_plan.json"
python - "${RESUME_SHARD_ROOT}/resume_plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], 'r'))
print('completed_total={}'.format(plan['completed_total']))
print('pending_total={}'.format(plan['pending_total']))
for item in plan['base']:
    print('{name}: completed={completed}, pending={pending}'.format(**item))
print('resume_sizes={}'.format([item['count'] for item in plan['resume']]))
PY
if [[ "${PLAN_ONLY}" == 1 ]]; then
  echo 'PLAN_ONLY=1; inference was not started.'
  exit 0
fi

if [[ "${RESET_RESUME_RESULTS}" == 1 ]]; then
  rm -rf "${RESUME_RESULTS_ROOT}"
fi
mkdir -p "${RESUME_RESULTS_ROOT}"

pids=(); names=()
for index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$index]}"
  name="shard${index}of${RESUME_SHARDS}"
  shard="${RESUME_SHARD_ROOT}/${name}"
  count="$(wc -l < "${shard}/watergan_air_manifest.jsonl")"
  alias="${TRAIN_DATA_NAME}_resume_${name}"
  alias_model="${CHECKPOINT_ROOT}/${alias}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
  [[ ! -e "${alias_model}" || -L "${alias_model}" ]] || { echo "Invalid alias: ${alias_model}" >&2; exit 1; }
  rm -f "${alias_model}"
  ln -s "${TRAIN_MODEL_DIR}" "${alias_model}"
  ln -sfn "${shard}/air_images" "${WATERGAN_DIR}/data/${alias}_air_images"
  ln -sfn "${shard}/air_depth" "${WATERGAN_DIR}/data/${alias}_air_depth"
  ln -sfn "${shard}/water_images" "${WATERGAN_DIR}/data/${alias}_water_images"
  results="${RESUME_RESULTS_ROOT}/${name}"
  mkdir -p "${results}"
  (
    cd "${WATERGAN_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" python mainmhl.py \
      --is_train=False \
      --water_dataset "${alias}_water_images" \
      --air_dataset "${alias}_air_images" \
      --depth_dataset "${alias}_air_depth" \
      --checkpoint_dir "${CHECKPOINT_ROOT}" \
      --sample_dir "samples_${alias}" --results_dir "${results}" \
      --epoch 1 --num_samples "${count}" --train_size "${count}" \
      --batch_size "${BATCH_SIZE}" \
      --input_height 480 --input_width 640 \
      --input_water_height 1024 --input_water_width 1360 \
      --output_height "${OUTPUT_HEIGHT}" --output_width "${OUTPUT_WIDTH}"
  ) > "${LOG_ROOT}/generate_${name}_gpu${gpu}.log" 2>&1 &
  pid="$!"; pids+=("${pid}"); names+=("${name}:gpu${gpu}:${count}")
  echo "started ${name}:gpu${gpu}:${count} pid=${pid}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then echo "finished ${names[$index]}"; else echo "FAILED ${names[$index]}" >&2; failed=1; fi
done
[[ "${failed}" == 0 ]] || exit 1

if [[ "${RESET_FINAL}" == 1 ]]; then
  rm -rf "${RESTORE_ROOT}" "${FINAL_ROOT}/train"
fi
mkdir -p "${RESTORE_ROOT}/base" "${RESTORE_ROOT}/resume" "${FINAL_ROOT}/train"

restore_pids=()
for index in $(seq 0 $((BASE_SHARDS - 1))); do
  name="shard${index}of${BASE_SHARDS}"
  python tools/restore_watergan_fake.py \
    --manifest "${RESUME_SHARD_ROOT}/completed_manifests/${name}.jsonl" \
    --results-dir "${BASE_RESULTS_ROOT}/${name}" \
    --out-dir "${RESTORE_ROOT}/base/${name}" --batch-size "${BATCH_SIZE}" --overwrite \
    > "${LOG_ROOT}/restore_base_${name}.log" 2>&1 &
  restore_pids+=("$!")
done
for index in "${!GPU_LIST[@]}"; do
  name="shard${index}of${RESUME_SHARDS}"
  python tools/restore_watergan_fake.py \
    --manifest "${RESUME_SHARD_ROOT}/${name}/watergan_air_manifest.jsonl" \
    --results-dir "${RESUME_RESULTS_ROOT}/${name}" \
    --out-dir "${RESTORE_ROOT}/resume/${name}" --batch-size "${BATCH_SIZE}" --overwrite \
    > "${LOG_ROOT}/restore_resume_${name}.log" 2>&1 &
  restore_pids+=("$!")
done
for pid in "${restore_pids[@]}"; do wait "${pid}"; done

for root in "${RESTORE_ROOT}/base"/* "${RESTORE_ROOT}/resume"/*; do
  cp -al "${root}/." "${FINAL_ROOT}/train/"
done
rm -f "${FINAL_ROOT}/train/restore_watergan_fake_summary.json"
final_count="$(find "${FINAL_ROOT}/train" -type f -name '*.png' | wc -l)"
[[ "${final_count}" -eq 250000 ]] || { echo "Error: final train count=${final_count}" >&2; exit 1; }
echo "train: ${final_count}/250000 complete"
rm -rf "${RESTORE_ROOT}"

if [[ "${RUN_VAL_AFTER}" == 1 ]]; then
  GPUS="${GPUS}" SPLITS=val RESET_SHARDS=0 RESET_OUTPUTS=1 \
    WATERGAN_DIR="${WATERGAN_DIR}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    TRAIN_DATA_NAME="${TRAIN_DATA_NAME}" FINAL_ROOT="${FINAL_ROOT}" \
    bash scripts/exp_2/synthesis/run_watergan_parallel_generate_train_val.sh
fi
