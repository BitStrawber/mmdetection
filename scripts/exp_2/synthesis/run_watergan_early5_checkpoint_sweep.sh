#!/usr/bin/env bash
set -euo pipefail

# Train one five-epoch WaterGAN trajectory on a single GPU, retain every early
# checkpoint, and compare the five epoch checkpoints on the same fixed images.
# The original TF1 model is single-GPU; GPUS parallelize checkpoint inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_mat_ssd}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"

TRAIN_GPU="${TRAIN_GPU:-0}"
INFER_GPUS="${INFER_GPUS:-0 1 2 3}"
EPOCHS="${EPOCHS:-5}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"
DISPLAY_EPOCH_START="${DISPLAY_EPOCH_START:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
NUM_COMPARE="${NUM_COMPARE:-40}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS:-8}"
WATERGAN_LOG_EVERY="${WATERGAN_LOG_EVERY:-100}"
WATERGAN_MAX_TO_KEEP="${WATERGAN_MAX_TO_KEEP:-20}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RESET_TRAIN="${RESET_TRAIN:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"

RUN_NAME="${RUN_NAME:-imagenet_ruod_watergan_balanced50_mat_early5_gpu${TRAIN_GPU}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${RUN_NAME}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${RUN_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-/media/SSD2/XCX/exp_2/watergan_early5_checkpoint_sweep}"
COMPARE_CKPT_ROOT="${COMPARE_CKPT_ROOT:-${WATERGAN_DIR}/checkpoint_compare_${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/synthesis_full/${RUN_NAME}}"

TRAIN_CHECKPOINT_ROOT="${WATERGAN_DIR}/${CHECKPOINT_DIR}"
TRAIN_MODEL_DIR="${TRAIN_CHECKPOINT_ROOT}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"

count_entries() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' '
}

require_path() {
  [[ -e "$1" ]] || { echo "Error: required path not found: $1" >&2; exit 1; }
}

read -r -a GPU_LIST <<< "${INFER_GPUS}"
[[ "${#GPU_LIST[@]}" -gt 0 ]] || { echo "Error: INFER_GPUS is empty" >&2; exit 1; }
(( NUM_COMPARE > 0 && NUM_COMPARE % BATCH_SIZE == 0 )) || {
  echo "Error: NUM_COMPARE must be positive and divisible by BATCH_SIZE" >&2
  exit 1
}

require_path "${WATERGAN_DIR}/mainmhl.py"
for name in air_images air_depth water_images; do
  require_path "${DATA_ROOT}/${name}"
done

WATERGAN_DIR="${WATERGAN_DIR}" \
  bash scripts/exp_2/synthesis/patch_watergan_gpu_selection.sh

cat <<EOF
============================================================
WaterGAN early-five checkpoint sweep
============================================================
DATA_ROOT:          ${DATA_ROOT}
TRAIN_GPU:          ${TRAIN_GPU}
INFER_GPUS:         ${INFER_GPUS}
EPOCHS:             ${EPOCHS}
CHECKPOINT_STEPS:   ${CHECKPOINT_STEPS:-auto}
DISPLAY_EPOCH_START:${DISPLAY_EPOCH_START}
BATCH_SIZE:         ${BATCH_SIZE}
TRAIN_SIZE:         ${TRAIN_SIZE}
NUM_COMPARE:        ${NUM_COMPARE}
MAX_TO_KEEP:        ${WATERGAN_MAX_TO_KEEP}
CHECKPOINT_ROOT:    ${TRAIN_CHECKPOINT_ROOT}
RESULT_ROOT:        ${RESULT_ROOT}
LOG_ROOT:           ${LOG_ROOT}
RUN_TRAIN:          ${RUN_TRAIN}
RUN_INFER:          ${RUN_INFER}
RESET_TRAIN:        ${RESET_TRAIN}
RESET_OUTPUTS:      ${RESET_OUTPUTS}
============================================================
EOF

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" "${WATERGAN_DIR}/data"

if [[ "${RUN_TRAIN}" == 1 ]]; then
  if [[ -e "${TRAIN_CHECKPOINT_ROOT}" ]]; then
    if [[ "${RESET_TRAIN}" == 1 ]]; then
      case "${TRAIN_CHECKPOINT_ROOT}" in
        "${WATERGAN_DIR}"/checkpoint_*) rm -rf "${TRAIN_CHECKPOINT_ROOT}" ;;
        *) echo "Error: refusing to reset unexpected path: ${TRAIN_CHECKPOINT_ROOT}" >&2; exit 1 ;;
      esac
    else
      echo "Error: training checkpoint root already exists: ${TRAIN_CHECKPOINT_ROOT}" >&2
      echo "Set RESET_TRAIN=1 only when intentionally restarting this early-five run." >&2
      exit 1
    fi
  fi

  WATERGAN_DIR="${WATERGAN_DIR}" \
    bash scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh

  grep -q "WATERGAN_MAX_TO_KEEP" "${WATERGAN_DIR}/modelmhl.py" || {
    echo "Error: configurable Saver retention was not patched into modelmhl.py" >&2
    exit 1
  }

  DATA_NAME="${DATA_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  SYN_ROOT="${SYN_ROOT}" \
  WATERGAN_DIR="${WATERGAN_DIR}" \
  GPU="${TRAIN_GPU}" \
  EPOCH="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TRAIN_SIZE="${TRAIN_SIZE}" \
  SAVE_EPOCH=1 \
  AUTO_PATCH=0 \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  SAMPLE_DIR="${SAMPLE_DIR}" \
  RESULTS_DIR="${RESULT_ROOT}/training_results" \
  LOG_DIR="${LOG_ROOT}" \
  WATERGAN_IO_WORKERS="${WATERGAN_IO_WORKERS}" \
  WATERGAN_LOG_EVERY="${WATERGAN_LOG_EVERY}" \
  WATERGAN_MAX_TO_KEEP="${WATERGAN_MAX_TO_KEEP}" \
  PYTHONUNBUFFERED=1 \
    bash scripts/exp_2/synthesis/run_watergan_train.sh \
    2>&1 | tee "${LOG_ROOT}/train_pipeline.log"
fi

require_path "${TRAIN_MODEL_DIR}"

if [[ -n "${CHECKPOINT_STEPS}" ]]; then
  read -r -a STEPS <<< "${CHECKPOINT_STEPS}"
  LABEL_MODE=step
  for step in "${STEPS[@]}"; do
    [[ "${step}" =~ ^[0-9]+$ ]] || {
      echo "Error: invalid checkpoint step: ${step}" >&2
      exit 1
    }
  done
else
  LABEL_MODE=epoch
  mapfile -t ALL_STEPS < <(
    find "${TRAIN_MODEL_DIR}" -maxdepth 1 \( -type f -o -type l \) \
      -name 'DCGAN.model-*.index' -printf '%f\n' |
      sed -E 's/^DCGAN\.model-([0-9]+)\.index$/\1/' |
      sort -n
  )

  if (( ${#ALL_STEPS[@]} < EPOCHS )); then
    echo "Error: found only ${#ALL_STEPS[@]} checkpoints; expected at least ${EPOCHS}" >&2
    printf '  %s\n' "${ALL_STEPS[@]}" >&2
    exit 1
  fi

  # The original loop also saves model-2 immediately. The final EPOCHS
  # entries are the completed epoch checkpoints when SAVE_EPOCH=1.
  STEPS=("${ALL_STEPS[@]: -EPOCHS}")
fi
printf '%s\n' "${STEPS[@]}" > "${RESULT_ROOT}/selected_checkpoint_steps.txt"

echo
echo "Selected completed-epoch checkpoints: ${STEPS[*]}"

if [[ "${RUN_INFER}" != 1 ]]; then
  echo "RUN_INFER=${RUN_INFER}; training/checkpoint selection complete."
  exit 0
fi

ln -sfn "${DATA_ROOT}/air_images" "${WATERGAN_DIR}/data/${DATA_NAME}_air_images"
ln -sfn "${DATA_ROOT}/air_depth" "${WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" "${WATERGAN_DIR}/data/${DATA_NAME}_water_images"

export WATERGAN_IO_WORKERS=4
export WATERGAN_SAVE_AUX_OUTPUTS=0
export WATERGAN_MAX_TO_KEEP
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

if [[ "${RESET_OUTPUTS}" == 1 ]]; then
  case "${RESULT_ROOT}" in
    /media/SSD2/XCX/exp_2/watergan_early5_checkpoint_sweep*)
      rm -rf "${RESULT_ROOT}/generated" "${RESULT_ROOT}/panels"
      ;;
    *) echo "Error: refusing to reset unexpected result root: ${RESULT_ROOT}" >&2; exit 1 ;;
  esac
fi
mkdir -p "${RESULT_ROOT}/generated" "${RESULT_ROOT}/panels" "${COMPARE_CKPT_ROOT}"

for STEP in "${STEPS[@]}"; do
  TEMP_ROOT="${COMPARE_CKPT_ROOT}/step_${STEP}"
  TEMP_MODEL="${TEMP_ROOT}/${DATA_NAME}_water_images_${BATCH_SIZE}_${OUTPUT_HEIGHT}_${OUTPUT_WIDTH}"
  mkdir -p "${TEMP_MODEL}"
  for suffix in index meta data-00000-of-00001; do
    require_path "${TRAIN_MODEL_DIR}/DCGAN.model-${STEP}.${suffix}"
    ln -sfn "${TRAIN_MODEL_DIR}/DCGAN.model-${STEP}.${suffix}" \
      "${TEMP_MODEL}/DCGAN.model-${STEP}.${suffix}"
  done
  printf '%s\n' \
    "model_checkpoint_path: \"DCGAN.model-${STEP}\"" \
    "all_model_checkpoint_paths: \"DCGAN.model-${STEP}\"" \
    > "${TEMP_MODEL}/checkpoint"
done

run_wave() {
  local start="$1" pids=() labels=() local_index index step gpu temp_root output log pid
  for local_index in "${!GPU_LIST[@]}"; do
    index=$((start + local_index))
    (( index < ${#STEPS[@]} )) || break
    step="${STEPS[$index]}"
    gpu="${GPU_LIST[$local_index]}"
    temp_root="${COMPARE_CKPT_ROOT}/step_${step}"
    output="${RESULT_ROOT}/generated/step_${step}"
    log="${LOG_ROOT}/infer_step_${step}_gpu${gpu}.log"
    mkdir -p "${output}"
    (
      cd "${WATERGAN_DIR}"
      env -u LD_LIBRARY_PATH -u LD_PRELOAD \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        python mainmhl.py \
        --is_train=False \
        --water_dataset "${DATA_NAME}_water_images" \
        --air_dataset "${DATA_NAME}_air_images" \
        --depth_dataset "${DATA_NAME}_air_depth" \
        --checkpoint_dir "${temp_root}" \
        --sample_dir "samples_${RUN_NAME}_step_${step}" \
        --results_dir "${output}" \
        --epoch 1 --num_samples "${NUM_COMPARE}" --train_size "${NUM_COMPARE}" \
        --batch_size "${BATCH_SIZE}" \
        --input_height 480 --input_width 640 \
        --input_water_height 1024 --input_water_width 1360 \
        --output_height "${OUTPUT_HEIGHT}" --output_width "${OUTPUT_WIDTH}"
    ) > "${log}" 2>&1 &
    pid="$!"
    pids+=("${pid}")
    labels+=("step=${step}:gpu=${gpu}")
    echo "started ${labels[-1]} pid=${pid}"
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "finished ${labels[$index]}"
    else
      echo "FAILED ${labels[$index]}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

for ((start=0; start<${#STEPS[@]}; start+=${#GPU_LIST[@]})); do
  run_wave "${start}"
done

for STEP in "${STEPS[@]}"; do
  count="$(find "${RESULT_ROOT}/generated/step_${STEP}" -maxdepth 1 -type f -name 'fake_*.png' | wc -l | tr -d ' ')"
  [[ "${count}" -eq "${NUM_COMPARE}" ]] || {
    echo "Error: step ${STEP} generated ${count}/${NUM_COMPARE}" >&2
    exit 1
  }
done

python - "${DATA_ROOT}/air_images" "${RESULT_ROOT}" "${NUM_COMPARE}" \
  "${DISPLAY_EPOCH_START}" "${LABEL_MODE}" "${STEPS[@]}" <<'PY'
from __future__ import print_function

import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

source_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
expected = int(sys.argv[3])
epoch_start = int(sys.argv[4])
label_mode = sys.argv[5]
steps = sys.argv[6:]
panel_root = result_root / 'panels'
panel_root.mkdir(parents=True, exist_ok=True)

resampling = getattr(Image, 'Resampling', Image)
font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()

def fake_key(path):
    match = re.fullmatch(r'fake_(\d+)_(\d+)_(\d+)\.png', path.name)
    if not match:
        raise RuntimeError('Unexpected result filename: {}'.format(path))
    epoch, within, batch = map(int, match.groups())
    return epoch, batch, within

sources = sorted(source_root.glob('*.png'))[:expected]
groups = []
for step in steps:
    files = sorted((result_root / 'generated' / ('step_' + step)).glob('fake_*.png'), key=fake_key)
    if len(files) != expected:
        raise RuntimeError('step {} has {}/{} images'.format(step, len(files), expected))
    groups.append(files)

tile_size = (320, 240)
header = 38
if label_mode == 'step':
    labels = ['Original'] + ['step {}'.format(step) for step in steps]
else:
    labels = ['Original'] + [
        'epoch {} / step {}'.format(epoch_start + i, step)
        for i, step in enumerate(steps)
    ]
manifest = ['index\tsource\t' + '\t'.join('step_' + step for step in steps) + '\tpanel']

for index in range(expected):
    paths = [sources[index]] + [group[index] for group in groups]
    canvas = Image.new('RGB', (tile_size[0] * len(paths), tile_size[1] + header), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (label, path) in enumerate(zip(labels, paths)):
        with Image.open(str(path)) as image:
            tile = ImageOps.fit(image.convert('RGB'), tile_size, method=resampling.BICUBIC)
        x = column * tile_size[0]
        canvas.paste(tile, (x, header))
        draw.text((x + 8, 10), label, fill='black', font=font)
        if column:
            draw.line([(x, 0), (x, tile_size[1] + header)], fill=(100, 100, 100), width=1)
    panel = panel_root / '{:03d}_early5_sweep.jpg'.format(index)
    canvas.save(str(panel), quality=95, subsampling=0)
    manifest.append('{}\t{}\t{}\t{}'.format(
        index, sources[index], '\t'.join(str(group[index]) for group in groups), panel
    ))

(result_root / 'comparison_manifest.tsv').write_text('\n'.join(manifest) + '\n', encoding='utf-8')
print('steps:', steps)
print('panels:', len(list(panel_root.glob('*.jpg'))))
print('result:', result_root)
PY

echo
echo "============================================================"
echo "WaterGAN early-five checkpoint sweep complete"
echo "============================================================"
echo "Selected steps: ${STEPS[*]}"
echo "Panels:         ${RESULT_ROOT}/panels"
echo "Manifest:       ${RESULT_ROOT}/comparison_manifest.tsv"
echo "Logs:           ${LOG_ROOT}"
du -sh "${RESULT_ROOT}"
