#!/usr/bin/env bash
set -euo pipefail

# Reproduce the WaterGAN 50k preparation/training path used around 2026-07-14.
# Historical scripts are materialized from LEGACY_COMMIT instead of reusing the
# current optimized preparation and runtime code.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LEGACY_COMMIT="${LEGACY_COMMIT:-a9686c0b}"
WATERGAN_SOURCE_DIR="${WATERGAN_SOURCE_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"
LEGACY_WATERGAN_DIR="${LEGACY_WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"
LEGACY_WATERGAN_COMMIT="${LEGACY_WATERGAN_COMMIT:-auto}"

SOURCE_DIR="${SOURCE_DIR:-/media/SSD1/XCX/exp_2/synthetic_imagenet/watergan/source/train}"
DEPTH_DIR="${DEPTH_DIR:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/watergan/train}"
WATER_SOURCE="${WATER_SOURCE:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"

DATA_NAME="${DATA_NAME:-imagenet_ruod_watergan_train_balanced50_legacy_20260714}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/watergan/datasets/${DATA_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_full/watergan_legacy_20260714}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${LOG_DIR}/historical_snapshot}"

GPU="${GPU:-0}"
EPOCH="${EPOCH:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SIZE="${TRAIN_SIZE:-50000}"
AIR_PER_CLASS="${AIR_PER_CLASS:-50}"
WATER_REPEAT_TO="${WATER_REPEAT_TO:-50000}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
PREP_WORKERS="${PREP_WORKERS:-16}"
SAVE_EPOCH="${SAVE_EPOCH:-1}"
MAX_TO_KEEP="${MAX_TO_KEEP:-20}"

AIR_WIDTH="${AIR_WIDTH:-640}"
AIR_HEIGHT="${AIR_HEIGHT:-480}"
WATER_WIDTH="${WATER_WIDTH:-1360}"
WATER_HEIGHT="${WATER_HEIGHT:-1024}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-64}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-48}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
BETA1="${BETA1:-0.5}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RESET_DATA="${RESET_DATA:-0}"
RESET_CODE="${RESET_CODE:-0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}_gpu${GPU}}"
SAMPLE_DIR="${SAMPLE_DIR:-samples_${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}_gpu${GPU}}"
RESULTS_DIR="${RESULTS_DIR:-${SYN_ROOT}/watergan/results/${DATA_NAME}_bs${BATCH_SIZE}_e${EPOCH}_gpu${GPU}}"

HISTORICAL_PREP_PATH="tools/prepare_watergan_imagenet_ruod_dataset.py"
HISTORICAL_PATCH_PATH="scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh"

die() {
  echo "Error: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || die "${label} not found: ${path}"
}

require_positive_integer() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${label} must be a positive integer: ${value}"
}

safe_remove_dir() {
  local path="$1"
  local allowed_parent="$2"
  local resolved_path
  local resolved_parent

  [[ -n "${path}" && -n "${allowed_parent}" ]] || die "refusing empty removal path"
  resolved_path="$(readlink -m "${path}")"
  resolved_parent="$(readlink -m "${allowed_parent}")"
  [[ "${resolved_path}" == "${resolved_parent}/"* ]] || \
    die "refusing to remove path outside ${resolved_parent}: ${resolved_path}"
  [[ "${resolved_path}" != "${resolved_parent}" ]] || \
    die "refusing to remove parent directory: ${resolved_path}"

  rm -rf -- "${resolved_path}"
}

count_flat_files() {
  local path="$1"
  find "${path}" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' '
}

require_positive_integer "${EPOCH}" "EPOCH"
require_positive_integer "${BATCH_SIZE}" "BATCH_SIZE"
require_positive_integer "${TRAIN_SIZE}" "TRAIN_SIZE"
require_positive_integer "${AIR_PER_CLASS}" "AIR_PER_CLASS"
require_positive_integer "${WATER_REPEAT_TO}" "WATER_REPEAT_TO"
require_positive_integer "${PREP_WORKERS}" "PREP_WORKERS"
require_positive_integer "${MAX_TO_KEEP}" "MAX_TO_KEEP"

git cat-file -e "${LEGACY_COMMIT}^{commit}" 2>/dev/null || \
  die "legacy commit is not available locally: ${LEGACY_COMMIT}"

mkdir -p "${LOG_DIR}" "${SNAPSHOT_DIR}"

git show "${LEGACY_COMMIT}:${HISTORICAL_PREP_PATH}" \
  > "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset.py"
git show "${LEGACY_COMMIT}:${HISTORICAL_PATCH_PATH}" \
  > "${SNAPSHOT_DIR}/patch_watergan_tf15_compat.sh"
chmod +x "${SNAPSHOT_DIR}/patch_watergan_tf15_compat.sh"

cp "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset.py" \
  "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset_parallel.py"

# Apply a narrowly scoped concurrency patch to the pinned historical tool. The
# ordered executor preserves filenames, selected samples, and manifest order.
python - "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset_parallel.py" <<'PY'
from __future__ import print_function

import io
import sys

path = sys.argv[1]
with io.open(path, "r", encoding="utf-8") as handle:
    text = handle.read()

replacements = [
    (
        "import argparse\n",
        "import argparse\nfrom concurrent.futures import ThreadPoolExecutor\n",
    ),
    (
        "    parser.add_argument('--overwrite', action='store_true')\n",
        "    parser.add_argument('--workers', type=int, default=16)\n"
        "    parser.add_argument('--overwrite', action='store_true')\n",
    ),
    (
        """    for index, image_path in enumerate(tqdm(air_images, desc='prepare WaterGAN air/depth', unit='image')):
        rel = image_path.relative_to(air_source)
        depth_path = depth_source / rel.with_suffix('.png')
        if not depth_path.exists():
            missing_depth.append(str(rel).replace('\\\\', '/'))
            continue
        stem = f'{index:08d}'
        air_dst = out_dir / 'air_images' / f'{stem}.png'
        depth_dst = out_dir / 'air_depth' / f'{stem}.mat'
        prepare_rgb(image_path, air_dst, air_size)
        prepare_depth(depth_path, depth_dst, air_size)
        records.append({
            'index': index,
            'source': str(image_path),
            'depth': str(depth_path),
            'relative': str(rel).replace('\\\\', '/'),
            'synset': rel.parts[0] if len(rel.parts) > 1 else 'unknown',
            'original_name': image_path.name,
            'air_image': str(air_dst),
            'air_depth': str(depth_dst),
        })
""",
        """    def prepare_air_item(item):
        index, image_path = item
        rel = image_path.relative_to(air_source)
        depth_path = depth_source / rel.with_suffix('.png')
        if not depth_path.exists():
            return None, str(rel).replace('\\\\', '/')
        stem = f'{index:08d}'
        air_dst = out_dir / 'air_images' / f'{stem}.png'
        depth_dst = out_dir / 'air_depth' / f'{stem}.mat'
        prepare_rgb(image_path, air_dst, air_size)
        prepare_depth(depth_path, depth_dst, air_size)
        return {
            'index': index,
            'source': str(image_path),
            'depth': str(depth_path),
            'relative': str(rel).replace('\\\\', '/'),
            'synset': rel.parts[0] if len(rel.parts) > 1 else 'unknown',
            'original_name': image_path.name,
            'air_image': str(air_dst),
            'air_depth': str(depth_dst),
        }, None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(prepare_air_item, enumerate(air_images))
        for record, missing in tqdm(
                results,
                total=len(air_images),
                desc='prepare WaterGAN air/depth',
                unit='image'):
            if missing is not None:
                missing_depth.append(missing)
            else:
                records.append(record)
""",
    ),
    (
        """    linked_water = 0
    for index, image_path in enumerate(tqdm(water_images, desc='prepare WaterGAN water', unit='image')):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        if args.water_repeat_to > 0 and selected_water_before_repeat > 0 and index >= selected_water_before_repeat:
            base_dst = out_dir / 'water_images' / f'{index % selected_water_before_repeat:08d}.png'
            link_or_copy(base_dst, water_dst)
            linked_water += 1
        else:
            prepare_rgb(image_path, water_dst, water_size)
""",
        """    linked_water = 0
    unique_water_count = len(water_images)
    if args.water_repeat_to > 0 and selected_water_before_repeat > 0:
        unique_water_count = min(selected_water_before_repeat, len(water_images))

    def prepare_water_item(item):
        index, image_path = item
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        prepare_rgb(image_path, water_dst, water_size)

    unique_water_items = enumerate(water_images[:unique_water_count])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(prepare_water_item, unique_water_items)
        for _ in tqdm(
                results,
                total=unique_water_count,
                desc='prepare WaterGAN water',
                unit='image'):
            pass

    for index in tqdm(
            range(unique_water_count, len(water_images)),
            desc='link repeated WaterGAN water',
            unit='image'):
        water_dst = out_dir / 'water_images' / f'{index:08d}.png'
        base_dst = out_dir / 'water_images' / f'{index % selected_water_before_repeat:08d}.png'
        link_or_copy(base_dst, water_dst)
        linked_water += 1
""",
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "legacy preparation patch expected one match, found {}".format(
                count
            )
        )
    text = text.replace(old, new)

with io.open(path, "w", encoding="utf-8", newline="") as handle:
    handle.write(text)
PY

python -m py_compile \
  "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset_parallel.py"

cat > "${SNAPSHOT_DIR}/provenance.txt" <<EOF
legacy_commit=${LEGACY_COMMIT}
historical_prepare=${HISTORICAL_PREP_PATH}
historical_patch=${HISTORICAL_PATCH_PATH}
watergan_source=${WATERGAN_SOURCE_DIR}
watergan_clone=${LEGACY_WATERGAN_DIR}
watergan_requested_commit=${LEGACY_WATERGAN_COMMIT}
data_name=${DATA_NAME}
data_root=${DATA_ROOT}
source_dir=${SOURCE_DIR}
depth_dir=${DEPTH_DIR}
water_source=${WATER_SOURCE}
air_per_class=${AIR_PER_CLASS}
water_repeat_to=${WATER_REPEAT_TO}
seed=${SAMPLE_SEED}
prep_workers=${PREP_WORKERS}
epoch=${EPOCH}
batch_size=${BATCH_SIZE}
train_size=${TRAIN_SIZE}
gpu=${GPU}
max_to_keep=${MAX_TO_KEEP}
EOF

echo "============================================================"
echo "WaterGAN legacy 2026-07-14 balanced50 reproduction"
echo "============================================================"
echo "LEGACY_COMMIT:       ${LEGACY_COMMIT}"
echo "WATERGAN_SOURCE_DIR: ${WATERGAN_SOURCE_DIR}"
echo "LEGACY_WATERGAN_DIR: ${LEGACY_WATERGAN_DIR}"
echo "WATERGAN_COMMIT:     ${LEGACY_WATERGAN_COMMIT}"
echo "SOURCE_DIR:          ${SOURCE_DIR}"
echo "DEPTH_DIR:           ${DEPTH_DIR}"
echo "WATER_SOURCE:        ${WATER_SOURCE}"
echo "DATA_ROOT:           ${DATA_ROOT}"
echo "AIR_PER_CLASS:       ${AIR_PER_CLASS}"
echo "WATER_REPEAT_TO:     ${WATER_REPEAT_TO}"
echo "SAMPLE_SEED:         ${SAMPLE_SEED}"
echo "PREP_WORKERS:        ${PREP_WORKERS}"
echo "GPU:                 ${GPU}"
echo "EPOCH:               ${EPOCH}"
echo "BATCH_SIZE:          ${BATCH_SIZE}"
echo "TRAIN_SIZE:          ${TRAIN_SIZE}"
echo "RUN_PREPARE:         ${RUN_PREPARE}"
echo "RUN_TRAIN:           ${RUN_TRAIN}"
echo "RESET_DATA:          ${RESET_DATA}"
echo "RESET_CODE:          ${RESET_CODE}"
echo "CHECKPOINT_DIR:      ${CHECKPOINT_DIR}"
echo "LOG_DIR:             ${LOG_DIR}"
echo "============================================================"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  require_path "${SOURCE_DIR}" "ImageNet source directory"
  require_path "${DEPTH_DIR}" "Depth Anything directory"
  require_path "${WATER_SOURCE}" "RUOD water directory"

  if [[ -e "${DATA_ROOT}" ]]; then
    [[ "${RESET_DATA}" == "1" ]] || die \
      "legacy data root already exists; use RESET_DATA=1 to rebuild it: ${DATA_ROOT}"
    echo "Reset legacy data root: ${DATA_ROOT}"
    safe_remove_dir "${DATA_ROOT}" "${WORK_ROOT}/watergan/datasets"
  fi

  echo
  echo "Step 1/3: prepare the historical balanced50 dataset"
  python "${SNAPSHOT_DIR}/prepare_watergan_imagenet_ruod_dataset_parallel.py" \
    --air-source "${SOURCE_DIR}" \
    --depth-source "${DEPTH_DIR}" \
    --water-source "${WATER_SOURCE}" \
    --out-dir "${DATA_ROOT}" \
    --air-limit 0 \
    --water-limit 0 \
    --air-per-class "${AIR_PER_CLASS}" \
    --water-repeat-to "${WATER_REPEAT_TO}" \
    --seed "${SAMPLE_SEED}" \
    --air-width "${AIR_WIDTH}" \
    --air-height "${AIR_HEIGHT}" \
    --water-width "${WATER_WIDTH}" \
    --water-height "${WATER_HEIGHT}" \
    --workers "${PREP_WORKERS}" \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/prepare_balanced50.log"
else
  echo
  echo "Step 1/3: skip historical dataset preparation"
fi

for dataset_part in air_images air_depth water_images; do
  require_path "${DATA_ROOT}/${dataset_part}" "prepared ${dataset_part}"
done

air_count="$(count_flat_files "${DATA_ROOT}/air_images")"
depth_count="$(count_flat_files "${DATA_ROOT}/air_depth")"
water_count="$(count_flat_files "${DATA_ROOT}/water_images")"

echo
echo "Prepared legacy dataset counts:"
echo "  air_images:   ${air_count}"
echo "  air_depth:    ${depth_count}"
echo "  water_images: ${water_count}"

[[ "${air_count}" == "${TRAIN_SIZE}" ]] || die \
  "air_images count is ${air_count}, expected ${TRAIN_SIZE}"
[[ "${depth_count}" == "${TRAIN_SIZE}" ]] || die \
  "air_depth count is ${depth_count}, expected ${TRAIN_SIZE}"
[[ "${water_count}" == "${TRAIN_SIZE}" ]] || die \
  "water_images count is ${water_count}, expected ${TRAIN_SIZE}"

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo
  echo "Step 2/3: skip isolated WaterGAN code setup"
  echo "Step 3/3: skip legacy training"
  echo "Historical preparation completed: ${DATA_ROOT}"
  exit 0
fi

require_path "${WATERGAN_SOURCE_DIR}/.git" "WaterGAN Git repository"

if [[ -e "${LEGACY_WATERGAN_DIR}" ]]; then
  [[ "${RESET_CODE}" == "1" ]] || die \
    "legacy WaterGAN clone already exists; use RESET_CODE=1 to recreate it: ${LEGACY_WATERGAN_DIR}"
  echo "Reset isolated WaterGAN clone: ${LEGACY_WATERGAN_DIR}"
  safe_remove_dir "${LEGACY_WATERGAN_DIR}" "$(dirname "${LEGACY_WATERGAN_DIR}")"
fi

echo
echo "Step 2/3: create and patch an isolated historical WaterGAN clone"
git clone --local --no-hardlinks "${WATERGAN_SOURCE_DIR}" "${LEGACY_WATERGAN_DIR}"

if [[ "${LEGACY_WATERGAN_COMMIT}" == "auto" ]]; then
  LEGACY_WATERGAN_COMMIT="$(
    git -C "${LEGACY_WATERGAN_DIR}" rev-list \
      --reverse \
      HEAD \
      -- modelmhl.py | sed -n '1p'
  )"
fi

[[ -n "${LEGACY_WATERGAN_COMMIT}" ]] || \
  die "could not resolve the original WaterGAN commit"
git -C "${LEGACY_WATERGAN_DIR}" cat-file \
  -e "${LEGACY_WATERGAN_COMMIT}^{commit}" 2>/dev/null || \
  die "WaterGAN commit is unavailable: ${LEGACY_WATERGAN_COMMIT}"

echo "Check out original WaterGAN commit: ${LEGACY_WATERGAN_COMMIT}"
git -C "${LEGACY_WATERGAN_DIR}" checkout \
  --detach "${LEGACY_WATERGAN_COMMIT}"

for legacy_file in \
  mainmhl.py \
  mainjamaica.py \
  modelmhl.py \
  modeljamaica.py \
  ops.py \
  utils.py; do
  require_path \
    "${LEGACY_WATERGAN_DIR}/${legacy_file}" \
    "original WaterGAN ${legacy_file}"
done

{
  echo "watergan_resolved_commit=${LEGACY_WATERGAN_COMMIT}"
  echo "watergan_resolved_subject=$(
    git -C "${LEGACY_WATERGAN_DIR}" log \
      -1 --format=%s "${LEGACY_WATERGAN_COMMIT}"
  )"
} >> "${SNAPSHOT_DIR}/provenance.txt"

WATERGAN_DIR="${LEGACY_WATERGAN_DIR}" \
  bash "${SNAPSHOT_DIR}/patch_watergan_tf15_compat.sh"

# Historical tf.train.Saver() retained only the latest five checkpoints. Keep
# more files for the requested epoch-by-epoch comparison; model math is intact.
python - "${LEGACY_WATERGAN_DIR}/modelmhl.py" "${LEGACY_WATERGAN_DIR}/modeljamaica.py" "${MAX_TO_KEEP}" <<'PY'
from __future__ import print_function

import io
import sys

max_to_keep = int(sys.argv[-1])

for filename in sys.argv[1:-1]:
    with io.open(filename, "r", encoding="utf-8") as handle:
        text = handle.read()

    current = "tf.train.Saver(max_to_keep={})".format(max_to_keep)
    if current in text:
        print("Saver retention already patched: {}".format(filename))
        continue

    old = "tf.train.Saver()"
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "expected one tf.train.Saver() in {}, found {}".format(
                filename, count
            )
        )

    text = text.replace(old, current)
    with io.open(filename, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    print("Saver retention patched: {}".format(filename))
PY

python -m py_compile \
  "${LEGACY_WATERGAN_DIR}/modelmhl.py" \
  "${LEGACY_WATERGAN_DIR}/modeljamaica.py"

mkdir -p "${LEGACY_WATERGAN_DIR}/data" "${RESULTS_DIR}"
ln -sfn "${DATA_ROOT}/air_images" \
  "${LEGACY_WATERGAN_DIR}/data/${DATA_NAME}_air_images"
ln -sfn "${DATA_ROOT}/air_depth" \
  "${LEGACY_WATERGAN_DIR}/data/${DATA_NAME}_air_depth"
ln -sfn "${DATA_ROOT}/water_images" \
  "${LEGACY_WATERGAN_DIR}/data/${DATA_NAME}_water_images"

echo
echo "Step 3/3: train with the historical direct mainmhl.py command"
echo "Training log: ${LOG_DIR}/train_legacy.log"

(
  cd "${LEGACY_WATERGAN_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python mainmhl.py \
    --water_dataset "${DATA_NAME}_water_images" \
    --air_dataset "${DATA_NAME}_air_images" \
    --depth_dataset "${DATA_NAME}_air_depth" \
    --epoch "${EPOCH}" \
    --train_size "${TRAIN_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --beta1 "${BETA1}" \
    --input_height "${AIR_HEIGHT}" \
    --input_width "${AIR_WIDTH}" \
    --input_water_height "${WATER_HEIGHT}" \
    --input_water_width "${WATER_WIDTH}" \
    --output_height "${OUTPUT_HEIGHT}" \
    --output_width "${OUTPUT_WIDTH}" \
    --save_epoch "${SAVE_EPOCH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --sample_dir "${SAMPLE_DIR}" \
    --results_dir "${RESULTS_DIR}"
) 2>&1 | tee "${LOG_DIR}/train_legacy.log"

echo
echo "Legacy WaterGAN reproduction completed."
echo "Data:       ${DATA_ROOT}"
echo "Code clone: ${LEGACY_WATERGAN_DIR}"
echo "Checkpoint: ${LEGACY_WATERGAN_DIR}/${CHECKPOINT_DIR}"
echo "Samples:    ${LEGACY_WATERGAN_DIR}/${SAMPLE_DIR}"
echo "Results:    ${RESULTS_DIR}"
