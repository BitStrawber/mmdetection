#!/bin/bash
set -Eeuo pipefail

# Serial controlled-data DINO comparison:
#   ImageNet-100K   R50 -> ViT-S
#   RealUW-100K     R50 -> ViT-S
#   Synthetic5-100K R50 -> ViT-S
#
# Every task uses the same 100 epochs, GPU group, per-GPU batch size, and
# facebookresearch/DINO recipe used by the existing full-dataset experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SUBSET_ROOT="${SUBSET_ROOT:-/media/SSD1/XCX/exp_2/dino_100k_control}"
RUNNER="${RUNNER:-scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh}"
R50_CONFIG="${R50_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_resnet50_100e.sh}"
VITS_CONFIG="${VITS_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_vits_100e.sh}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-29830}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_DIR/dino_100k_control_six_serial.log}"

DINO_EPOCHS="${DINO_EPOCHS:-100}"
DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-50}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-100000}"
CHECK_DATA="${CHECK_DATA:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

mkdir -p "$WORK_ROOT" "$LOG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

die() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

validate_dataset() {
    local name="$1"
    local data_root="$2"
    local expected_classes="$3"

    python - "$name" "$data_root/imagefolder/train" \
        "$EXPECTED_IMAGES" "$expected_classes" <<'PY'
import os
import sys
from pathlib import Path

name, root, expected_images, expected_classes = sys.argv[1:]
root = Path(root)
expected_images = int(expected_images)
expected_classes = int(expected_classes)
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

if not root.is_dir():
    raise SystemExit(f'{name}: missing train directory: {root}')

class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
images = 0
zero_size = 0
for class_dir in class_dirs:
    for path in class_dir.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        images += 1
        if path.stat().st_size == 0:
            zero_size += 1

print(
    f'{name}: root={root} classes={len(class_dirs)} '
    f'images={images} zero_size={zero_size}',
    flush=True)

if len(class_dirs) != expected_classes:
    raise SystemExit(
        f'{name}: expected {expected_classes} classes, '
        f'found {len(class_dirs)}')
if images != expected_images:
    raise SystemExit(
        f'{name}: expected {expected_images} images, found {images}')
if zero_size:
    raise SystemExit(f'{name}: found {zero_size} zero-size images')
PY
}

validate_checkpoint() {
    local name="$1"
    local expected_arch="$2"
    local checkpoint="$WORK_ROOT/$name/checkpoint.pth"

    [ -s "$checkpoint" ] || die "Missing checkpoint: $checkpoint"

    python - "$checkpoint" "$DINO_EPOCHS" "$expected_arch" <<'PY'
import sys
import torch

path, expected_epoch, expected_arch = sys.argv[1:]
expected_epoch = int(expected_epoch)

try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location='cpu')

epoch = checkpoint.get('epoch')
args = checkpoint.get('args')
arch = getattr(args, 'arch', None)
teacher = checkpoint.get('teacher')

print(f'checkpoint={path}')
print(f'epoch={epoch} expected_epoch={expected_epoch}')
print(f'arch={arch} expected_arch={expected_arch}')
print(f'teacher_tensors={len(teacher) if isinstance(teacher, dict) else 0}')

if epoch != expected_epoch:
    raise SystemExit(f'Incomplete checkpoint epoch: {epoch}')
if arch != expected_arch:
    raise SystemExit(f'Unexpected architecture: {arch}')
if not isinstance(teacher, dict) or not teacher:
    raise SystemExit('Checkpoint has no teacher state dict')
PY
}

run_one() {
    local dataset_name="$1"
    local data_root="$2"
    local backbone_name="$3"
    local exp_id="$4"
    local task_config="$5"
    local expected_arch="$6"
    local port="$7"
    local run_name="control100k_${dataset_name}_dino_${backbone_name}_100e"

    echo "================================================================================"
    echo "[$(timestamp)] Start $run_name"
    echo "data_path: $data_root/imagefolder/train"
    echo "gpu_ids: $GPU_IDS"
    echo "wait_for_gpus: 0"
    echo "epochs: $DINO_EPOCHS"
    echo "batch_size_per_gpu: $DINO_BATCH_SIZE_PER_GPU"
    echo "port: $port"
    echo "work_dir: $WORK_ROOT/$run_name"
    echo "log: $LOG_DIR/${run_name}_s1.log"
    echo "================================================================================"

    env \
        EXP_ID="$exp_id" \
        TASK_CONFIG="$task_config" \
        DINO_NAME="$run_name" \
        DINO_EPOCHS="$DINO_EPOCHS" \
        DINO_BATCH_SIZE_PER_GPU="$DINO_BATCH_SIZE_PER_GPU" \
        DINO_NUM_WORKERS="$DINO_NUM_WORKERS" \
        DINO_SAVECKP_FREQ="$DINO_SAVECKP_FREQ" \
        DINO_INIT_CHECKPOINT= \
        REALUW_SSL_ROOT="$data_root" \
        BUILD_REALUW_SSL=0 \
        GPU_IDS="$GPU_IDS" \
        PORT="$port" \
        WORK_ROOT="$WORK_ROOT" \
        LOG_DIR="$LOG_DIR" \
        WAIT_FOR_GPUS=0 \
        bash "$RUNNER"

    validate_checkpoint "$run_name" "$expected_arch"
    echo "[$(timestamp)] Complete $run_name"
}

for required in "$RUNNER" "$R50_CONFIG" "$VITS_CONFIG"; do
    [ -f "$required" ] || die "Required file not found: $required"
done

declare -a DATASETS=(
    "imagenet100k|$SUBSET_ROOT/imagenet100k|1000"
    "realuw100k|$SUBSET_ROOT/realuw100k|1"
    "synthetic5_100k|$SUBSET_ROOT/synthetic5_100k|1000"
)

if [ "$CHECK_DATA" = "1" ]; then
    echo "[$(timestamp)] Validate three controlled 100K datasets"
    for spec in "${DATASETS[@]}"; do
        IFS='|' read -r dataset_name data_root expected_classes <<< "$spec"
        validate_dataset "$dataset_name" "$data_root" "$expected_classes"
    done
else
    echo "CHECK_DATA=$CHECK_DATA, skip dataset validation."
fi

if [ "$VALIDATE_ONLY" = "1" ]; then
    echo "[$(timestamp)] VALIDATE_ONLY=1, stop before training"
    exit 0
fi

job_index=0
for spec in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name data_root expected_classes <<< "$spec"

    run_one \
        "$dataset_name" "$data_root" resnet50 j7 \
        "$R50_CONFIG" resnet50 "$((BASE_PORT + job_index))"
    job_index=$((job_index + 1))

    run_one \
        "$dataset_name" "$data_root" vits j14 \
        "$VITS_CONFIG" vit_small "$((BASE_PORT + job_index))"
    job_index=$((job_index + 1))
done

echo "================================================================================"
echo "[$(timestamp)] All six DINO 100K controlled pretraining tasks completed"
for spec in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name data_root expected_classes <<< "$spec"
    echo "$WORK_ROOT/control100k_${dataset_name}_dino_resnet50_100e/checkpoint.pth"
    echo "$WORK_ROOT/control100k_${dataset_name}_dino_vits_100e/checkpoint.pth"
done
echo "pipeline_log: $PIPELINE_LOG"
echo "================================================================================"
