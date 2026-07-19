#!/bin/bash
set -Eeuo pipefail

# Pretrain DINO ResNet-50 and then DINO ViT-S/16 for 100 epochs on the
# merged five-method synthetic ImageNet training split. DINO does not use the
# validation split; CHECK_VAL only validates that the retained split is intact.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/merged_5methods}"
IMAGEFOLDER_ROOT="$DATA_ROOT/imagefolder"
TRAIN_ROOT="$IMAGEFOLDER_ROOT/train"
VAL_ROOT="$IMAGEFOLDER_ROOT/val"

RUNNER="${RUNNER:-scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh}"
R50_CONFIG="${R50_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_resnet50_100e.sh}"
VITS_CONFIG="${VITS_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_vits_100e.sh}"
OCCUPIER="${OCCUPIER:-scripts/exp_2/utils/run_exp_2_gpu_occupier.sh}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
R50_PORT="${R50_PORT:-29810}"
VITS_PORT="${VITS_PORT:-29811}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"

R50_NAME="${R50_NAME:-synthetic5_merged_dino_resnet50_100e}"
VITS_NAME="${VITS_NAME:-synthetic5_merged_dino_vits_100e}"

EXPECTED_TRAIN_IMAGES="${EXPECTED_TRAIN_IMAGES:-1250000}"
EXPECTED_VAL_IMAGES="${EXPECTED_VAL_IMAGES:-50000}"
EXPECTED_CLASSES="${EXPECTED_CLASSES:-1000}"
CHECK_DATA="${CHECK_DATA:-1}"
CHECK_VAL="${CHECK_VAL:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

DINO_EPOCHS="${DINO_EPOCHS:-100}"
DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-50}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
R50_WAIT_FOR_GPUS="${R50_WAIT_FOR_GPUS:-$WAIT_FOR_GPUS}"
VITS_WAIT_FOR_GPUS="${VITS_WAIT_FOR_GPUS:-0}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

OCCUPY_ON_FAILURE="${OCCUPY_ON_FAILURE:-1}"
OCCUPY_MEM_MB="${OCCUPY_MEM_MB:-20000}"
OCCUPY_TARGET_UTIL="${OCCUPY_TARGET_UTIL:-70}"
OCCUPY_START_MAX_USED_MB="${OCCUPY_START_MAX_USED_MB:-3000}"
OCCUPY_LOG="${OCCUPY_LOG:-$LOG_DIR/synthetic5_dino_failure_gpu_occupier.log}"

CURRENT_STAGE="initialization"
OCCUPIER_STARTED=0

mkdir -p "$WORK_ROOT" "$LOG_DIR"

start_failure_occupier() {
    local exit_code="$1"

    if [ "$OCCUPY_ON_FAILURE" != "1" ]; then
        echo "OCCUPY_ON_FAILURE=$OCCUPY_ON_FAILURE, skip GPU occupier."
        return
    fi
    if [ "$OCCUPIER_STARTED" = "1" ]; then
        return
    fi
    if [ ! -f "$OCCUPIER" ]; then
        echo "Warning: GPU occupier script not found: $OCCUPIER"
        return
    fi

    OCCUPIER_STARTED=1
    echo "Training failed during stage: $CURRENT_STAGE (exit=$exit_code)"
    echo "Start failure GPU occupier: GPUs=$GPU_IDS memory=${OCCUPY_MEM_MB}MB target_util=${OCCUPY_TARGET_UTIL}%"

    nohup env \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_START_MAX_USED_MB="$OCCUPY_START_MAX_USED_MB" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        OCCUPY_TARGET_UTIL="$OCCUPY_TARGET_UTIL" \
        LOG_DIR="$LOG_DIR" \
        LOG_FILE="$OCCUPY_LOG" \
        PYTHON_BIN="$(command -v python)" \
        bash "$OCCUPIER" "$GPU_IDS" "$OCCUPY_MEM_MB" \
        >/dev/null 2>&1 &

    echo "Failure GPU occupier PID: $!"
    echo "Failure GPU occupier log: $OCCUPY_LOG"
}

on_error() {
    local exit_code=$?
    trap - ERR
    start_failure_occupier "$exit_code"
    exit "$exit_code"
}
trap on_error ERR

validate_split() {
    local split="$1"
    local split_root="$2"
    local expected_images="$3"

    python - "$split" "$split_root" "$expected_images" "$EXPECTED_CLASSES" <<'PY'
import os
import sys
from pathlib import Path

split, root, expected_images, expected_classes = sys.argv[1:]
root = Path(root)
expected_images = int(expected_images)
expected_classes = int(expected_classes)
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

if not root.is_dir():
    raise SystemExit(f'Missing {split} directory: {root}')

class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
if len(class_dirs) != expected_classes:
    raise SystemExit(
        f'{split}: expected {expected_classes} classes, found {len(class_dirs)}')

images = 0
zero_size = 0
for class_dir in class_dirs:
    with os.scandir(class_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            if Path(entry.name).suffix.lower() not in extensions:
                continue
            images += 1
            if entry.stat(follow_symlinks=True).st_size == 0:
                zero_size += 1

print(f'{split}: classes={len(class_dirs)} images={images} zero_size={zero_size}')
if images != expected_images:
    raise SystemExit(
        f'{split}: expected {expected_images} images, found {images}')
if zero_size:
    raise SystemExit(f'{split}: found {zero_size} zero-size images')
PY
}

validate_checkpoint() {
    local name="$1"
    local expected_arch="$2"
    local checkpoint="$WORK_ROOT/$name/checkpoint.pth"

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
print(f'teacher_parameters={len(teacher) if isinstance(teacher, dict) else 0}')

if epoch != expected_epoch:
    raise SystemExit(f'Incomplete checkpoint epoch: {epoch}')
if arch != expected_arch:
    raise SystemExit(f'Unexpected checkpoint architecture: {arch}')
if not isinstance(teacher, dict) or not teacher:
    raise SystemExit('Checkpoint has no teacher state dict')
PY
}

run_dino() {
    local exp_id="$1"
    local task_config="$2"
    local name="$3"
    local port="$4"
    local expected_arch="$5"
    local wait_for_gpus="$6"

    CURRENT_STAGE="$name training"
    echo "================================================================"
    echo "stage: $CURRENT_STAGE"
    echo "data_path: $TRAIN_ROOT"
    echo "validation_split_used_by_dino: no"
    echo "gpu_ids: $GPU_IDS"
    echo "wait_for_gpus: $wait_for_gpus"
    echo "epochs: $DINO_EPOCHS"
    echo "batch_size_per_gpu: $DINO_BATCH_SIZE_PER_GPU"
    echo "work_dir: $WORK_ROOT/$name"
    echo "================================================================"

    env \
        EXP_ID="$exp_id" \
        TASK_CONFIG="$task_config" \
        DINO_NAME="$name" \
        DINO_EPOCHS="$DINO_EPOCHS" \
        DINO_BATCH_SIZE_PER_GPU="$DINO_BATCH_SIZE_PER_GPU" \
        DINO_NUM_WORKERS="$DINO_NUM_WORKERS" \
        DINO_SAVECKP_FREQ="$DINO_SAVECKP_FREQ" \
        DINO_INIT_CHECKPOINT= \
        REALUW_SSL_ROOT="$DATA_ROOT" \
        BUILD_REALUW_SSL=0 \
        GPU_IDS="$GPU_IDS" \
        PORT="$port" \
        WORK_ROOT="$WORK_ROOT" \
        LOG_DIR="$LOG_DIR" \
        WAIT_FOR_GPUS="$wait_for_gpus" \
        GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        bash "$RUNNER"

    CURRENT_STAGE="$name checkpoint validation"
    validate_checkpoint "$name" "$expected_arch"
}

for required in "$RUNNER" "$R50_CONFIG" "$VITS_CONFIG"; do
    if [ ! -f "$required" ]; then
        echo "Error: required file not found: $required"
        false
    fi
done

if [ "$CHECK_DATA" = "1" ]; then
    CURRENT_STAGE="training dataset validation"
    validate_split train "$TRAIN_ROOT" "$EXPECTED_TRAIN_IMAGES"
    if [ "$CHECK_VAL" = "1" ]; then
        CURRENT_STAGE="retained validation dataset validation"
        validate_split val "$VAL_ROOT" "$EXPECTED_VAL_IMAGES"
    else
        echo "CHECK_VAL=$CHECK_VAL, skip retained validation split validation."
    fi
else
    echo "CHECK_DATA=$CHECK_DATA, skip dataset validation."
fi

if [ "$VALIDATE_ONLY" = "1" ]; then
    CURRENT_STAGE="validation complete"
    trap - ERR
    echo "VALIDATE_ONLY=1, dataset and required-file checks passed."
    echo "DINO would train on: $TRAIN_ROOT"
    echo "DINO would not train on: $VAL_ROOT"
    exit 0
fi

run_dino j7 "$R50_CONFIG" "$R50_NAME" "$R50_PORT" resnet50 "$R50_WAIT_FOR_GPUS"
run_dino j14 "$VITS_CONFIG" "$VITS_NAME" "$VITS_PORT" vit_small "$VITS_WAIT_FOR_GPUS"

CURRENT_STAGE="complete"
trap - ERR
echo "================================================================"
echo "Both merged synthetic ImageNet DINO pretraining tasks completed."
echo "ResNet-50: $WORK_ROOT/$R50_NAME/checkpoint.pth"
echo "ViT-S:     $WORK_ROOT/$VITS_NAME/checkpoint.pth"
echo "DINO used only: $TRAIN_ROOT"
echo "Retained for later evaluation only: $VAL_ROOT"
echo "================================================================"
