#!/bin/bash
set -euo pipefail

# Smoke-test ImageNet DINO ViT-S and ResNet-50 pretraining with the already
# extracted subset, then continue full ImageNet extraction and run both 100e
# pretraining jobs serially.
#
# The smoke dataset is built with symlinks, so it does not duplicate ImageNet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

IMAGENET_TAR_ROOT="${IMAGENET_TAR_ROOT:-/media/HDD0/XCX/IMAGENET}"
IMAGENET_SSL_ROOT="${IMAGENET_SSL_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K}"
SMOKE_SSL_ROOT="${SMOKE_SSL_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K_SMOKE}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_FULL_VITS="${RUN_FULL_VITS:-1}"
RUN_FULL_RESNET50="${RUN_FULL_RESNET50:-1}"

SMOKE_CLASSES="${SMOKE_CLASSES:-8}"
SMOKE_IMAGES_PER_CLASS="${SMOKE_IMAGES_PER_CLASS:-32}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE_PER_GPU="${SMOKE_BATCH_SIZE_PER_GPU:-8}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-4}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

mkdir -p "$LOG_DIR"

build_smoke_imagenet() {
    local src_train="$IMAGENET_SSL_ROOT/imagefolder/train"
    local dst_train="$SMOKE_SSL_ROOT/imagefolder/train"

    if [ ! -d "$src_train" ]; then
        echo "Error: extracted ImageNet train directory not found: $src_train"
        echo "Run partial extraction first, or set IMAGENET_SSL_ROOT to the extracted ImageNet root."
        exit 1
    fi

    export SRC_TRAIN="$src_train"
    export DST_TRAIN="$dst_train"
    export SMOKE_CLASSES
    export SMOKE_IMAGES_PER_CLASS

    python -u - <<'PY'
from pathlib import Path
import json
import os
import shutil

src = Path(os.environ['SRC_TRAIN'])
dst = Path(os.environ['DST_TRAIN'])
num_classes = int(os.environ.get('SMOKE_CLASSES', '8'))
imgs_per_class = int(os.environ.get('SMOKE_IMAGES_PER_CLASS', '32'))

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

classes = [p for p in sorted(src.iterdir()) if p.is_dir() and p.name.startswith('n')]
if not classes:
    raise SystemExit(f'No ImageNet class directories found under {src}')

selected = classes[:num_classes]
summary = {
    'src': str(src),
    'dst': str(dst),
    'requested_classes': num_classes,
    'requested_images_per_class': imgs_per_class,
    'classes': [],
    'total_images': 0,
}

suffixes = {'.jpg', '.jpeg', '.png'}
for cls_dir in selected:
    out_cls = dst / cls_dir.name
    out_cls.mkdir(parents=True, exist_ok=True)
    images = [
        p for p in sorted(cls_dir.iterdir())
        if p.is_file() and p.suffix.lower() in suffixes
    ][:imgs_per_class]
    if not images:
        continue
    for img in images:
        target = out_cls / img.name
        target.symlink_to(img)
    summary['classes'].append({'class': cls_dir.name, 'images': len(images)})
    summary['total_images'] += len(images)

if summary['total_images'] == 0:
    raise SystemExit(f'No images linked into smoke dataset from {src}')

print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_smoke_job() {
    local label="$1"
    local exp_id="$2"
    local task_config="$3"
    local name="$4"
    local port="$5"

    echo "========================================="
    echo "ImageNet DINO smoke: $label"
    echo "smoke_root: $SMOKE_SSL_ROOT"
    echo "epochs: $SMOKE_EPOCHS"
    echo "batch_size_per_gpu: $SMOKE_BATCH_SIZE_PER_GPU"
    echo "========================================="

    EXP_ID="$exp_id" \
    TASK_CONFIG="$task_config" \
    REALUW_SSL_ROOT="$SMOKE_SSL_ROOT" \
    BUILD_REALUW_SSL=0 \
    DINO_NAME="$name" \
    DINO_EPOCHS="$SMOKE_EPOCHS" \
    DINO_WARMUP_EPOCHS=0 \
    DINO_BATCH_SIZE_PER_GPU="$SMOKE_BATCH_SIZE_PER_GPU" \
    DINO_NUM_WORKERS="$SMOKE_NUM_WORKERS" \
    DINO_SAVECKP_FREQ=1 \
    GPU_IDS="$GPU_IDS" \
    PORT="$port" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
}

if [ "$RUN_SMOKE" = "1" ]; then
    build_smoke_imagenet 2>&1 | tee "$LOG_DIR/imagenet_dino_smoke_dataset.log"
    run_smoke_job \
        "ViT-Small 100e recipe, 1e smoke" \
        j14 \
        configs/exp_2/tri_pretrain/s1_imagenet_dino_vits_100e.sh \
        imagenet_dino_vits_smoke \
        "${VITS_SMOKE_PORT:-29694}"
    run_smoke_job \
        "ResNet-50 official 100e recipe, 1e smoke" \
        j7 \
        configs/exp_2/tri_pretrain/s1_imagenet_dino_resnet50_100e.sh \
        imagenet_dino_resnet50_smoke \
        "${RESNET50_SMOKE_PORT:-29695}"
fi

if [ "$RUN_EXTRACT" = "1" ]; then
    IMAGENET_TAR_ROOT="$IMAGENET_TAR_ROOT" \
    IMAGENET_SSL_ROOT="$IMAGENET_SSL_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXTRACT_IMAGENET=1 \
    VERIFY_IMAGENET=1 \
    RUN_PRETRAIN=0 \
    bash "$SCRIPT_DIR/run_exp_2_imagenet_dino_vits_100e.sh"
fi

if [ "$RUN_FULL_VITS" = "1" ]; then
    IMAGENET_TAR_ROOT="$IMAGENET_TAR_ROOT" \
    IMAGENET_SSL_ROOT="$IMAGENET_SSL_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXTRACT_IMAGENET=0 \
    VERIFY_IMAGENET=1 \
    RUN_PRETRAIN=1 \
    GPU_IDS="$GPU_IDS" \
    PORT="${VITS_FULL_PORT:-29692}" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    bash "$SCRIPT_DIR/run_exp_2_imagenet_dino_vits_100e.sh"
fi

if [ "$RUN_FULL_RESNET50" = "1" ]; then
    IMAGENET_TAR_ROOT="$IMAGENET_TAR_ROOT" \
    IMAGENET_SSL_ROOT="$IMAGENET_SSL_ROOT" \
    LOG_DIR="$LOG_DIR" \
    EXTRACT_IMAGENET=0 \
    VERIFY_IMAGENET=1 \
    RUN_PRETRAIN=1 \
    GPU_IDS="$GPU_IDS" \
    PORT="${RESNET50_FULL_PORT:-29693}" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    bash "$SCRIPT_DIR/run_exp_2_imagenet_dino_resnet50_100e.sh"
fi
