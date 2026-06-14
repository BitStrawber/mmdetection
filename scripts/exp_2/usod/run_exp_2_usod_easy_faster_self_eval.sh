#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Train Faster R-CNN on USOD easy_merged and evaluate on the same easy_merged.
# Purpose: check how easy the selected USOD subset is as a single-class
# objectness detection set. This is a self-eval sanity check, not a
# generalization test.

cd "$REPO_ROOT"

USOD_ROOT="${USOD_ROOT:-/media/HDD0/XCX/exp_2/USOD10K}"
EASY_ANN="${EASY_ANN:-$USOD_ROOT/annotations/cross_split_det/easy_merged.json}"
GPU_IDS="${GPU_IDS:-2,3}"
NUM_GPUS="${NUM_GPUS:-2}"
PORT="${PORT:-29820}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WORK_DIR="${WORK_DIR:-work_dirs/usod_easy_faster_rcnn_self_eval}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/usod_easy_faster_rcnn_self_eval.log}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"

mkdir -p "$LOG_DIR" "$WORK_DIR"

if [ ! -f "$EASY_ANN" ]; then
    echo "Error: USOD easy annotation not found: $EASY_ANN"
    echo "Run first:"
    echo "  bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh"
    exit 1
fi

if [ ! -d "$USOD_ROOT/images" ]; then
    echo "Error: USOD image directory not found: $USOD_ROOT/images"
    exit 1
fi

echo "========================================="
echo "USOD easy Faster R-CNN self-eval"
echo "========================================="
echo "USOD_ROOT: $USOD_ROOT"
echo "EASY_ANN: $EASY_ANN"
echo "GPU_IDS: $GPU_IDS"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"
echo "MAX_EPOCHS: $MAX_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "WORK_DIR: $WORK_DIR"
echo "LOG_FILE: $LOG_FILE"
echo "========================================="

export PORT
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

CUDA_VISIBLE_DEVICES="$GPU_IDS" bash tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    "$NUM_GPUS" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
        model.roi_head.bbox_head.num_classes=1 \
        train_dataloader.batch_size="$BATCH_SIZE" \
        train_dataloader.num_workers="$NUM_WORKERS" \
        train_dataloader.dataset.type=CocoDataset \
        train_dataloader.dataset.data_root="$USOD_ROOT/" \
        train_dataloader.dataset.ann_file=annotations/cross_split_det/easy_merged.json \
        train_dataloader.dataset.data_prefix.img=images/ \
        train_dataloader.dataset.metainfo.classes="('object',)" \
        val_dataloader.batch_size=1 \
        val_dataloader.num_workers="$NUM_WORKERS" \
        val_dataloader.dataset.type=CocoDataset \
        val_dataloader.dataset.data_root="$USOD_ROOT/" \
        val_dataloader.dataset.ann_file=annotations/cross_split_det/easy_merged.json \
        val_dataloader.dataset.data_prefix.img=images/ \
        val_dataloader.dataset.metainfo.classes="('object',)" \
        test_dataloader.dataset.type=CocoDataset \
        test_dataloader.dataset.data_root="$USOD_ROOT/" \
        test_dataloader.dataset.ann_file=annotations/cross_split_det/easy_merged.json \
        test_dataloader.dataset.data_prefix.img=images/ \
        test_dataloader.dataset.metainfo.classes="('object',)" \
        val_evaluator.ann_file="$EASY_ANN" \
        test_evaluator.ann_file="$EASY_ANN" \
        train_cfg.max_epochs="$MAX_EPOCHS" \
        default_hooks.checkpoint.save_best=coco/bbox_mAP \
        default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
    2>&1 | tee "$LOG_FILE"

echo "========================================="
echo "Best / latest USOD easy self-eval mAP"
echo "========================================="
grep -a "coco/bbox_mAP:" "$LOG_FILE" | \
    sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*coco\/bbox_mAP_50: ([0-9.]+).*coco\/bbox_mAP_75: ([0-9.]+).*/epoch_\1 mAP=\2 AP50=\3 AP75=\4/' | \
    sort -t= -k2 -nr | head -n 10 || true

echo "========================================="
echo "Checkpoints"
echo "========================================="
ls -lh "$WORK_DIR"/best_coco_bbox_mAP*.pth 2>/dev/null || true
