#!/bin/bash

# J10 HDP/RFTM paper-style pipeline:
#   S0: make HD patches from easy(DFUI) and RUOD with transmission threshold T
#   S1: train only RFTM by feature transference loss on HD patches
#   S2: load RFTM and finetune Cascade R-CNN on RUOD
#
# This differs from the old J10 S1: S1 here is not detection training.

set -e

WORK_DIR="${WORK_DIR:-work_dirs}"
LOG_DIR="${LOG_DIR:-logs}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29530}"
PYTHON="${PYTHON:-python}"

# Adjust these paths to your actual data layout.
RUOD_IMG_DIR="${RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
EASY_IMG_DIR="${EASY_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/easy}"
RUOD_ANN="${RUOD_ANN:-}"
EASY_ANN="${EASY_ANN:-}"

THRESHOLD="${THRESHOLD:-0.6}"
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCH_STRIDE="${PATCH_STRIDE:-128}"
PATCH_ROOT="${PATCH_ROOT:-/media/HDD0/XCX/exp_2/HDP_PATCHES/t${THRESHOLD}}"

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/j10_hdp_s1"

echo "========================================="
echo "J10 HDP/RFTM (paper-style feature prior)"
echo "========================================="
echo "GPU: $GPU_IDS"
echo "T: $THRESHOLD"
echo "RUOD_IMG_DIR: $RUOD_IMG_DIR"
echo "EASY_IMG_DIR: $EASY_IMG_DIR"
echo ""

echo ">>> S0-A: build RUOD HD patches"
RUOD_ANN_ARG=()
if [ -n "$RUOD_ANN" ]; then
    RUOD_ANN_ARG=(--ann "$RUOD_ANN")
fi
"$PYTHON" tools/make_hdp_patches.py \
    --img-dir "$RUOD_IMG_DIR" \
    "${RUOD_ANN_ARG[@]}" \
    --out-dir "$PATCH_ROOT/ruod_hd" \
    --threshold "$THRESHOLD" \
    --patch-size "$PATCH_SIZE" \
    --stride "$PATCH_STRIDE" \
    2>&1 | tee "$LOG_DIR/j10_hdp_s0_ruod.log"

echo ">>> S0-B: build easy/DFUI HD patches"
EASY_ANN_ARG=()
if [ -n "$EASY_ANN" ]; then
    EASY_ANN_ARG=(--ann "$EASY_ANN")
fi
"$PYTHON" tools/make_hdp_patches.py \
    --img-dir "$EASY_IMG_DIR" \
    "${EASY_ANN_ARG[@]}" \
    --out-dir "$PATCH_ROOT/easy_hd" \
    --threshold "$THRESHOLD" \
    --patch-size "$PATCH_SIZE" \
    --stride "$PATCH_STRIDE" \
    2>&1 | tee "$LOG_DIR/j10_hdp_s0_easy.log"

echo ">>> S1: train RFTM prior on HD patches"
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" "$PYTHON" tools/train_rftm_prior.py \
    --easy-patch-dir "$PATCH_ROOT/easy_hd" \
    --ruod-patch-dir "$PATCH_ROOT/ruod_hd" \
    --work-dir "$WORK_DIR/j10_hdp_s1" \
    --out "$WORK_DIR/j10_hdp_s1/rftm_prior.pth" \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    2>&1 | tee "$LOG_DIR/j10_hdp_s1.log"

if [ ! -f "$WORK_DIR/j10_hdp_s1/rftm_prior.pth" ]; then
    echo "Error: RFTM prior checkpoint was not created."
    exit 1
fi

echo ">>> S2: RUOD finetune with pretrained RFTM"
mkdir -p "$WORK_DIR/j10_hdp_s2"
export PORT
CUDA_VISIBLE_DEVICES=$GPU_IDS PYTHONPATH="$(dirname "$0"):$PYTHONPATH" "$PYTHON" -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    tools/train.py \
    configs/exp_2/cascade-rcnn_r50-rftm-hdp_fpn_2x_ruod_j10_s2.py \
    --launcher pytorch \
    --work-dir "$WORK_DIR/j10_hdp_s2" \
    --cfg-options \
        model.backbone.rftm_init="$WORK_DIR/j10_hdp_s1/rftm_prior.pth" \
        default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_hdp_s2.log"

echo "<<< J10 HDP/RFTM done: $(date)"
