#!/bin/bash

# Full pipeline:
#   1) Convert UIIS10K instance segmentation COCO annotations to detection COCO.
#   2) Run UIIS10K A/B cross filtering to obtain UIIS10K easy data.
#   3) Merge DFUI + RUOD easy + UIIS10K easy.
#   4) Run two J10 experiments on the merged DFUI source:
#      - Cascade R-CNN S1 detection pretraining -> original RUOD S2.
#      - HDP/RFTM feature adaptation -> original HDP/RFTM RUOD S2.
#
# Defaults match the current server layout under /media/HDD0/XCX/exp_2.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

PYTHON="${PYTHON:-python}"

UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K}"
UIIS_DATA_ROOT="${UIIS_DATA_ROOT:-$UIIS_ROOT/}"
UIIS_IMG_PREFIX="${UIIS_IMG_PREFIX:-img/}"
UIIS_TRAIN_SEG_ANN="${UIIS_TRAIN_SEG_ANN:-$UIIS_ROOT/coco/annotations/instances_train.json}"
UIIS_TRAIN_DET_ANN="${UIIS_TRAIN_DET_ANN:-$UIIS_ROOT/coco/annotations/instances_train_det.json}"
UIIS_CROSS_DIR="${UIIS_CROSS_DIR:-$UIIS_ROOT/coco/annotations/cross_split_det}"
UIIS_EASY_ANN="${UIIS_EASY_ANN:-$UIIS_CROSS_DIR/easy_merged.json}"

UIIS_GPU_IDS="${UIIS_GPU_IDS:-4,5}"
UIIS_NUM_GPUS="${UIIS_NUM_GPUS:-2}"
UIIS_THRESHOLD="${UIIS_THRESHOLD:-0.6}"

# FORCE_UIIS_CONVERT=1: rebuild instances_train_det.json.
# FORCE_UIIS_EASY=1: rerun UIIS10K A/B training and easy filtering.
# FORCE_MERGE=1: force image recopy during source merge.
FORCE_UIIS_CONVERT="${FORCE_UIIS_CONVERT:-0}"
FORCE_UIIS_EASY="${FORCE_UIIS_EASY:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

echo "========================================="
echo "Full UIIS easy + J10 comparison pipeline"
echo "========================================="
echo "UIIS_ROOT: $UIIS_ROOT"
echo "UIIS_GPU_IDS: $UIIS_GPU_IDS"
echo "UIIS_THRESHOLD: $UIIS_THRESHOLD"
echo ""

mkdir -p logs

echo "========================================="
echo "Step 1: UIIS10K segmentation -> detection COCO"
echo "========================================="
if [ "$FORCE_UIIS_CONVERT" = "1" ] || [ ! -f "$UIIS_TRAIN_DET_ANN" ]; then
    "$PYTHON" tools/convert_uiis10k_seg_to_det.py \
        --ann "$UIIS_TRAIN_SEG_ANN" \
        --out "$UIIS_TRAIN_DET_ANN" \
        2>&1 | tee logs/uiis10k_convert_det.log
else
    echo "Skip conversion, found: $UIIS_TRAIN_DET_ANN"
fi

echo "========================================="
echo "Step 2: UIIS10K A/B easy filtering"
echo "========================================="
if [ "$FORCE_UIIS_EASY" = "1" ] || [ ! -f "$UIIS_EASY_ANN" ]; then
    "$PYTHON" tools/uiis10k_cross_easy.py \
        --step all \
        --data-root "$UIIS_DATA_ROOT" \
        --ann "$UIIS_TRAIN_DET_ANN" \
        --cross-dir "$UIIS_CROSS_DIR" \
        --img-prefix "$UIIS_IMG_PREFIX" \
        --gpu-ids "$UIIS_GPU_IDS" \
        --num-gpus "$UIIS_NUM_GPUS" \
        --threshold "$UIIS_THRESHOLD" \
        2>&1 | tee logs/uiis10k_cross_easy_full.log
else
    echo "Skip UIIS easy filtering, found: $UIIS_EASY_ANN"
fi

if [ ! -f "$UIIS_EASY_ANN" ]; then
    echo "Error: UIIS easy annotation was not created: $UIIS_EASY_ANN"
    exit 1
fi

echo "========================================="
echo "Step 3-4: merge source and run two J10 experiments"
echo "========================================="
MERGE_ARGS=""
if [ "$FORCE_MERGE" = "1" ]; then
    MERGE_ARGS="--overwrite"
fi

UIIS_EASY_IMG_DIR="${UIIS_EASY_IMG_DIR:-$UIIS_ROOT/img}" \
UIIS_EASY_ANN="$UIIS_EASY_ANN" \
MERGE_EXTRA_ARGS="$MERGE_ARGS" \
bash "$SCRIPT_DIR/../j10/run_exp_2_j10_dfui_ruod_uiis_compare.sh"

echo "========================================="
echo "Full pipeline finished: $(date)"
echo "UIIS easy: $UIIS_EASY_ANN"
echo "Merged DFUI root: ${MERGED_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY}"
echo "========================================="
