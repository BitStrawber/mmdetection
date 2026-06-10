#!/bin/bash
set -euo pipefail

# USOD10K objectness expansion for DFUI.
#
# Prerequisite:
#   /media/HDD0/XCX/exp_2/USOD10K/
#     images/
#     annotations/instances_trainval.json
#
# Workflow:
#   1) A/B cross filtering on converted USOD10K with threshold 0.6.
#   2) Merge DFUI + RUOD_easy + UIIS_easy + USOD_easy into 12-class source.

PYTHON="${PYTHON:-python}"
USOD_ROOT="${USOD_ROOT:-/media/HDD0/XCX/exp_2/USOD10K}"
USOD_ANN="${USOD_ANN:-$USOD_ROOT/annotations/instances_trainval.json}"
USOD_CROSS_DIR="${USOD_CROSS_DIR:-$USOD_ROOT/annotations/cross_split_det}"
USOD_GPU_IDS="${USOD_GPU_IDS:-2,3}"
USOD_NUM_GPUS="${USOD_NUM_GPUS:-2}"
USOD_THRESHOLD="${USOD_THRESHOLD:-0.6}"
FORCE_USOD_EASY="${FORCE_USOD_EASY:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"

mkdir -p logs

echo "========================================="
echo "USOD10K easy + DFUI merge"
echo "========================================="
echo "USOD_ROOT: $USOD_ROOT"
echo "USOD_ANN: $USOD_ANN"
echo "USOD_CROSS_DIR: $USOD_CROSS_DIR"
echo "USOD_GPU_IDS: $USOD_GPU_IDS"
echo "USOD_THRESHOLD: $USOD_THRESHOLD"
echo "========================================="

if [ "$FORCE_USOD_EASY" = "1" ] || [ ! -f "$USOD_CROSS_DIR/easy_merged.json" ]; then
    "$PYTHON" tools/usod10k_cross_easy.py \
        --step all \
        --data-root "$USOD_ROOT/" \
        --ann "$USOD_ANN" \
        --cross-dir "$USOD_CROSS_DIR" \
        --img-prefix images/ \
        --gpu-ids "$USOD_GPU_IDS" \
        --num-gpus "$USOD_NUM_GPUS" \
        --threshold "$USOD_THRESHOLD" \
        2>&1 | tee logs/usod10k_cross_easy_full.log
else
    echo "Skip USOD easy filtering, found: $USOD_CROSS_DIR/easy_merged.json"
fi

if [ ! -f "$USOD_CROSS_DIR/easy_merged.json" ]; then
    echo "Error: USOD easy annotation was not created: $USOD_CROSS_DIR/easy_merged.json"
    exit 1
fi

MERGE_ARGS=()
if [ "$FORCE_MERGE" = "1" ]; then
    MERGE_ARGS+=(--overwrite)
fi

"$PYTHON" tools/merge_dfui_ruod_uiis_usod_easy.py \
    --usod-easy-img-dir "$USOD_ROOT/images" \
    --usod-easy-ann "$USOD_CROSS_DIR/easy_merged.json" \
    "${MERGE_ARGS[@]}" \
    2>&1 | tee logs/dfui_ruod_uiis_usod_easy_merge.log

echo "========================================="
echo "USOD expansion finished"
echo "USOD easy: $USOD_CROSS_DIR/easy_merged.json"
echo "Merged root: /media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY"
echo "S1 config: configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py"
echo "========================================="
