#!/bin/bash

# Compare two J10 variants with the same merged DFUI source:
#   1) Cascade R-CNN S1 detection pretraining on DFUI+RUOD_easy+UIIS10K_easy,
#      then extract backbone/neck/rpn and finetune RUOD with the original S2.
#   2) HDP/RFTM paper-style feature adaptation on the same merged DFUI source,
#      then finetune RUOD with the existing HDP/RFTM S2 config.

set -e
set -o pipefail

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-work_dirs}"
LOG_DIR="${LOG_DIR:-logs}"
MERGED_ROOT="${MERGED_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY}"

DFUI_IMG_DIR="${DFUI_IMG_DIR:-/media/HDD0/XCX/exp_2/dfui/images}"
RUOD_EASY_IMG_DIR="${RUOD_EASY_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
RUOD_EASY_ANN="${RUOD_EASY_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json}"
UIIS_EASY_IMG_DIR="${UIIS_EASY_IMG_DIR:-/media/HDD0/XCX/exp_2/UIIS10K/img}"
UIIS_EASY_ANN="${UIIS_EASY_ANN:-/media/HDD0/XCX/exp_2/UIIS10K/coco/annotations/cross_split_det/easy_merged.json}"

RUOD_IMG_DIR="${RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
RUOD_ANN="${RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/instances_train.json}"

CASCADE_GPU_IDS="${CASCADE_GPU_IDS:-4,5}"
CASCADE_PORT="${CASCADE_PORT:-29620}"
HDP_GPU_IDS="${HDP_GPU_IDS:-6,7}"
HDP_PORT="${HDP_PORT:-29621}"
NUM_GPUS="${NUM_GPUS:-2}"

THRESHOLD="${THRESHOLD:-0.6}"
S1_LR="${S1_LR:-0.001}"
S1_EPOCHS="${S1_EPOCHS:-20}"
RUOD_PATCH_DIR="${RUOD_PATCH_DIR:-/media/HDD0/XCX/exp_2/HDP_PATCHES/ruod_common_t${THRESHOLD}/ruod_hd}"
EASY_PATCH_DIR="${EASY_PATCH_DIR:-/media/HDD0/XCX/exp_2/HDP_PATCHES/${HDP_EXP_NAME:-j10_hdp_dfui_ruod_uiis}_t${THRESHOLD}/easy_hd}"

mkdir -p "$LOG_DIR"

DFUI_ANN_ARGS=()
if [ -n "${DFUI_ANN:-}" ]; then
    read -r -a DFUI_ANN_LIST <<< "$DFUI_ANN"
    DFUI_ANN_ARGS=(--dfui-ann "${DFUI_ANN_LIST[@]}")
fi

echo "========================================="
echo "Build merged DFUI source"
echo "========================================="
"$PYTHON" tools/merge_dfui_ruod_uiis_easy.py \
    --dfui-img-dir "$DFUI_IMG_DIR" \
    "${DFUI_ANN_ARGS[@]}" \
    --ruod-easy-img-dir "$RUOD_EASY_IMG_DIR" \
    --ruod-easy-ann "$RUOD_EASY_ANN" \
    --uiis-easy-img-dir "$UIIS_EASY_IMG_DIR" \
    --uiis-easy-ann "$UIIS_EASY_ANN" \
    --out-root "$MERGED_ROOT" \
    ${MERGE_EXTRA_ARGS:-} \
    2>&1 | tee "$LOG_DIR/dfui_ruod_uiis_easy_merge.log"

echo "========================================="
echo "Experiment 1: Cascade S1 -> backbone/neck/rpn -> original RUOD S2"
echo "========================================="
CASCADE_S1_DIR="$WORK_DIR/j10_dfui_ruod_uiis_cascade_s1"
CASCADE_S2_DIR="$WORK_DIR/j10_dfui_ruod_uiis_cascade_s2"
mkdir -p "$CASCADE_S1_DIR" "$CASCADE_S2_DIR"

export PORT="$CASCADE_PORT"
CUDA_VISIBLE_DEVICES="$CASCADE_GPU_IDS" bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_s1.py \
    "$NUM_GPUS" \
    --work-dir "$CASCADE_S1_DIR" \
    --cfg-options \
        data_root="$MERGED_ROOT/" \
        train_dataloader.dataset.data_root="$MERGED_ROOT/" \
        train_dataloader.dataset.ann_file="$MERGED_ROOT/annotations/instances_train.json" \
        train_dataloader.dataset.data_prefix.img=images/ \
        val_dataloader.dataset.data_root="$MERGED_ROOT/" \
        val_dataloader.dataset.ann_file="$MERGED_ROOT/annotations/instances_val.json" \
        val_dataloader.dataset.data_prefix.img=images/ \
        test_dataloader.dataset.data_root="$MERGED_ROOT/" \
        test_dataloader.dataset.ann_file="$MERGED_ROOT/annotations/instances_val.json" \
        test_dataloader.dataset.data_prefix.img=images/ \
        val_evaluator.ann_file="$MERGED_ROOT/annotations/instances_val.json" \
        test_evaluator.ann_file="$MERGED_ROOT/annotations/instances_val.json" \
        default_hooks.checkpoint.save_best=coco/bbox_mAP \
        default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_dfui_ruod_uiis_cascade_s1.log"

BEST_CKPT=$(ls -t "$CASCADE_S1_DIR"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$CASCADE_S1_DIR/latest.pth"
fi
if [ ! -f "$BEST_CKPT" ]; then
    echo "Error: missing Cascade S1 checkpoint."
    exit 1
fi
echo "Cascade S1 checkpoint: $BEST_CKPT"

"$PYTHON" tools/extract_backbone.py \
    --checkpoint "$BEST_CKPT" \
    --output "$CASCADE_S1_DIR/backbone_only.pth"

export PORT="$CASCADE_PORT"
CUDA_VISIBLE_DEVICES="$CASCADE_GPU_IDS" bash tools/dist_train.sh \
    configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py \
    "$NUM_GPUS" \
    --work-dir "$CASCADE_S2_DIR" \
    --cfg-options \
        load_from="$CASCADE_S1_DIR/backbone_only.pth" \
        default_hooks.checkpoint.max_keep_ckpts=10 \
    2>&1 | tee "$LOG_DIR/j10_dfui_ruod_uiis_cascade_s2.log"

echo "========================================="
echo "Experiment 2: HDP/RFTM feature adaptation -> existing HDP RUOD S2"
echo "========================================="
EXP_NAME="${HDP_EXP_NAME:-j10_hdp_dfui_ruod_uiis}"
PYTHON="$PYTHON" \
WORK_DIR="$WORK_DIR" \
LOG_DIR="$LOG_DIR" \
EXP_NAME="$EXP_NAME" \
GPU_IDS="$HDP_GPU_IDS" \
NUM_GPUS="$NUM_GPUS" \
PORT="$HDP_PORT" \
THRESHOLD="$THRESHOLD" \
S1_LR="$S1_LR" \
S1_EPOCHS="$S1_EPOCHS" \
RUOD_IMG_DIR="$RUOD_IMG_DIR" \
RUOD_ANN="$RUOD_ANN" \
EASY_IMG_DIR="$MERGED_ROOT/images" \
EASY_ANN="$MERGED_ROOT/annotations/instances_all.json" \
RUOD_PATCH_DIR="$RUOD_PATCH_DIR" \
EASY_PATCH_DIR="$EASY_PATCH_DIR" \
bash run_exp_2_j10_hdp.sh 2>&1 | tee "$LOG_DIR/${EXP_NAME}_launcher.log"

echo "========================================="
echo "Done: $(date)"
echo "Cascade S2: $CASCADE_S2_DIR"
echo "HDP S2: $WORK_DIR/$EXP_NAME/s2"
echo "========================================="
