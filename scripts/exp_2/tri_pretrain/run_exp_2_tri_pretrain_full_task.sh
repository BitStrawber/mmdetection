#!/bin/bash
set -euo pipefail

# Run one complete Tri-pretrain task:
#   S1 SSL pretraining -> backbone conversion -> fixed downstream RUOD/UIIS run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="${EXP_ID:?EXP_ID is required: j6, j7, j11, j12, or j13}"
RUN_S1="${RUN_S1:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"
PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"

mkdir -p "$PRETRAIN_DIR"

case "$EXP_ID" in
    j6)
        s1_name="j6_realuw_spark_resnet50"
        default_s1_ckpt="work_dirs/tri_pretrain/$s1_name/resnet50_1kpretrained_timm_style.pth"
        default_pretrain_ckpt="$PRETRAIN_DIR/j6_realuw_spark_resnet50_backbone.pth"
        export DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_realuw-pretrain_fpn_2x_ruod_j6.py}"
        export MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_realuw-pretrain_fpn_2x_uiis10k_j6_mask.py}"
        ;;
    j7)
        s1_name="j7_realuw_dino_resnet50"
        default_s1_ckpt="work_dirs/tri_pretrain/$s1_name/checkpoint.pth"
        default_pretrain_ckpt="$PRETRAIN_DIR/j7_realuw_dino_resnet50_backbone.pth"
        export DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino-realuw_fpn_2x_ruod_j7.py}"
        export MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_dino-realuw_fpn_2x_uiis10k_j7_mask.py}"
        ;;
    j11)
        s1_name="j11_realuw_mae_vit_base"
        default_s1_ckpt="work_dirs/tri_pretrain/$s1_name/latest.pth"
        default_pretrain_ckpt="$PRETRAIN_DIR/j11_realuw_mae_vit_base_backbone.pth"
        export DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_vit-base_mae-realuw_fpn_2x_ruod_j11.py}"
        export MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_vit-base_mae-realuw_fpn_2x_uiis10k_j11_mask.py}"
        ;;
    j12)
        s1_name="j12_realuw_simmim_swinv2_base"
        default_s1_ckpt="work_dirs/tri_pretrain/$s1_name/latest.pth"
        default_pretrain_ckpt="$PRETRAIN_DIR/j12_realuw_simmim_swinv2_base_backbone.pth"
        export DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_swinv2-base_mae-realuw_fpn_2x_ruod_j12.py}"
        export MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_swinv2-base_mae-realuw_fpn_2x_uiis10k_j12_mask.py}"
        ;;
    j13)
        s1_name="j13_realuw_spark_convnextv2_tiny"
        default_s1_ckpt="work_dirs/tri_pretrain/$s1_name/latest.pth"
        default_pretrain_ckpt="$PRETRAIN_DIR/j13_realuw_spark_convnextv2_tiny_backbone.pth"
        export DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_convnext-tiny_mae-realuw_fpn_2x_ruod_j13.py}"
        export MASK_CONFIG="${MASK_CONFIG:-configs/exp_2/mask-rcnn_convnext-tiny_mae-realuw_fpn_2x_uiis10k_j13_mask.py}"
        ;;
    *)
        echo "Error: unsupported EXP_ID=$EXP_ID"
        exit 1
        ;;
esac

if [ "$RUN_S1" = "1" ]; then
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
fi

S1_CKPT="${S1_CKPT:-$default_s1_ckpt}"
if [ ! -f "$S1_CKPT" ]; then
    latest_in_dir=$(ls -t "work_dirs/tri_pretrain/$s1_name"/*.pth 2>/dev/null | head -1 || true)
    if [ -n "$latest_in_dir" ]; then
        S1_CKPT="$latest_in_dir"
    fi
fi
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-$default_pretrain_ckpt}"

if [ "$RUN_CONVERT" = "1" ]; then
    if [ ! -f "$S1_CKPT" ]; then
        echo "Error: S1 checkpoint not found: $S1_CKPT"
        exit 1
    fi
    python tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$S1_CKPT" \
        --out "$PRETRAIN_CKPT"
fi

if [ "$RUN_DOWNSTREAM" = "0" ]; then
    echo "RUN_DOWNSTREAM=0, stop after S1/conversion."
    echo "S1 checkpoint: $S1_CKPT"
    echo "Backbone checkpoint: $PRETRAIN_CKPT"
    exit 0
fi

bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_single.sh"
