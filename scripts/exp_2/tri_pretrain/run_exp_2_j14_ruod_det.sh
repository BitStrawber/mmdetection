#!/bin/bash
set -euo pipefail

# J14 downstream: DINO ViT-Small RealUW checkpoint -> RUOD Cascade R-CNN.
# The DINO teacher backbone is converted once, then the best RUOD checkpoint
# is evaluated by the common downstream runner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="${EXP_ID:-j14}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-29689}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_TEST="${RUN_TEST:-1}"
S1_CKPT="${S1_CKPT:-work_dirs/tri_pretrain/j14_realuw_dino_vits_100e/checkpoint.pth}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-../pretrained_weights/j14_realuw_dino_vits_100e_backbone.pth}"
DET_CONFIG="${DET_CONFIG:-configs/exp_2/cascade-rcnn_vit-small_dino-realuw_fpn_2x_ruod_j14.py}"

if [ "$RUN_CONVERT" = "1" ]; then
    if [ ! -f "$S1_CKPT" ]; then
        echo "Error: J14 DINO S1 checkpoint not found: $S1_CKPT"
        exit 1
    fi
    mkdir -p "$(dirname "$PRETRAIN_CKPT")"
    python tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$S1_CKPT" \
        --source teacher \
        --out "$PRETRAIN_CKPT"
fi

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Error: converted J14 backbone checkpoint not found: $PRETRAIN_CKPT"
    exit 1
fi

EXP_ID="$EXP_ID" \
GPU_IDS="$GPU_IDS" \
PORT="$PORT" \
PRETRAIN_CKPT="$PRETRAIN_CKPT" \
DET_CONFIG="$DET_CONFIG" \
RUN_DET=1 \
RUN_MASK=0 \
RUN_TEST="$RUN_TEST" \
bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_single.sh"
