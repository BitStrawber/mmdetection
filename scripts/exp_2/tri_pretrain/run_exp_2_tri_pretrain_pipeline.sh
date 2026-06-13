#!/bin/bash
set -euo pipefail

# Full organization:
#   1. S1: run selected RealUW SSL pretraining tasks sequentially with 8 GPUs.
#   2. S2: run RUOD det and UIIS10K mask in parallel with 2 GPUs each.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S1_TASKS="${S1_TASKS:-j6 j7 j11 j12 j13}"
S2_TASKS="${S2_TASKS:-$S1_TASKS}"
S1_GPU_IDS="${S1_GPU_IDS:-0,1,2,3,4,5,6,7}"
DET_GPU_IDS="${DET_GPU_IDS:-0,1}"
MASK_GPU_IDS="${MASK_GPU_IDS:-2,3}"
RUN_S1="${RUN_S1:-1}"
RUN_S2="${RUN_S2:-1}"

if [ "$RUN_S1" = "1" ]; then
    TASKS="$S1_TASKS" GPU_IDS="$S1_GPU_IDS" \
        bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1_all.sh"
fi

if [ "$RUN_S2" = "1" ]; then
    TASKS="$S2_TASKS" DET_GPU_IDS="$DET_GPU_IDS" MASK_GPU_IDS="$MASK_GPU_IDS" \
        bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_downstream_all.sh"
fi
