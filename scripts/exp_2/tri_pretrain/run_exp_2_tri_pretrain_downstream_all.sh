#!/bin/bash
set -euo pipefail

# Run downstream pairs for several strategies. Within each strategy, RUOD det
# and UIIS10K mask run in parallel on two separate 2-GPU groups.
#
# Default layout on an 8-GPU server:
#   det  -> GPU 0,1
#   mask -> GPU 2,3
# Strategies are run sequentially to avoid over-subscribing GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS="${TASKS:-j6 j7 j11 j12 j13}"
DET_GPU_IDS="${DET_GPU_IDS:-0,1}"
MASK_GPU_IDS="${MASK_GPU_IDS:-2,3}"
BASE_PORT="${BASE_PORT:-29740}"

i=0
for task in $TASKS; do
    export EXP_ID="$task"
    export DET_GPU_IDS MASK_GPU_IDS
    export DET_PORT=$((BASE_PORT + i * 2))
    export MASK_PORT=$((BASE_PORT + i * 2 + 1))
    echo "========================================="
    echo "Downstream strategy: $task"
    echo "DET_GPU_IDS=$DET_GPU_IDS MASK_GPU_IDS=$MASK_GPU_IDS"
    echo "DET_PORT=$DET_PORT MASK_PORT=$MASK_PORT"
    echo "========================================="
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_downstream_pair.sh"
    i=$((i + 1))
done
