#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_ID="${EXP_ID:-j13}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-29693}"

bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_full_task.sh"
