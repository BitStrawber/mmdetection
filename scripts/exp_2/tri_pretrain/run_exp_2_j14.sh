#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_ID="${EXP_ID:-j14}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-29688}"
export BUILD_REALUW_SSL="${BUILD_REALUW_SSL:-0}"
export REALUW_SSL_ROOT="${REALUW_SSL_ROOT:-/media/SSD1/XCX/exp_2/REALUW}"

bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
