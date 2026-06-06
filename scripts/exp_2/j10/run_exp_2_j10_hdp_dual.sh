#!/bin/bash

# Run two J10 HDP/RFTM experiments:
#   1) easy_ruod as DFUI on GPU 4,5
#   2) DFUI_NEW as DFUI on GPU 6,7
#
# Override PYTHON/RUOD paths if your environment differs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-work_dirs}"
LOG_DIR="${LOG_DIR:-logs}"
THRESHOLD="${THRESHOLD:-0.6}"
S1_LR="${S1_LR:-0.001}"

RUOD_IMG_DIR="${RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
RUOD_ANN="${RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/instances_train.json}"

EASY_RUOD_IMG_DIR="${EASY_RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
EASY_RUOD_ANN="${EASY_RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json}"

DFUI_NEW_IMG_DIR="${DFUI_NEW_IMG_DIR:-/media/HDD0/XCX/exp_2/DFUI_NEW/images}"
DFUI_NEW_ANN="${DFUI_NEW_ANN:-/media/HDD0/XCX/exp_2/DFUI_NEW/annotations/instances_train.json}"

mkdir -p "$LOG_DIR"

echo "Launching easy_ruod-as-DFUI on GPU 4,5"
PYTHON="$PYTHON" \
WORK_DIR="$WORK_DIR" \
LOG_DIR="$LOG_DIR" \
EXP_NAME="j10_hdp_easy_ruod" \
GPU_IDS="4,5" \
NUM_GPUS="2" \
PORT="29545" \
THRESHOLD="$THRESHOLD" \
S1_LR="$S1_LR" \
RUOD_IMG_DIR="$RUOD_IMG_DIR" \
RUOD_ANN="$RUOD_ANN" \
EASY_IMG_DIR="$EASY_RUOD_IMG_DIR" \
EASY_ANN="$EASY_RUOD_ANN" \
bash "$SCRIPT_DIR/run_exp_2_j10_hdp.sh" > "$LOG_DIR/j10_hdp_easy_ruod_launcher.log" 2>&1 &
PID_EASY=$!

echo "Launching dfui_new-as-DFUI on GPU 6,7"
PYTHON="$PYTHON" \
WORK_DIR="$WORK_DIR" \
LOG_DIR="$LOG_DIR" \
EXP_NAME="j10_hdp_dfui_new" \
GPU_IDS="6,7" \
NUM_GPUS="2" \
PORT="29567" \
THRESHOLD="$THRESHOLD" \
S1_LR="$S1_LR" \
RUOD_IMG_DIR="$RUOD_IMG_DIR" \
RUOD_ANN="$RUOD_ANN" \
EASY_IMG_DIR="$DFUI_NEW_IMG_DIR" \
EASY_ANN="$DFUI_NEW_ANN" \
bash "$SCRIPT_DIR/run_exp_2_j10_hdp.sh" > "$LOG_DIR/j10_hdp_dfui_new_launcher.log" 2>&1 &
PID_DFUI=$!

echo "easy_ruod PID: $PID_EASY"
echo "dfui_new  PID: $PID_DFUI"
echo "Logs:"
echo "  $LOG_DIR/j10_hdp_easy_ruod_launcher.log"
echo "  $LOG_DIR/j10_hdp_dfui_new_launcher.log"

wait "$PID_EASY"
wait "$PID_DFUI"

echo "Both J10 HDP/RFTM experiments finished: $(date)"
