#!/bin/bash

# Run three J10 HDP/RFTM experiments:
#   1) easy_ruod as DFUI on GPU 2,3
#   2) dfui as DFUI on GPU 4,5
#   3) DFUI_NEW as DFUI on GPU 6,7
#
# The S2 target dataset is RUOD for all three experiments. S2 keeps the
# settings in configs/exp_2/cascade-rcnn_r50-rftm-hdp_fpn_2x_ruod_j10_s2.py;
# this launcher only injects the RFTM prior checkpoint produced by S1.

set -e

PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-work_dirs}"
LOG_DIR="${LOG_DIR:-logs}"
THRESHOLD="${THRESHOLD:-0.6}"
S1_LR="${S1_LR:-0.001}"

RUOD_IMG_DIR="${RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
RUOD_ANN="${RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/instances_train.json}"

EASY_RUOD_IMG_DIR="${EASY_RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
EASY_RUOD_ANN="${EASY_RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json}"

DFUI_IMG_DIR="${DFUI_IMG_DIR:-/media/HDD0/XCX/exp_2/dfui/images}"
DFUI_ANN="${DFUI_ANN:-/media/HDD0/XCX/exp_2/dfui/annotations/instances_trainval2017.json}"

DFUI_NEW_IMG_DIR="${DFUI_NEW_IMG_DIR:-/media/HDD0/XCX/exp_2/DFUI_NEW/images}"
DFUI_NEW_ANN="${DFUI_NEW_ANN:-/media/HDD0/XCX/exp_2/DFUI_NEW/annotations/instances_train.json}"

mkdir -p "$LOG_DIR"

launch_exp() {
    local exp_name=$1
    local gpu_ids=$2
    local port=$3
    local easy_img_dir=$4
    local easy_ann=$5
    local pid_var=$6
    local launcher_log="$LOG_DIR/${exp_name}_launcher.log"

    echo "Launching $exp_name on GPU $gpu_ids"
    (
        set -o pipefail
        PYTHON="$PYTHON" \
        WORK_DIR="$WORK_DIR" \
        LOG_DIR="$LOG_DIR" \
        EXP_NAME="$exp_name" \
        GPU_IDS="$gpu_ids" \
        NUM_GPUS="2" \
        PORT="$port" \
        THRESHOLD="$THRESHOLD" \
        S1_LR="$S1_LR" \
        RUOD_IMG_DIR="$RUOD_IMG_DIR" \
        RUOD_ANN="$RUOD_ANN" \
        EASY_IMG_DIR="$easy_img_dir" \
        EASY_ANN="$easy_ann" \
        bash run_exp_2_j10_hdp.sh 2>&1 \
            | sed "s/^/[$exp_name] /" \
            | tee "$launcher_log"
    ) &
    printf -v "$pid_var" '%s' "$!"
}

launch_exp "j10_hdp_easy_ruod" "2,3" "29523" "$EASY_RUOD_IMG_DIR" "$EASY_RUOD_ANN" PID_EASY
launch_exp "j10_hdp_dfui" "4,5" "29545" "$DFUI_IMG_DIR" "$DFUI_ANN" PID_DFUI
launch_exp "j10_hdp_dfui_new" "6,7" "29567" "$DFUI_NEW_IMG_DIR" "$DFUI_NEW_ANN" PID_DFUI_NEW

echo "easy_ruod PID: $PID_EASY"
echo "dfui      PID: $PID_DFUI"
echo "dfui_new  PID: $PID_DFUI_NEW"
echo "Logs:"
echo "  $LOG_DIR/j10_hdp_easy_ruod_launcher.log"
echo "  $LOG_DIR/j10_hdp_dfui_launcher.log"
echo "  $LOG_DIR/j10_hdp_dfui_new_launcher.log"

wait "$PID_EASY"
wait "$PID_DFUI"
wait "$PID_DFUI_NEW"

echo "All three J10 HDP/RFTM experiments finished: $(date)"
