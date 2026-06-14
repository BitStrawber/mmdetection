#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Auto tuning for the USOD-expanded ResNet/Cascade J10 route.
#
# Each round runs three experiments in parallel, waits for all of them, parses
# RUOD S2 best mAP, then derives the next round from the best setting.
#
# Round 1: epoch/milestone sweep at lr=0.00375.
# Round 2: lr sweep, epoch/milestone adapted from Round 1 best.
# Round 3: frozen_stages / weight_decay local sweep from Round 2 best.

cd "$REPO_ROOT"

BASE_WORK_DIR="${BASE_WORK_DIR:-work_dirs/j10_usod_auto_tuning}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs/j10_usod_auto_tuning}"
SUMMARY_FILE="${SUMMARY_FILE:-$BASE_LOG_DIR/summary.tsv}"

GPU_GROUPS_STR="${GPU_GROUPS:-2,3 4,5 6,7}"
PORTS_STR="${PORTS:-29701 29702 29703}"
read -r -a GPU_GROUPS <<< "$GPU_GROUPS_STR"
read -r -a PORTS <<< "$PORTS_STR"

S1_CONFIG="${S1_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py}"
S2_CONFIG="${S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}"
MERGED_ROOT="${MERGED_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY}"

MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

# Control how far the script runs. Default: all three rounds.
MAX_ROUND="${MAX_ROUND:-3}"

mkdir -p "$BASE_LOG_DIR" "$BASE_WORK_DIR"

if [ ! -f "$MERGED_ROOT/annotations/instances_train.json" ] || \
   [ ! -f "$MERGED_ROOT/annotations/instances_val.json" ]; then
    echo "Merged USOD-expanded DFUI source is missing: $MERGED_ROOT"
    echo "Run first:"
    echo "  bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh"
    exit 1
fi

if [ "${#GPU_GROUPS[@]}" -lt 3 ] || [ "${#PORTS[@]}" -lt 3 ]; then
    echo "Error: need three GPU groups and three ports."
    echo "GPU_GROUPS=${GPU_GROUPS[*]}"
    echo "PORTS=${PORTS[*]}"
    exit 1
fi

if [ ! -f "$SUMMARY_FILE" ]; then
    printf "round\texp\tgpu\tport\tfrozen\tlr\tepochs\tmilestones\twd\ts1_best_epoch\ts1_best_map\ts2_best_epoch\ts2_best_map\ts2_ap50\ts2_ap75\ts2_small\ts2_medium\ts2_large\n" \
        > "$SUMMARY_FILE"
fi

sanitize_token() {
    echo "$1" | tr -d '[]' | tr ',' '_'
}

best_line_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "NA NA NA NA NA NA NA"
        return
    fi

    local line
    line=$(grep -a "coco/bbox_mAP:" "$log_file" 2>/dev/null | \
        sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*coco\/bbox_mAP_50: ([0-9.]+).*coco\/bbox_mAP_75: ([0-9.]+).*coco\/bbox_mAP_s: ([0-9.]+).*coco\/bbox_mAP_m: ([0-9.]+).*coco\/bbox_mAP_l: ([0-9.]+).*/\1 \2 \3 \4 \5 \6 \7/' | \
        sort -k2,2nr | head -n 1 || true)
    if [ -z "$line" ]; then
        echo "NA NA NA NA NA NA NA"
    else
        echo "$line"
    fi
}

s1_best_line_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "NA NA"
        return
    fi

    local line
    line=$(grep -a "coco/bbox_mAP:" "$log_file" 2>/dev/null | \
        sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*/\1 \2/' | \
        sort -k2,2nr | head -n 1 || true)
    if [ -z "$line" ]; then
        echo "NA NA"
    else
        echo "$line"
    fi
}

record_result() {
    local round="$1"
    local exp="$2"
    local gpu="$3"
    local port="$4"
    local frozen="$5"
    local lr="$6"
    local epochs="$7"
    local milestones="$8"
    local wd="$9"
    local log_dir="${10}"

    local s1_epoch="NA"
    local s1_map="NA"
    local s2_epoch="NA"
    local s2_map="NA"
    local s2_ap50="NA"
    local s2_ap75="NA"
    local s2_small="NA"
    local s2_medium="NA"
    local s2_large="NA"

    read -r s1_epoch s1_map <<< "$(s1_best_line_from_log "$log_dir/${exp}_s1.log")"
    read -r s2_epoch s2_map s2_ap50 s2_ap75 s2_small s2_medium s2_large <<< "$(best_line_from_log "$log_dir/${exp}_s2.log")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$round" "$exp" "$gpu" "$port" "$frozen" "$lr" "$epochs" "$milestones" "$wd" \
        "$s1_epoch" "$s1_map" "$s2_epoch" "$s2_map" "$s2_ap50" "$s2_ap75" "$s2_small" "$s2_medium" "$s2_large" \
        >> "$SUMMARY_FILE"

    echo "$s2_map $exp $frozen $lr $epochs $milestones $wd"
}

launch_experiment() {
    local round="$1"
    local exp="$2"
    local gpu="$3"
    local port="$4"
    local frozen="$5"
    local lr="$6"
    local epochs="$7"
    local milestones="$8"
    local wd="$9"
    local round_work_dir="$BASE_WORK_DIR/round${round}"
    local round_log_dir="$BASE_LOG_DIR/round${round}"

    mkdir -p "$round_work_dir" "$round_log_dir"

    echo "[$exp] launch on GPU $gpu, port $port"
    (
        WORK_DIR="$round_work_dir" \
        LOG_DIR="$round_log_dir" \
        EXP_NAME="$exp" \
        GPU_IDS="$gpu" \
        PORT="$port" \
        S1_CONFIG="$S1_CONFIG" \
        S2_CONFIG="$S2_CONFIG" \
        FROZEN_STAGES="$frozen" \
        S1_LR="$lr" \
        S1_EPOCHS="$epochs" \
        S1_MILESTONES="$milestones" \
        S1_WEIGHT_DECAY="$wd" \
        MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
        RUN_S2=1 \
        WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
        GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        bash "$REPO_ROOT/scripts/exp_2/j10/run_exp_2_j10_scheme_c.sh" \
            2>&1 | tee "$round_log_dir/${exp}_launcher.log"
    ) &
}

run_round() {
    local round="$1"
    shift
    local -n names_ref="$1"
    local -n frozen_ref="$2"
    local -n lr_ref="$3"
    local -n epochs_ref="$4"
    local -n milestones_ref="$5"
    local -n wd_ref="$6"

    local pids=()
    local exp_names=()

    echo "========================================="
    echo "Round $round start"
    echo "========================================="

    for i in 0 1 2; do
        launch_experiment \
            "$round" \
            "${names_ref[$i]}" \
            "${GPU_GROUPS[$i]}" \
            "${PORTS[$i]}" \
            "${frozen_ref[$i]}" \
            "${lr_ref[$i]}" \
            "${epochs_ref[$i]}" \
            "${milestones_ref[$i]}" \
            "${wd_ref[$i]}"
        pids+=($!)
        exp_names+=("${names_ref[$i]}")
    done

    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[${exp_names[$i]}] finished successfully."
        else
            echo "[${exp_names[$i]}] failed."
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "Round $round failed. Stop auto tuning."
        exit 1
    fi

    echo "========================================="
    echo "Round $round results"
    echo "========================================="

    local best_map="-1"
    local best_exp=""
    local best_frozen=""
    local best_lr=""
    local best_epochs=""
    local best_milestones=""
    local best_wd=""
    local round_log_dir="$BASE_LOG_DIR/round${round}"
    local result_line

    for i in 0 1 2; do
        result_line=$(record_result \
            "$round" \
            "${names_ref[$i]}" \
            "${GPU_GROUPS[$i]}" \
            "${PORTS[$i]}" \
            "${frozen_ref[$i]}" \
            "${lr_ref[$i]}" \
            "${epochs_ref[$i]}" \
            "${milestones_ref[$i]}" \
            "${wd_ref[$i]}" \
            "$round_log_dir")
        echo "$result_line"

        read -r map exp frozen lr epochs milestones wd <<< "$result_line"
        if [ "$map" != "NA" ]; then
            if awk "BEGIN {exit !($map > $best_map)}"; then
                best_map="$map"
                best_exp="$exp"
                best_frozen="$frozen"
                best_lr="$lr"
                best_epochs="$epochs"
                best_milestones="$milestones"
                best_wd="$wd"
            fi
        fi
    done

    if [ -z "$best_exp" ]; then
        echo "Error: could not find a valid best result in round $round."
        exit 1
    fi

    BEST_EXP="$best_exp"
    BEST_MAP="$best_map"
    BEST_FROZEN="$best_frozen"
    BEST_LR="$best_lr"
    BEST_EPOCHS="$best_epochs"
    BEST_MILESTONES="$best_milestones"
    BEST_WD="$best_wd"

    echo "Round $round best: $BEST_EXP mAP=$BEST_MAP frozen=$BEST_FROZEN lr=$BEST_LR epochs=$BEST_EPOCHS milestones=$BEST_MILESTONES wd=$BEST_WD"
}

echo "========================================="
echo "USOD RCNN auto tuning"
echo "========================================="
echo "BASE_WORK_DIR: $BASE_WORK_DIR"
echo "BASE_LOG_DIR: $BASE_LOG_DIR"
echo "SUMMARY_FILE: $SUMMARY_FILE"
echo "GPU_GROUPS: ${GPU_GROUPS[*]}"
echo "PORTS: ${PORTS[*]}"
echo "MAX_ROUND: $MAX_ROUND"
echo "========================================="

R1_NAMES=(
    j10_usod_r1_f1_lr00375_e36_ms24_32
    j10_usod_r1_f1_lr00375_e48_ms28_40
    j10_usod_r1_f1_lr00375_e60_ms36_52
)
R1_FROZEN=(1 1 1)
R1_LR=(0.00375 0.00375 0.00375)
R1_EPOCHS=(36 48 60)
R1_MILESTONES=("[24,32]" "[28,40]" "[36,52]")
R1_WD=(0.0001 0.0001 0.0001)

run_round 1 R1_NAMES R1_FROZEN R1_LR R1_EPOCHS R1_MILESTONES R1_WD

if [ "$MAX_ROUND" -le 1 ]; then
    echo "MAX_ROUND=$MAX_ROUND, stop after Round 1."
    exit 0
fi

case "$BEST_EPOCHS" in
    36)
        R2_EPOCHS=(48 36 30)
        R2_MILESTONES=("[32,44]" "[24,32]" "[20,26]")
        ;;
    48)
        R2_EPOCHS=(60 48 36)
        R2_MILESTONES=("[40,52]" "[28,40]" "[24,32]")
        ;;
    60)
        R2_EPOCHS=(72 60 48)
        R2_MILESTONES=("[48,64]" "[36,52]" "[32,44]")
        ;;
    *)
        R2_EPOCHS=("$BEST_EPOCHS" "$BEST_EPOCHS" "$BEST_EPOCHS")
        R2_MILESTONES=("$BEST_MILESTONES" "$BEST_MILESTONES" "$BEST_MILESTONES")
        ;;
esac

R2_NAMES=(
    "j10_usod_r2_f${BEST_FROZEN}_lr001875_e${R2_EPOCHS[0]}_ms$(sanitize_token "${R2_MILESTONES[0]}")"
    "j10_usod_r2_f${BEST_FROZEN}_lr00375_e${R2_EPOCHS[1]}_ms$(sanitize_token "${R2_MILESTONES[1]}")"
    "j10_usod_r2_f${BEST_FROZEN}_lr0075_e${R2_EPOCHS[2]}_ms$(sanitize_token "${R2_MILESTONES[2]}")"
)
R2_FROZEN=("$BEST_FROZEN" "$BEST_FROZEN" "$BEST_FROZEN")
R2_LR=(0.001875 0.00375 0.0075)
R2_WD=("$BEST_WD" "$BEST_WD" "$BEST_WD")

run_round 2 R2_NAMES R2_FROZEN R2_LR R2_EPOCHS R2_MILESTONES R2_WD

if [ "$MAX_ROUND" -le 2 ]; then
    echo "MAX_ROUND=$MAX_ROUND, stop after Round 2."
    exit 0
fi

R3_NAMES=(
    "j10_usod_r3_f0_lr${BEST_LR}_e${BEST_EPOCHS}_wd0001"
    "j10_usod_r3_f1_lr${BEST_LR}_e${BEST_EPOCHS}_wd00005"
    "j10_usod_r3_f1_lr${BEST_LR}_e${BEST_EPOCHS}_wd0002"
)
R3_FROZEN=(0 1 1)
R3_LR=("$BEST_LR" "$BEST_LR" "$BEST_LR")
R3_EPOCHS=("$BEST_EPOCHS" "$BEST_EPOCHS" "$BEST_EPOCHS")
R3_MILESTONES=("$BEST_MILESTONES" "$BEST_MILESTONES" "$BEST_MILESTONES")
R3_WD=(0.0001 0.00005 0.0002)

run_round 3 R3_NAMES R3_FROZEN R3_LR R3_EPOCHS R3_MILESTONES R3_WD

echo "========================================="
echo "Auto tuning finished"
echo "Summary: $SUMMARY_FILE"
echo "Best final round: $BEST_EXP mAP=$BEST_MAP"
echo "========================================="
