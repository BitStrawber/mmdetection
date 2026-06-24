#!/bin/bash
set -euo pipefail

# Compare three DINO backbones on downstream tasks:
#   1. official_rn50: ImageNet DINO ResNet-50 100e official checkpoint
#   2. realuw_r50_ssd: RealUW DINO ResNet-50 100e checkpoint
#   3. realuw_vits: RealUW DINO ViT-Small/16 100e checkpoint
#
# Stage order:
#   1. Download/convert required backbone checkpoints.
#   2. Run RUOD Cascade R-CNN tasks in parallel on GPU groups 01/23/45.
#   3. After all Cascade tasks finish, run UIIS10K Mask R-CNN tasks in parallel
#      on the same GPU groups.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

TASKS="${TASKS:-official_rn50 realuw_r50_ssd realuw_vits}"
GPU_GROUPS="${GPU_GROUPS:-0,1 2,3 4,5}"

PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECKPOINT_SAVE_BEST="${CHECKPOINT_SAVE_BEST:-coco/bbox_mAP}"
RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
RUN_TEST="${RUN_TEST:-1}"

OFFICIAL_URL="${OFFICIAL_URL:-https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_checkpoint.pth}"
OFFICIAL_RAW_CKPT="${OFFICIAL_RAW_CKPT:-$PRETRAIN_DIR/dino_rn50_checkpoint.pth}"
OFFICIAL_BACKBONE_CKPT="${OFFICIAL_BACKBONE_CKPT:-$PRETRAIN_DIR/dino_rn50_official_100e_backbone.pth}"

REALUW_R50_HDD_S1_CKPT="${REALUW_R50_HDD_S1_CKPT:-$WORK_ROOT/j7_realuw_dino_resnet50/checkpoint.pth}"
REALUW_R50_HDD_BACKBONE_CKPT="${REALUW_R50_HDD_BACKBONE_CKPT:-$PRETRAIN_DIR/j7_realuw_dino_resnet50_backbone.pth}"

REALUW_R50_SSD_S1_CKPT="${REALUW_R50_SSD_S1_CKPT:-$WORK_ROOT/j7_realuw_dino_resnet50_ssd100e/checkpoint.pth}"
REALUW_R50_SSD_BACKBONE_CKPT="${REALUW_R50_SSD_BACKBONE_CKPT:-$PRETRAIN_DIR/j7_realuw_dino_resnet50_ssd100e_backbone.pth}"

REALUW_VITS_S1_CKPT="${REALUW_VITS_S1_CKPT:-$WORK_ROOT/j14_realuw_dino_vits_100e/checkpoint.pth}"
REALUW_VITS_BACKBONE_CKPT="${REALUW_VITS_BACKBONE_CKPT:-$PRETRAIN_DIR/j14_realuw_dino_vits_100e_backbone.pth}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

CASCADE_BASE_PORT="${CASCADE_BASE_PORT:-29930}"
MASK_BASE_PORT="${MASK_BASE_PORT:-29940}"

mkdir -p "$PRETRAIN_DIR" "$WORK_ROOT" "$LOG_DIR"

read -r -a task_array <<< "$TASKS"
read -r -a gpu_group_array <<< "$GPU_GROUPS"

if [ "${#task_array[@]}" -ne "${#gpu_group_array[@]}" ]; then
    echo "Error: TASKS count (${#task_array[@]}) must equal GPU_GROUPS count (${#gpu_group_array[@]})."
    echo "TASKS=$TASKS"
    echo "GPU_GROUPS=$GPU_GROUPS"
    exit 1
fi

download_official() {
    if [ "$RUN_DOWNLOAD" != "1" ]; then
        echo "RUN_DOWNLOAD=$RUN_DOWNLOAD, skip official checkpoint download."
        return
    fi
    if [ -f "$OFFICIAL_RAW_CKPT" ]; then
        echo "Official raw checkpoint exists: $OFFICIAL_RAW_CKPT"
        return
    fi

    echo "Download official DINO ResNet-50 checkpoint:"
    echo "  url: $OFFICIAL_URL"
    echo "  out: $OFFICIAL_RAW_CKPT"
    if command -v wget >/dev/null 2>&1; then
        wget -c --show-progress -O "$OFFICIAL_RAW_CKPT" "$OFFICIAL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$OFFICIAL_RAW_CKPT" "$OFFICIAL_URL"
    else
        python - <<PY
from urllib.request import urlretrieve
urlretrieve("$OFFICIAL_URL", "$OFFICIAL_RAW_CKPT")
PY
    fi
}

convert_if_needed() {
    local name="$1"
    local input_ckpt="$2"
    local output_ckpt="$3"
    local source="${4:-teacher}"

    if [ ! -f "$input_ckpt" ]; then
        echo "Error: missing input checkpoint for $name: $input_ckpt"
        exit 1
    fi
    if [ "$FORCE_CONVERT" != "1" ] && [ -f "$output_ckpt" ]; then
        echo "Converted checkpoint exists for $name: $output_ckpt"
        return
    fi

    mkdir -p "$(dirname "$output_ckpt")"
    echo "Convert $name checkpoint:"
    echo "  input : $input_ckpt"
    echo "  output: $output_ckpt"
    echo "  source: $source"
    python tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$input_ckpt" \
        --source "$source" \
        --out "$output_ckpt" \
        2>&1 | tee "$LOG_DIR/${name}_convert_backbone.log"
}

prepare_checkpoints() {
    if [[ " $TASKS " == *" official_rn50 "* ]]; then
        download_official
    fi

    if [ "$RUN_CONVERT" != "1" ]; then
        echo "RUN_CONVERT=$RUN_CONVERT, skip checkpoint conversion."
        return
    fi

    if [[ " $TASKS " == *" official_rn50 "* ]]; then
        convert_if_needed official_rn50 "$OFFICIAL_RAW_CKPT" "$OFFICIAL_BACKBONE_CKPT" teacher
    fi

    if [[ " $TASKS " == *" realuw_r50_hdd "* ]]; then
        convert_if_needed realuw_r50_hdd "$REALUW_R50_HDD_S1_CKPT" "$REALUW_R50_HDD_BACKBONE_CKPT" teacher
    fi
    if [[ " $TASKS " == *" realuw_r50_ssd "* ]]; then
        convert_if_needed realuw_r50_ssd "$REALUW_R50_SSD_S1_CKPT" "$REALUW_R50_SSD_BACKBONE_CKPT" teacher
    fi
    if [[ " $TASKS " == *" realuw_vits "* ]]; then
        convert_if_needed realuw_vits "$REALUW_VITS_S1_CKPT" "$REALUW_VITS_BACKBONE_CKPT" teacher
    fi
}

query_gpu_state() {
    local gpu_id="$1"
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v id="$gpu_id" '
            {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                gsub(/[[:space:]]/, "", $3)
                if ($1 == id) {
                    print $2, $3
                    exit
                }
            }'
}

wait_msg() {
    if [ -w /dev/tty ]; then
        echo "$*" > /dev/tty
    else
        echo "$*" >&2
    fi
}

wait_for_gpu_group() {
    local gpu_ids="$1"
    local label="$2"

    if [ "$WAIT_FOR_GPUS" != "1" ]; then
        wait_msg "WAIT_FOR_GPUS=$WAIT_FOR_GPUS, skip GPU idle waiting for $label."
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        wait_msg "Warning: nvidia-smi not found, skip GPU idle waiting for $label."
        return
    fi

    local idle_rounds=0
    local gpu_array=()
    IFS=',' read -r -a gpu_array <<< "$gpu_ids"

    wait_msg "Waiting for GPU group [$gpu_ids] before $label..."
    wait_msg "Idle rule: memory.used <= ${GPU_MAX_MEM_MB}MB and utilization.gpu <= ${GPU_MAX_UTIL}% for ${GPU_IDLE_CHECKS} consecutive check(s)."

    while true; do
        local all_idle=1
        local status_parts=()
        for gpu in "${gpu_array[@]}"; do
            gpu="${gpu//[[:space:]]/}"
            [ -z "$gpu" ] && continue
            local state=""
            state=$(query_gpu_state "$gpu" || true)
            if [ -z "$state" ]; then
                status_parts+=("gpu${gpu}=not_found")
                all_idle=0
                continue
            fi
            local mem_used util
            read -r mem_used util <<< "$state"
            status_parts+=("gpu${gpu}=mem:${mem_used}MB,util:${util}%")
            if [ "$mem_used" -gt "$GPU_MAX_MEM_MB" ] || [ "$util" -gt "$GPU_MAX_UTIL" ]; then
                all_idle=0
            fi
        done

        wait_msg "GPU status: ${status_parts[*]}"
        if [ "$all_idle" -eq 1 ]; then
            idle_rounds=$((idle_rounds + 1))
            wait_msg "Idle check ${idle_rounds}/${GPU_IDLE_CHECKS} passed."
            if [ "$idle_rounds" -ge "$GPU_IDLE_CHECKS" ]; then
                wait_msg "GPU group [$gpu_ids] is idle. Start $label."
                return
            fi
        else
            idle_rounds=0
            wait_msg "GPU group [$gpu_ids] is busy. Recheck after ${GPU_WAIT_INTERVAL}s."
        fi
        sleep "$GPU_WAIT_INTERVAL"
    done
}

set_task_info() {
    local task="$1"
    case "$task" in
        official_rn50)
            EXP_NAME="official_dino_rn50_100e"
            PRETRAIN_CKPT="$OFFICIAL_BACKBONE_CKPT"
            CASCADE_CONFIG="configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_ruod.py"
            MASK_CONFIG="configs/exp_2/mask-rcnn_r50_dino-official_fpn_2x_uiis10k_mask.py"
            ;;
        realuw_r50_hdd)
            EXP_NAME="realuw_dino_rn50_hdd100e"
            PRETRAIN_CKPT="$REALUW_R50_HDD_BACKBONE_CKPT"
            CASCADE_CONFIG="configs/exp_2/cascade-rcnn_r50_dino-realuw_fpn_2x_ruod_j7.py"
            MASK_CONFIG="configs/exp_2/mask-rcnn_r50_dino-realuw_fpn_2x_uiis10k_j7_mask.py"
            ;;
        realuw_r50_ssd)
            EXP_NAME="realuw_dino_rn50_ssd100e"
            PRETRAIN_CKPT="$REALUW_R50_SSD_BACKBONE_CKPT"
            CASCADE_CONFIG="configs/exp_2/cascade-rcnn_r50_dino-realuw-ssd_fpn_2x_ruod_j7.py"
            MASK_CONFIG="configs/exp_2/mask-rcnn_r50_dino-realuw-ssd_fpn_2x_uiis10k_j7_mask.py"
            ;;
        realuw_vits)
            EXP_NAME="realuw_dino_vits_100e"
            PRETRAIN_CKPT="$REALUW_VITS_BACKBONE_CKPT"
            CASCADE_CONFIG="configs/exp_2/cascade-rcnn_vit-small_dino-realuw_fpn_2x_ruod_j14.py"
            MASK_CONFIG="configs/exp_2/mask-rcnn_vit-small_dino-realuw_fpn_2x_uiis10k_j14_mask.py"
            ;;
        *)
            echo "Error: unsupported TASK=$task"
            exit 1
            ;;
    esac
}

run_one() {
    local task="$1"
    local stage="$2"
    local config="$3"
    local gpu_ids="$4"
    local port="$5"
    local exp_name="$6"
    local pretrain_ckpt="$7"
    local num_gpus
    local work_dir="$WORK_ROOT/${exp_name}_${stage}"
    local log_file="$LOG_DIR/${exp_name}_${stage}.log"

    num_gpus=$(awk -F, '{print NF}' <<< "$gpu_ids")

    if [ ! -f "$pretrain_ckpt" ]; then
        echo "Error: pretrain checkpoint not found for $task: $pretrain_ckpt"
        exit 1
    fi

    (
        wait_for_gpu_group "$gpu_ids" "$exp_name $stage"

        mkdir -p "$work_dir"
        export PORT="$port"
        export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

        echo "========================================="
        echo "$exp_name $stage"
        echo "task: $task"
        echo "config: $config"
        echo "pretrain: $pretrain_ckpt"
        echo "gpu_ids: $gpu_ids"
        echo "port: $port"
        echo "work_dir: $work_dir"
        echo "log: $log_file"
        echo "========================================="

        CUDA_VISIBLE_DEVICES="$gpu_ids" bash tools/dist_train.sh \
            "$config" \
            "$num_gpus" \
            --work-dir "$work_dir" \
            --cfg-options \
                model.backbone.init_cfg.checkpoint="$pretrain_ckpt" \
                default_hooks.checkpoint.save_best="$CHECKPOINT_SAVE_BEST" \
                default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
            2>&1 | tee "$log_file"

        if [ "$RUN_TEST" = "1" ]; then
            local best_ckpt
            local test_log="$LOG_DIR/${exp_name}_${stage}_test.log"
            best_ckpt=$(ls -t "$work_dir"/best_*.pth 2>/dev/null | head -1 || true)
            [ -z "$best_ckpt" ] && best_ckpt="$work_dir/latest.pth"
            if [ ! -f "$best_ckpt" ]; then
                echo "Error: no checkpoint found for test in $work_dir"
                exit 1
            fi
            echo "Test $exp_name $stage with $best_ckpt"
            CUDA_VISIBLE_DEVICES="$gpu_ids" bash tools/dist_test.sh \
                "$config" \
                "$best_ckpt" \
                "$num_gpus" \
                --cfg-options model.backbone.init_cfg.checkpoint="$pretrain_ckpt" \
                2>&1 | tee "$test_log"
        fi
    ) &
}

run_stage() {
    local stage="$1"
    local base_port="$2"
    local pids=()
    local status=0
    local i=0

    echo "========================================="
    echo "Run $stage stage"
    echo "TASKS: $TASKS"
    echo "GPU_GROUPS: $GPU_GROUPS"
    echo "========================================="

    for task in "${task_array[@]}"; do
        local gpu_group="${gpu_group_array[$i]}"
        local port=$((base_port + i))
        set_task_info "$task"
        local config="$CASCADE_CONFIG"
        if [ "$stage" = "mask" ]; then
            config="$MASK_CONFIG"
        fi
        echo "Launch $stage: task=$task exp=$EXP_NAME gpu=$gpu_group port=$port"
        run_one "$task" "$stage" "$config" "$gpu_group" "$port" "$EXP_NAME" "$PRETRAIN_CKPT"
        pids+=("$!")
        i=$((i + 1))
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "Error: $stage stage failed."
        exit "$status"
    fi
    echo "$stage stage finished."
}

prepare_checkpoints
run_stage cascade "$CASCADE_BASE_PORT"
run_stage mask "$MASK_BASE_PORT"

echo "========================================="
echo "DINO three-backbone downstream pipeline finished: $(date)"
echo "logs: $LOG_DIR"
echo "work dirs: $WORK_ROOT"
echo "========================================="
