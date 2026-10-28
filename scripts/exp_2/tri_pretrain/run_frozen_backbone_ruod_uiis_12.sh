#!/usr/bin/env bash
# Train the six frozen-backbone RUOD experiments on GPUs 4,5 and the six
# frozen-backbone UIIS10K experiments on GPUs 6,7.
#
# Each experiment keeps the established Cascade R-CNN or Mask R-CNN config,
# uses a fresh task head, freezes the complete DINO backbone, and runs for 24e.
# The RUOD and UIIS queues run concurrently; each queue is serial.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

RUOD_GPUS="${RUOD_GPUS:-4,5}"
UIIS_GPUS="${UIIS_GPUS:-6,7}"
RUOD_PORT="${RUOD_PORT:-29541}"
UIIS_PORT="${UIIS_PORT:-29551}"
DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
RUOD_ROOT="${RUOD_ROOT:-$DATA_ROOT/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-$DATA_ROOT/UIIS10K/coco}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/frozen_backbone_ruod_uiis_$(date +%Y%m%d_%H%M%S)}"
MAX_EPOCHS="${MAX_EPOCHS:-24}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-3}"
AUTO_OCCUPY="${AUTO_OCCUPY:-1}"
OCCUPY_GPU_IDS="${OCCUPY_GPU_IDS:-4,5,6,7}"
OCCUPY_MEMORY_GB="${OCCUPY_MEMORY_GB:-20}"
OCCUPY_INTERVAL="${OCCUPY_INTERVAL:-60}"
CHECK_ONLY="${CHECK_ONLY:-0}"

RUOD_R50_CONFIG="${RUOD_R50_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py}"
RUOD_VITS_CONFIG="${RUOD_VITS_CONFIG:-configs/exp_2/cascade-rcnn_vit-small_dino-realuw_fpn_2x_ruod_j14.py}"
UIIS_R50_CONFIG="${UIIS_R50_CONFIG:-configs/exp_2/mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py}"
UIIS_VITS_CONFIG="${UIIS_VITS_CONFIG:-configs/exp_2/mask-rcnn_vit-small_dino-realuw_fpn_2x_uiis10k_j14_mask.py}"

IMAGENET_R50_CKPT="${IMAGENET_R50_CKPT:-$PRETRAIN_ROOT/ImageNet/DINO_ResNet50_100e/checkpoint.pth}"
IMAGENET_VITS_CKPT="${IMAGENET_VITS_CKPT:-$PRETRAIN_ROOT/ImageNet/DINO_ViTS_100e/checkpoint.pth}"
REALUW_R50_CKPT="${REALUW_R50_CKPT:-$PRETRAIN_ROOT/RealUW/DINO_ResNet50_100e/checkpoint.pth}"
REALUW_VITS_CKPT="${REALUW_VITS_CKPT:-$PRETRAIN_ROOT/RealUW/DINO_ViTS_100e/checkpoint.pth}"
SYNTHETIC5_R50_CKPT="${SYNTHETIC5_R50_CKPT:-$PRETRAIN_ROOT/Synthetic5Merged_CorrectedWaterGAN/DINO_ResNet50_100e/checkpoint.pth}"
SYNTHETIC5_VITS_CKPT="${SYNTHETIC5_VITS_CKPT:-$PRETRAIN_ROOT/Synthetic5Merged_CorrectedWaterGAN/DINO_ViTS_100e/checkpoint.pth}"

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
MASTER_LOG="$OUTPUT_ROOT/logs/launcher.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo "ERROR: $*" >&2
    exit 1
}

count_gpus() {
    awk -F, '{print NF}' <<< "$1"
}

check_file() {
    [ -s "$1" ] || die "missing or empty file: $1"
}

check_config() {
    if [[ "$1" = /* ]]; then
        check_file "$1"
    else
        check_file "$REPO_ROOT/$1"
    fi
}

check_dataset() {
    local root="$1"
    check_file "$root/annotations/instances_train.json"
    check_file "$root/annotations/instances_val.json"
}

check_dataset "$RUOD_ROOT"
check_dataset "$UIIS_ROOT"
check_config "$RUOD_R50_CONFIG"
check_config "$RUOD_VITS_CONFIG"
check_config "$UIIS_R50_CONFIG"
check_config "$UIIS_VITS_CONFIG"

declare -A R50_CKPTS=(
    [imagenet]="$IMAGENET_R50_CKPT"
    [realuw]="$REALUW_R50_CKPT"
    [synthetic5]="$SYNTHETIC5_R50_CKPT"
)
declare -A VITS_CKPTS=(
    [imagenet]="$IMAGENET_VITS_CKPT"
    [realuw]="$REALUW_VITS_CKPT"
    [synthetic5]="$SYNTHETIC5_VITS_CKPT"
)

for source in imagenet realuw synthetic5; do
    check_file "${R50_CKPTS[$source]}"
    check_file "${VITS_CKPTS[$source]}"
done

check_gpu_group() {
    local group="$1"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available"
    local gpu
    for gpu in ${group//,/ }; do
        nvidia-smi -i "$gpu" --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader || die "GPU $gpu is not available"
    done
}

check_gpu_group "$RUOD_GPUS"
check_gpu_group "$UIIS_GPUS"

if [ "$CHECK_ONLY" = 1 ]; then
    echo "CHECK_ONLY=1: paths, configs, checkpoints, datasets, and GPUs passed validation."
    echo "No training, testing, resume, or GPU occupancy was started."
    exit 0
fi

cat > "$OUTPUT_ROOT/run_manifest.tsv" <<EOF
field	value
ruod_gpus	$RUOD_GPUS
uiis_gpus	$UIIS_GPUS
ruod_root	$RUOD_ROOT
uiis_root	$UIIS_ROOT
max_epochs	$MAX_EPOCHS
run_test	$RUN_TEST
backbone_policy	frozen_all_stages
ruod_config_r50	$RUOD_R50_CONFIG
ruod_config_vits	$RUOD_VITS_CONFIG
uiis_config_r50	$UIIS_R50_CONFIG
uiis_config_vits	$UIIS_VITS_CONFIG
EOF

run_task() {
    local dataset="$1"
    local source="$2"
    local arch="$3"
    local config="$4"
    local checkpoint="$5"
    local gpu_ids="$6"
    local port="$7"
    local data_root="$8"
    local save_best="$9"
    local freeze_stages="${10}"
    local task_name="${dataset}_${source}_${arch}"
    local work_dir="$OUTPUT_ROOT/work_dirs/$task_name"
    local log_file="$OUTPUT_ROOT/logs/$task_name.log"
    local complete_marker="$OUTPUT_ROOT/status/$task_name.complete"
    local num_gpus
    local -a resume_args=()

    if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$complete_marker" ]; then
        echo "SKIP completed task=$task_name"
        return 0
    fi

    num_gpus="$(count_gpus "$gpu_ids")"
    mkdir -p "$work_dir"

    if [ -s "$work_dir/latest.pth" ] || [ -s "$work_dir/last_checkpoint" ]; then
        resume_args+=(--resume)
        echo "RESUME task=$task_name from existing latest checkpoint in $work_dir"
    fi

    echo "============================================================"
    echo "START task=$task_name"
    echo "config=$config"
    echo "checkpoint=$checkpoint"
    echo "gpus=$gpu_ids"
    echo "work_dir=$work_dir"
    echo "log=$log_file"
    echo "============================================================"

    local -a cfg_options=(
        "model.backbone.init_cfg.checkpoint=$checkpoint"
        "model.backbone.frozen_stages=$freeze_stages"
        "train_cfg.max_epochs=$MAX_EPOCHS"
        "train_cfg.val_interval=1"
        "default_hooks.checkpoint.max_keep_ckpts=$MAX_KEEP_CKPTS"
        "default_hooks.checkpoint.save_best=$save_best"
        "train_dataloader.dataset.data_root=$data_root"
        "train_dataloader.dataset.data_prefix.img=train/"
        "train_dataloader.dataset.ann_file=$data_root/annotations/instances_train.json"
        "val_dataloader.dataset.data_root=$data_root"
        "val_dataloader.dataset.data_prefix.img=val/"
        "val_dataloader.dataset.ann_file=$data_root/annotations/instances_val.json"
        "test_dataloader.dataset.data_root=$data_root"
        "test_dataloader.dataset.data_prefix.img=val/"
        "test_dataloader.dataset.ann_file=$data_root/annotations/instances_val.json"
        "val_evaluator.ann_file=$data_root/annotations/instances_val.json"
        "test_evaluator.ann_file=$data_root/annotations/instances_val.json"
    )

    # The ViT configs inherit a 100e schedule. Keep the established 24e
    # milestones when the inherited scheduler exposes the standard entry.
    if [ "$arch" = "vits" ]; then
        cfg_options+=(
            'param_scheduler.1.end=24'
            'param_scheduler.1.milestones=[16,22]'
        )
    else
        cfg_options+=(
            'model.backbone.norm_eval=True'
        )
    fi

    (
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
        export PORT="$port"
        export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

        bash tools/dist_train.sh \
            "$config" \
            "$num_gpus" \
            --work-dir "$work_dir" \
            "${resume_args[@]}" \
            --cfg-options "${cfg_options[@]}"

        if [ "$RUN_TEST" = 1 ]; then
            local best_ckpt
            best_ckpt="$(find "$work_dir" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
            if [ -z "$best_ckpt" ]; then
                best_ckpt="$work_dir/latest.pth"
            fi
            check_file "$best_ckpt"

            bash tools/dist_test.sh \
                "$config" \
                "$best_ckpt" \
                "$num_gpus" \
                --cfg-options "${cfg_options[@]}" \
                2>&1 | tee "$OUTPUT_ROOT/logs/${task_name}_test.log"
        fi

        touch "$complete_marker"
        echo "DONE task=$task_name"
    ) 2>&1 | tee "$log_file"
}

run_ruod_queue() {
    local port="$RUOD_PORT"
    for source in imagenet realuw synthetic5; do
        run_task ruod "$source" r50 "$RUOD_R50_CONFIG" "${R50_CKPTS[$source]}" \
            "$RUOD_GPUS" "$port" "$RUOD_ROOT" coco/bbox_mAP 4
        port=$((port + 1))
        run_task ruod "$source" vits "$RUOD_VITS_CONFIG" "${VITS_CKPTS[$source]}" \
            "$RUOD_GPUS" "$port" "$RUOD_ROOT" coco/bbox_mAP 12
        port=$((port + 1))
    done
    touch "$OUTPUT_ROOT/status/ruod.complete"
}

run_uiis_queue() {
    local port="$UIIS_PORT"
    for source in imagenet realuw synthetic5; do
        run_task uiis10k "$source" r50 "$UIIS_R50_CONFIG" "${R50_CKPTS[$source]}" \
            "$UIIS_GPUS" "$port" "$UIIS_ROOT" coco/segm_mAP 4
        port=$((port + 1))
        run_task uiis10k "$source" vits "$UIIS_VITS_CONFIG" "${VITS_CKPTS[$source]}" \
            "$UIIS_GPUS" "$port" "$UIIS_ROOT" coco/segm_mAP 12
        port=$((port + 1))
    done
    touch "$OUTPUT_ROOT/status/uiis10k.complete"
}

echo "============================================================"
echo "Frozen-backbone RUOD/UIIS downstream matrix"
echo "output_root=$OUTPUT_ROOT"
echo "RUOD GPUs=$RUOD_GPUS"
echo "UIIS GPUs=$UIIS_GPUS"
echo "Each queue: ImageNet -> RealUW -> Synthetic5, R50 -> ViT-S"
echo "Each task: established task config, frozen backbone, ${MAX_EPOCHS}e"
echo "============================================================"

run_ruod_queue > "$OUTPUT_ROOT/logs/ruod_queue.log" 2>&1 &
ruod_pid=$!
run_uiis_queue > "$OUTPUT_ROOT/logs/uiis_queue.log" 2>&1 &
uiis_pid=$!

status=0
wait "$ruod_pid" || status=1
wait "$uiis_pid" || status=1

if [ "$status" -ne 0 ]; then
    echo "FAILED: inspect $OUTPUT_ROOT/logs/ruod_queue.log and $OUTPUT_ROOT/logs/uiis_queue.log" >&2
    exit "$status"
fi

touch "$OUTPUT_ROOT/status/matrix.complete"

if [ "$AUTO_OCCUPY" = 1 ]; then
    echo "All 12 experiments completed. Starting ${OCCUPY_MEMORY_GB}GiB GPU occupancy on GPUs $OCCUPY_GPU_IDS."
    OCCUPY_LOG_DIR="$OUTPUT_ROOT/logs/gpu_occupy" \
        nohup bash scripts/gpu_auto_occupy.sh \
            "$OCCUPY_GPU_IDS" \
            "$OCCUPY_INTERVAL" \
            "$OCCUPY_MEMORY_GB" \
            > "$OUTPUT_ROOT/logs/gpu_auto_occupy.log" 2>&1 < /dev/null &
    echo "$!" > "$OUTPUT_ROOT/status/gpu_auto_occupy.pid"
    touch "$OUTPUT_ROOT/status/gpu_occupy.started"
fi

echo "COMPLETE output_root=$OUTPUT_ROOT"
