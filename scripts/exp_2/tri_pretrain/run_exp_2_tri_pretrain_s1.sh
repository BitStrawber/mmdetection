#!/bin/bash
set -euo pipefail

# Run one Tri-pretrain S1 self-supervised pretraining task on full RealUW.
#
# Usage:
#   EXP_ID=j11 GPU_IDS=0,1,2,3,4,5,6,7 \
#     bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
#
# EXP_ID:
#   j6  = keyu-tian/SparK ResNet-50
#   j7  = facebookresearch/dino ResNet-50
#   j11 = MMPreTrain MAE ViT-Base
#   j12 = MMPreTrain SimMIM/MixMIM SwinV2/Swin-Base
#   j13 = MMPreTrain SparK ConvNeXtV2-Tiny

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

EXP_ID="${EXP_ID:-j11}"
TASK_CONFIG="${TASK_CONFIG:-}"
REALUW_SSL_ROOT="${REALUW_SSL_ROOT:-/media/HDD1/XCX/exp_2/REALUW_SSL}"
BUILD_REALUW_SSL="${BUILD_REALUW_SSL:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPU_IDS")
PORT="${PORT:-29721}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
S1_MAX_KEEP_CKPTS="${S1_MAX_KEEP_CKPTS:-${MAX_KEEP_CKPTS:-3}}"
MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

if [ -z "${MMPRETRAIN_DIR:-}" ]; then
    if [ -f "$REPO_ROOT/third_party/mmpretrain/tools/train.py" ]; then
        MMPRETRAIN_DIR="$REPO_ROOT/third_party/mmpretrain"
    else
        MMPRETRAIN_DIR="../mmpretrain"
    fi
fi
if [ -z "${DINO_DIR:-}" ]; then
    if [ -f "$REPO_ROOT/third_party/dino/main_dino.py" ]; then
        DINO_DIR="$REPO_ROOT/third_party/dino"
    else
        DINO_DIR="../dino"
    fi
fi
if [ -z "${SPARK_DIR:-}" ]; then
    if [ -f "$REPO_ROOT/third_party/SparK/pretrain/main.py" ]; then
        SPARK_DIR="$REPO_ROOT/third_party/SparK"
    else
        SPARK_DIR="../SparK"
    fi
fi

case "$EXP_ID" in
    j6) DEFAULT_TASK_CONFIG="configs/exp_2/tri_pretrain/s1_j6_spark_resnet50_realuw.sh" ;;
    j7) DEFAULT_TASK_CONFIG="configs/exp_2/tri_pretrain/s1_j7_dino_resnet50_realuw.sh" ;;
    j11) DEFAULT_TASK_CONFIG="configs/exp_2/tri_pretrain/s1_j11_mae_vit_base_realuw.sh" ;;
    j12) DEFAULT_TASK_CONFIG="configs/exp_2/tri_pretrain/s1_j12_simmim_swin_base_realuw.sh" ;;
    j13) DEFAULT_TASK_CONFIG="configs/exp_2/tri_pretrain/s1_j13_spark_convnextv2_tiny_realuw.sh" ;;
    *) DEFAULT_TASK_CONFIG="" ;;
esac

if [ -z "$TASK_CONFIG" ]; then
    TASK_CONFIG="$DEFAULT_TASK_CONFIG"
fi
if [ -n "$TASK_CONFIG" ] && [ -f "$TASK_CONFIG" ]; then
    # shellcheck disable=SC1090
    source "$TASK_CONFIG"
fi

mkdir -p "$LOG_DIR" "$WORK_ROOT"

if [ "$BUILD_REALUW_SSL" = "1" ]; then
    python tools/build_realuw_ssl_dataset.py \
        --preset exp2_bbox20pct \
        --out-root "$REALUW_SSL_ROOT" \
        --val-ratio 0 \
        --write-imagefolder
fi

REALUW_IMAGEFOLDER="$REALUW_SSL_ROOT/imagefolder"

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

choose_existing_config() {
    for candidate in "$@"; do
        if [ -f "$MMPRETRAIN_DIR/$candidate" ]; then
            echo "$MMPRETRAIN_DIR/$candidate"
            return 0
        fi
    done
    return 1
}

run_mmpretrain() {
    local name="$1"
    local config="$2"
    local batch_size="$3"
    local work_dir="$WORK_ROOT/$name"
    local log_file="$LOG_DIR/${name}_s1.log"

    if [ ! -f "$MMPRETRAIN_DIR/tools/train.py" ]; then
        echo "Error: MMPreTrain tools/train.py not found: $MMPRETRAIN_DIR/tools/train.py"
        echo "Set MMPRETRAIN_DIR=/path/to/mmpretrain"
        exit 1
    fi

    mkdir -p "$work_dir"
    export PORT MKL_THREADING_LAYER

    echo "========================================="
    echo "Tri-pretrain S1: $name"
    echo "runner: MMPreTrain"
    echo "config: $config"
    echo "work_dir: $work_dir"
    echo "gpu_ids: $GPU_IDS"
    echo "num_gpus: $NUM_GPUS"
    echo "batch_size_per_gpu: $batch_size"
    echo "realuw_imagefolder: $REALUW_IMAGEFOLDER"
    echo "========================================="

    wait_for_gpu_group "$GPU_IDS" "$name S1"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$MMPRETRAIN_DIR:${PYTHONPATH:-}" \
    python -m torch.distributed.launch \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$PORT" \
        "$MMPRETRAIN_DIR/tools/train.py" \
        "$config" \
        --work-dir "$work_dir" \
        --launcher pytorch \
        --cfg-options \
            train_dataloader.batch_size="$batch_size" \
            train_dataloader.dataset.type=ImageNet \
            train_dataloader.dataset.data_root="$REALUW_IMAGEFOLDER" \
            train_dataloader.dataset.split=train \
            default_hooks.checkpoint.max_keep_ckpts="$S1_MAX_KEEP_CKPTS" \
            resume=False \
            load_from=None \
        2>&1 | tee "$log_file"
}

run_dino_resnet50() {
    local name="j7_realuw_dino_resnet50"
    local work_dir="$WORK_ROOT/$name"
    local log_file="$LOG_DIR/${name}_s1.log"
    local data_path="$REALUW_IMAGEFOLDER/train"

    if [ ! -f "$DINO_DIR/main_dino.py" ]; then
        echo "Error: DINO main_dino.py not found: $DINO_DIR/main_dino.py"
        echo "Set DINO_DIR=/path/to/facebookresearch/dino"
        exit 1
    fi

    mkdir -p "$work_dir"
    export MKL_THREADING_LAYER

    echo "========================================="
    echo "Tri-pretrain S1: $name"
    echo "runner: facebookresearch/dino"
    echo "work_dir: $work_dir"
    echo "gpu_ids: $GPU_IDS"
    echo "num_gpus: $NUM_GPUS"
    echo "data_path: $data_path"
    echo "========================================="

    wait_for_gpu_group "$GPU_IDS" "$name S1"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$DINO_DIR:${PYTHONPATH:-}" \
    python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$PORT" \
        "$DINO_DIR/main_dino.py" \
        --arch "${DINO_ARCH:-resnet50}" \
        --optimizer "${DINO_OPTIMIZER:-sgd}" \
        --lr "${DINO_LR:-0.03}" \
        --weight_decay "${DINO_WEIGHT_DECAY:-1e-4}" \
        --weight_decay_end "${DINO_WEIGHT_DECAY_END:-1e-4}" \
        --global_crops_scale ${DINO_GLOBAL_CROPS_SCALE:-0.14 1} \
        --local_crops_scale ${DINO_LOCAL_CROPS_SCALE:-0.05 0.14} \
        --epochs "${DINO_EPOCHS:-100}" \
        --batch_size_per_gpu "${DINO_BATCH_SIZE_PER_GPU:-64}" \
        --num_workers "${DINO_NUM_WORKERS:-10}" \
        --saveckp_freq "${DINO_SAVECKP_FREQ:-50}" \
        --data_path "$data_path" \
        --output_dir "$work_dir" \
        2>&1 | tee "$log_file"
}

run_spark_resnet50() {
    local name="j6_realuw_spark_resnet50"
    local work_dir="$WORK_ROOT/$name"
    local log_file="$LOG_DIR/${name}_s1.log"
    local spark_log_file
    local data_path="$REALUW_IMAGEFOLDER/train"
    local spark_entry="$SPARK_DIR/pretrain/main.py"

    if [ ! -f "$spark_entry" ]; then
        echo "Error: SparK pretrain/main.py not found: $spark_entry"
        echo "Set SPARK_DIR=/path/to/keyu-tian/SparK"
        exit 1
    fi

    mkdir -p "$work_dir"
    work_dir="$(cd "$work_dir" && pwd)"
    case "$log_file" in
        /*) spark_log_file="$log_file" ;;
        *) spark_log_file="$REPO_ROOT/$log_file" ;;
    esac
    export MKL_THREADING_LAYER

    echo "========================================="
    echo "Tri-pretrain S1: $name"
    echo "runner: keyu-tian/SparK"
    echo "work_dir: $work_dir"
    echo "gpu_ids: $GPU_IDS"
    echo "num_gpus: $NUM_GPUS"
    echo "data_path: $data_path"
    echo "model: resnet50"
    echo "========================================="

    wait_for_gpu_group "$GPU_IDS" "$name S1"

    (
        cd "$SPARK_DIR/pretrain"
        CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        python -m torch.distributed.launch \
            --use_env \
            --nproc_per_node="$NUM_GPUS" \
            --master_port="$PORT" \
            main.py \
            --data_path "$data_path" \
            --exp_name "$name" \
            --exp_dir "$work_dir" \
            --model "${SPARK_MODEL:-resnet50}" \
            --bs "${SPARK_BS:-4096}" \
            --ep "${SPARK_EPOCHS:-1600}" \
            --wp_ep "${SPARK_WARMUP_EPOCHS:-40}" \
            --base_lr "${SPARK_BASE_LR:-2e-4}" \
            --wd "${SPARK_WEIGHT_DECAY:-0.04}" \
            --wde "${SPARK_WEIGHT_DECAY_END:-0.2}" \
            --mask "${SPARK_MASK_RATIO:-0.6}" \
            --input_size "${SPARK_INPUT_SIZE:-224}" \
            --opt "${SPARK_OPTIMIZER:-lamb}" \
            --dataloader_workers "${SPARK_WORKERS:-8}" \
            2>&1 | tee "$spark_log_file"
    )
}

case "$EXP_ID" in
    j6)
        run_spark_resnet50
        ;;
    j7)
        run_dino_resnet50
        ;;
    j11)
        run_mmpretrain \
            "j11_realuw_mae_vit_base" \
            "${CONFIG:-configs/exp_2/mmpretrain/realuw_ssl_mae_vit-base-p16_8xb512-amp-coslr-300e.py}" \
            "${BATCH_SIZE:-512}"
        ;;
    j12)
        if [ -n "${CONFIG:-}" ]; then
            selected_config="$CONFIG"
        else
            echo "Error: J12 requires a verified SwinV2-Base masked-modeling config."
            echo "The cloned MMPreTrain repo has Swin-Base SimMIM, but that does not strictly match SwinV2-Base downstream."
            echo "Set CONFIG=/path/to/your/swinv2_base_masked_modeling_config.py and rerun EXP_ID=j12."
            exit 1
        fi
        run_mmpretrain \
            "j12_realuw_simmim_swinv2_base" \
            "$selected_config" \
            "${BATCH_SIZE:-128}"
        ;;
    j13)
        run_mmpretrain \
            "j13_realuw_spark_convnextv2_tiny" \
            "${CONFIG:-configs/exp_2/mmpretrain/realuw_ssl_spark-convnextv2-tiny_16xb256-amp-coslr-800e.py}" \
            "${BATCH_SIZE:-256}"
        ;;
    *)
        echo "Error: unsupported EXP_ID=$EXP_ID"
        echo "Supported: j6, j7, j11, j12, j13"
        exit 1
        ;;
esac
