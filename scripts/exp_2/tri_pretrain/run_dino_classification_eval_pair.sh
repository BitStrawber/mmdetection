#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PRESET="${PRESET:-imagenet}"
EXP_PREFIX="${EXP_PREFIX:-$PRESET}"
DATA_ROOT="${DATA_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K/imagefolder}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/third_party/dino}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/work_dirs/tri_pretrain/classification}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/tri_pretrain/classification}"

R50_GPU_IDS="${R50_GPU_IDS:-${R50_GPUS:-4,5}}"
VITS_GPU_IDS="${VITS_GPU_IDS:-${VITS_GPUS:-6,7}}"
R50_PORT_BASE="${R50_PORT_BASE:-29951}"
VITS_PORT_BASE="${VITS_PORT_BASE:-29952}"

RUN_KNN="${RUN_KNN:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
REUSE_KNN_FEATURES="${REUSE_KNN_FEATURES:-1}"
DOWNLOAD_OFFICIAL_EVAL="${DOWNLOAD_OFFICIAL_EVAL:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

KNN_BATCH_SIZE_PER_GPU="${KNN_BATCH_SIZE_PER_GPU:-128}"
KNN_NUM_WORKERS="${KNN_NUM_WORKERS:-10}"
KNN_VALUES="${KNN_VALUES:-10 20 100 200}"
KNN_TEMPERATURE="${KNN_TEMPERATURE:-0.07}"
KNN_USE_CUDA="${KNN_USE_CUDA:-true}"

LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LINEAR_BATCH_SIZE_PER_GPU="${LINEAR_BATCH_SIZE_PER_GPU:-128}"
LINEAR_NUM_WORKERS="${LINEAR_NUM_WORKERS:-10}"
LINEAR_VAL_FREQ="${LINEAR_VAL_FREQ:-1}"
R50_LINEAR_LR="${R50_LINEAR_LR:-0.075}"
VITS_LINEAR_LR="${VITS_LINEAR_LR:-0.001}"

EXPECTED_TRAIN="${EXPECTED_TRAIN:-1281167}"
EXPECTED_VAL="${EXPECTED_VAL:-50000}"
EXPECTED_CLASSES="${EXPECTED_CLASSES:-1000}"
STRICT_COUNTS="${STRICT_COUNTS:-1}"

resolve_preset_checkpoint() {
    local experiment_name="$1"
    local local_fallback="$2"
    local output_relative_path="${3:-}"
    local candidate
    local -a matches=()

    if [ -n "$output_relative_path" ]; then
        candidate="$OUTPUT_ROOT/$output_relative_path"
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    fi

    for candidate in \
        "$OUTPUT_ROOT/$experiment_name/checkpoint.pth" \
        "$OUTPUT_ROOT/work_dirs/tri_pretrain/$experiment_name/checkpoint.pth" \
        "$OUTPUT_ROOT/tri_pretrain/$experiment_name/checkpoint.pth"
    do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    if [ -d "$OUTPUT_ROOT" ]; then
        while IFS= read -r candidate; do
            matches+=("$candidate")
        done < <(
            find "$OUTPUT_ROOT" -type f \
                -path "*/$experiment_name/checkpoint.pth" \
                -print 2>/dev/null | sort
        )
    fi

    if [ "${#matches[@]}" -eq 1 ]; then
        printf '%s\n' "${matches[0]}"
        return
    fi
    if [ "${#matches[@]}" -gt 1 ]; then
        echo "Error: multiple checkpoints found for $experiment_name:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        echo "Set R50_CKPT or VITS_CKPT explicitly." >&2
        exit 2
    fi

    printf '%s\n' "$local_fallback"
}

case "$PRESET" in
    imagenet)
        R50_CKPT="${R50_CKPT:-$(resolve_preset_checkpoint imagenet_dino_resnet50_100e "$REPO_ROOT/work_dirs/tri_pretrain/imagenet_dino_resnet50_100e/checkpoint.pth" PRETRAIN/ImageNet/DINO_ResNet50_100e/checkpoint.pth)}"
        VITS_CKPT="${VITS_CKPT:-$(resolve_preset_checkpoint imagenet_dino_vits_100e "$REPO_ROOT/work_dirs/tri_pretrain/imagenet_dino_vits_100e/checkpoint.pth" PRETRAIN/ImageNet/DINO_ViTS_100e/checkpoint.pth)}"
        ;;
    realuw)
        R50_CKPT="${R50_CKPT:-$(resolve_preset_checkpoint j7_realuw_dino_resnet50_ssd100e "$REPO_ROOT/work_dirs/tri_pretrain/j7_realuw_dino_resnet50_ssd100e/checkpoint.pth" PRETRAIN/RealUW/DINO_ResNet50_100e/checkpoint.pth)}"
        VITS_CKPT="${VITS_CKPT:-$(resolve_preset_checkpoint j14_realuw_dino_vits_100e "$REPO_ROOT/work_dirs/tri_pretrain/j14_realuw_dino_vits_100e/checkpoint.pth" PRETRAIN/RealUW/DINO_ViTS_100e/checkpoint.pth)}"
        ;;
    synthetic5)
        R50_CKPT="${R50_CKPT:-$(resolve_preset_checkpoint synthetic5_merged_dino_resnet50_100e "$REPO_ROOT/work_dirs/tri_pretrain/synthetic5_merged_dino_resnet50_100e/checkpoint.pth")}"
        VITS_CKPT="${VITS_CKPT:-$(resolve_preset_checkpoint synthetic5_merged_dino_vits_100e "$REPO_ROOT/work_dirs/tri_pretrain/synthetic5_merged_dino_vits_100e/checkpoint.pth")}"
        ;;
    custom)
        R50_CKPT="${R50_CKPT:?R50_CKPT is required for PRESET=custom}"
        VITS_CKPT="${VITS_CKPT:?VITS_CKPT is required for PRESET=custom}"
        ;;
    *)
        echo "Error: unsupported PRESET=$PRESET (use imagenet, realuw, synthetic5, or custom)" >&2
        exit 2
        ;;
esac

count_gpus() {
    awk -F, '{print NF}' <<< "$1"
}

R50_NUM_GPUS="$(count_gpus "$R50_GPU_IDS")"
VITS_NUM_GPUS="$(count_gpus "$VITS_GPU_IDS")"

if [ "$R50_NUM_GPUS" -lt 1 ] || [ "$VITS_NUM_GPUS" -lt 1 ]; then
    echo "Error: both GPU groups must contain at least one GPU" >&2
    exit 2
fi

if [ "$RUN_KNN" != "1" ] && [ "$RUN_LINEAR" != "1" ]; then
    echo "Error: at least one of RUN_KNN or RUN_LINEAR must be 1" >&2
    exit 2
fi

mkdir -p "$WORK_ROOT" "$LOG_ROOT" "$DINO_DIR"

download_official_script() {
    local name="$1"
    local path="$DINO_DIR/$name"
    local url="https://raw.githubusercontent.com/facebookresearch/dino/main/$name"

    if [ -s "$path" ]; then
        return
    fi
    if [ "$DOWNLOAD_OFFICIAL_EVAL" != "1" ]; then
        echo "Error: missing $path and DOWNLOAD_OFFICIAL_EVAL=0" >&2
        exit 1
    fi
    command -v curl >/dev/null 2>&1 || {
        echo "Error: curl is required to download $url" >&2
        exit 1
    }
    echo "Download official DINO script: $url"
    curl -fL --retry 5 --retry-delay 5 "$url" -o "$path"
}

download_official_script eval_knn.py
download_official_script eval_linear.py

for path in \
    "$DINO_DIR/utils.py" \
    "$DINO_DIR/vision_transformer.py" \
    "$DINO_DIR/eval_knn.py" \
    "$DINO_DIR/eval_linear.py" \
    "$R50_CKPT" \
    "$VITS_CKPT" \
    "$DATA_ROOT/train" \
    "$DATA_ROOT/val"; do
    if [ ! -e "$path" ]; then
        echo "Error: required path not found: $path" >&2
        exit 1
    fi
done

count_images() {
    find "$1" -type f \
        \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) \
        2>/dev/null | wc -l
}

count_classes() {
    find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

TRAIN_IMAGES="$(count_images "$DATA_ROOT/train")"
VAL_IMAGES="$(count_images "$DATA_ROOT/val")"
TRAIN_CLASSES="$(count_classes "$DATA_ROOT/train")"
VAL_CLASSES="$(count_classes "$DATA_ROOT/val")"

echo "============================================================"
echo "Official DINO ImageNet classification evaluation"
echo "============================================================"
echo "PRESET:                    $PRESET"
echo "EXP_PREFIX:                $EXP_PREFIX"
echo "DATA_ROOT:                 $DATA_ROOT"
echo "OUTPUT_ROOT:               $OUTPUT_ROOT"
echo "R50_CKPT:                  $R50_CKPT"
echo "VITS_CKPT:                 $VITS_CKPT"
echo "R50_GPU_IDS:               $R50_GPU_IDS ($R50_NUM_GPUS GPUs)"
echo "VITS_GPU_IDS:              $VITS_GPU_IDS ($VITS_NUM_GPUS GPUs)"
echo "RUN_KNN:                   $RUN_KNN"
echo "RUN_LINEAR:                $RUN_LINEAR"
echo "REUSE_KNN_FEATURES:        $REUSE_KNN_FEATURES"
echo "LINEAR_EPOCHS:             $LINEAR_EPOCHS"
echo "LINEAR_BATCH_SIZE_PER_GPU: $LINEAR_BATCH_SIZE_PER_GPU"
echo "Dataset train:             $TRAIN_IMAGES images, $TRAIN_CLASSES classes"
echo "Dataset val:               $VAL_IMAGES images, $VAL_CLASSES classes"
echo "WORK_ROOT:                 $WORK_ROOT"
echo "LOG_ROOT:                  $LOG_ROOT"
echo "============================================================"

if [ "$STRICT_COUNTS" = "1" ]; then
    if [ "$TRAIN_IMAGES" -ne "$EXPECTED_TRAIN" ] || \
       [ "$VAL_IMAGES" -ne "$EXPECTED_VAL" ] || \
       [ "$TRAIN_CLASSES" -ne "$EXPECTED_CLASSES" ] || \
       [ "$VAL_CLASSES" -ne "$EXPECTED_CLASSES" ]; then
        echo "Error: dataset counts do not match expected values" >&2
        exit 1
    fi
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY=1: preflight passed; no evaluation was started."
    exit 0
fi

export PYTHONPATH="$DINO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

run_knn() {
    local model_name="$1"
    local gpu_ids="$2"
    local num_gpus="$3"
    local arch="$4"
    local checkpoint="$5"
    local port="$6"
    shift 6

    local task_name="${EXP_PREFIX}_${model_name}"
    local feature_dir="$WORK_ROOT/${task_name}_knn_features"
    local task_log="$LOG_ROOT/${task_name}_knn_top15.log"
    local feature_args=(--dump_features "$feature_dir")
    local knn_values=()
    # shellcheck disable=SC2206
    knn_values=($KNN_VALUES)

    mkdir -p "$feature_dir"
    if [ "$REUSE_KNN_FEATURES" = "1" ] && \
       [ -s "$feature_dir/trainfeat.pth" ] && \
       [ -s "$feature_dir/testfeat.pth" ] && \
       [ -s "$feature_dir/trainlabels.pth" ] && \
       [ -s "$feature_dir/testlabels.pth" ]; then
        feature_args=(--load_features "$feature_dir")
        echo "Reuse k-NN features: $feature_dir"
    fi

    echo "START $task_name k-NN GPUs=$gpu_ids at $(date)"
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    python -u -m torch.distributed.launch \
        --nproc_per_node="$num_gpus" \
        --use_env \
        --master_port="$port" \
        "$DINO_DIR/eval_knn.py" \
        --arch "$arch" \
        "$@" \
        --pretrained_weights "$checkpoint" \
        --checkpoint_key teacher \
        --data_path "$DATA_ROOT" \
        --batch_size_per_gpu "$KNN_BATCH_SIZE_PER_GPU" \
        --num_workers "$KNN_NUM_WORKERS" \
        --nb_knn "${knn_values[@]}" \
        --temperature "$KNN_TEMPERATURE" \
        --use_cuda "$KNN_USE_CUDA" \
        "${feature_args[@]}" \
        2>&1 | tee "$task_log"
    echo "DONE $task_name k-NN at $(date)"
}

run_linear() {
    local model_name="$1"
    local gpu_ids="$2"
    local num_gpus="$3"
    local arch="$4"
    local checkpoint="$5"
    local learning_rate="$6"
    local port="$7"
    shift 7

    local task_name="${EXP_PREFIX}_${model_name}"
    local output_dir="$WORK_ROOT/${task_name}_linear_${LINEAR_EPOCHS}e"
    local task_log="$LOG_ROOT/${task_name}_linear_${LINEAR_EPOCHS}e_top15.log"

    mkdir -p "$output_dir"
    echo "START $task_name Linear GPUs=$gpu_ids at $(date)"
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    python -u -m torch.distributed.launch \
        --nproc_per_node="$num_gpus" \
        --use_env \
        --master_port="$port" \
        "$DINO_DIR/eval_linear.py" \
        --arch "$arch" \
        "$@" \
        --pretrained_weights "$checkpoint" \
        --checkpoint_key teacher \
        --data_path "$DATA_ROOT" \
        --output_dir "$output_dir" \
        --epochs "$LINEAR_EPOCHS" \
        --lr "$learning_rate" \
        --batch_size_per_gpu "$LINEAR_BATCH_SIZE_PER_GPU" \
        --num_workers "$LINEAR_NUM_WORKERS" \
        --val_freq "$LINEAR_VAL_FREQ" \
        2>&1 | tee "$task_log"
    echo "DONE $task_name Linear at $(date)"
}

wait_pair() {
    local first_pid="$1"
    local second_pid="$2"
    local stage="$3"
    local status=0

    wait "$first_pid" || status=1
    wait "$second_pid" || status=1
    if [ "$status" -ne 0 ]; then
        echo "Error: one or more $stage tasks failed at $(date)" >&2
        return 1
    fi
    echo "BOTH $stage TASKS FINISHED at $(date)"
}

if [ "$RUN_KNN" = "1" ]; then
    echo "STAGE 1: PARALLEL k-NN"
    run_knn r50 "$R50_GPU_IDS" "$R50_NUM_GPUS" resnet50 \
        "$R50_CKPT" "$R50_PORT_BASE" &
    R50_PID=$!
    run_knn vits "$VITS_GPU_IDS" "$VITS_NUM_GPUS" vit_small \
        "$VITS_CKPT" "$VITS_PORT_BASE" --patch_size 16 &
    VITS_PID=$!
    wait_pair "$R50_PID" "$VITS_PID" k-NN
fi

if [ "$RUN_LINEAR" = "1" ]; then
    echo "STAGE 2: PARALLEL LINEAR"
    run_linear r50 "$R50_GPU_IDS" "$R50_NUM_GPUS" resnet50 \
        "$R50_CKPT" "$R50_LINEAR_LR" "$((R50_PORT_BASE + 10))" &
    R50_PID=$!
    run_linear vits "$VITS_GPU_IDS" "$VITS_NUM_GPUS" vit_small \
        "$VITS_CKPT" "$VITS_LINEAR_LR" "$((VITS_PORT_BASE + 10))" \
        --patch_size 16 --n_last_blocks 4 --avgpool_patchtokens false &
    VITS_PID=$!
    wait_pair "$R50_PID" "$VITS_PID" LINEAR
fi

SUMMARY_PATH="$LOG_ROOT/${EXP_PREFIX}_classification_summary.txt"
{
    echo "Experiment: $EXP_PREFIX"
    echo "Data: $DATA_ROOT"
    echo "Generated: $(date)"
    echo
    echo "k-NN Top-1 / Top-5:"
    grep -H -E '[0-9]+-NN classifier result' \
        "$LOG_ROOT/${EXP_PREFIX}_r50_knn_top15.log" \
        "$LOG_ROOT/${EXP_PREFIX}_vits_knn_top15.log" 2>/dev/null || true
    echo
    echo "Linear best Top-1 and corresponding Top-5:"
    python - \
        "$WORK_ROOT/${EXP_PREFIX}_r50_linear_${LINEAR_EPOCHS}e/log.txt" \
        "$WORK_ROOT/${EXP_PREFIX}_vits_linear_${LINEAR_EPOCHS}e/log.txt" <<'PY'
import json
import sys
from pathlib import Path

for value in sys.argv[1:]:
    path = Path(value)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "test_acc1" in row and "test_acc5" in row:
                rows.append(row)

    name = path.parent.name
    if not rows:
        print("{}: no validation records".format(name))
        continue
    best = max(rows, key=lambda row: row["test_acc1"])
    print(
        "{}: epoch={} Top-1={:.3f} Top-5={:.3f}".format(
            name,
            best["epoch"],
            best["test_acc1"],
            best["test_acc5"],
        )
    )
PY
} | tee "$SUMMARY_PATH"

echo "ALL CLASSIFICATION TASKS FINISHED at $(date)"
echo "Summary: $SUMMARY_PATH"
