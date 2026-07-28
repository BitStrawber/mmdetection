#!/bin/bash
set -Eeuo pipefail

# Three GPU-group pipelines, four serial tasks per controlled 100K dataset:
#   R50 RUOD det -> R50 UIIS10K mask -> ViT-S RUOD det -> ViT-S UIIS10K mask.
# The three dataset pipelines run concurrently on GPU groups 01, 34, and 56.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

WORK_ROOT="${WORK_ROOT:-work_dirs/tri_pretrain}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"
HF_ROOT="${HF_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_DIR/dino_100k_control_downstream_12tasks.log}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"

R50_DET_CONFIG="${R50_DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py}"
R50_MASK_CONFIG="${R50_MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py}"
VITS_DET_CONFIG="${VITS_DET_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
VITS_MASK_CONFIG="${VITS_MASK_CONFIG:-configs/exp_2/tri_pretrain/mask-rcnn_vit-small_dino_fpn_24e_uiis10k_control100k.py}"
CONVERTER="${CONVERTER:-tools/convert_ssl_backbone_to_mmdet.py}"

IMAGENET_GPUS="${IMAGENET_GPUS:-0,1}"
REALUW_GPUS="${REALUW_GPUS:-3,4}"
SYNTHETIC5_GPUS="${SYNTHETIC5_GPUS:-5,6}"
BASE_PORT="${BASE_PORT:-29960}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

RUN_CONVERT="${RUN_CONVERT:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

mkdir -p "$WORK_ROOT" "$LOG_DIR" "$PRETRAIN_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

die() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

dataset_info() {
    local dataset="$1"
    case "$dataset" in
        imagenet100k)
            HF_DATASET="ImageNet"
            GPU_GROUP="$IMAGENET_GPUS"
            DATASET_INDEX=0
            ;;
        realuw100k)
            HF_DATASET="RealUW"
            GPU_GROUP="$REALUW_GPUS"
            DATASET_INDEX=1
            ;;
        synthetic5_100k)
            HF_DATASET="Synthetic5"
            GPU_GROUP="$SYNTHETIC5_GPUS"
            DATASET_INDEX=2
            ;;
        *)
            die "Unsupported dataset: $dataset"
            ;;
    esac
}

raw_checkpoint() {
    local dataset="$1"
    local backbone="$2"
    local local_path="$WORK_ROOT/control100k_${dataset}_dino_${backbone}_100e/checkpoint.pth"
    local hf_model

    dataset_info "$dataset"
    case "$backbone" in
        resnet50) hf_model="DINO_ResNet50_100e" ;;
        vits) hf_model="DINO_ViTS_100e" ;;
        *) die "Unsupported backbone: $backbone" ;;
    esac

    local hf_path="$HF_ROOT/PRETRAIN/Controlled100K/$HF_DATASET/$hf_model/checkpoint.pth"
    if [ -s "$local_path" ]; then
        printf '%s\n' "$local_path"
    elif [ -s "$hf_path" ]; then
        printf '%s\n' "$hf_path"
    else
        die "Checkpoint not found for $dataset $backbone: $local_path or $hf_path"
    fi
}

converted_checkpoint() {
    local dataset="$1"
    local backbone="$2"
    printf '%s/control100k_%s_dino_%s_100e_teacher_backbone.pth\n' \
        "$PRETRAIN_DIR" "$dataset" "$backbone"
}

validate_raw_checkpoint() {
    local checkpoint="$1"
    local expected_arch="$2"
    python - "$checkpoint" "$expected_arch" <<'PY'
import sys
import torch

path, expected_arch = sys.argv[1:]
try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location='cpu')

args = checkpoint.get('args')
epoch = checkpoint.get('epoch')
arch = getattr(args, 'arch', None)
teacher = checkpoint.get('teacher')

print(
    f'raw_checkpoint={path} epoch={epoch} arch={arch} '
    f'teacher_tensors={len(teacher) if isinstance(teacher, dict) else 0}')
if epoch != 100:
    raise SystemExit(f'Incomplete checkpoint epoch: {epoch}')
if arch != expected_arch:
    raise SystemExit(f'Expected arch={expected_arch}, found {arch}')
if not isinstance(teacher, dict) or not teacher:
    raise SystemExit('Missing DINO teacher state dict')
PY
}

validate_converted_checkpoint() {
    local checkpoint="$1"
    local backbone="$2"
    python - "$checkpoint" "$backbone" <<'PY'
import sys
import torch

path, backbone = sys.argv[1:]
try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location='cpu')

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get(
        'state_dict', checkpoint.get('model', checkpoint))
else:
    state_dict = checkpoint

if not isinstance(state_dict, dict) or not state_dict:
    raise SystemExit(f'Converted checkpoint has no state dict: {path}')

keys = set(state_dict)
if any(key.startswith('module.') for key in keys):
    raise SystemExit(f'Unexpected module. prefix in {path}')

if backbone == 'resnet50':
    required = {'conv1.weight', 'bn1.weight'}
    if not required.issubset(keys):
        raise SystemExit(
            f'ResNet50 keys missing from {path}: {sorted(required - keys)}')
    if any(key.startswith('backbone.') for key in keys):
        raise SystemExit(f'ResNet50 checkpoint has stale backbone. prefix: {path}')
elif backbone == 'vits':
    required = {
        'backbone.patch_embed.proj.weight',
        'backbone.blocks.0.attn.qkv.weight',
        'backbone.blocks.11.attn.qkv.weight',
        'backbone.norm.weight',
    }
    if not required.issubset(keys):
        raise SystemExit(
            f'ViT-S keys missing from {path}: {sorted(required - keys)}')
else:
    raise SystemExit(f'Unsupported backbone: {backbone}')

print(
    f'converted_checkpoint={path} backbone={backbone} '
    f'tensors={len(state_dict)} prefix_check=PASS')
PY
}

convert_one() {
    local dataset="$1"
    local backbone="$2"
    local expected_arch="$3"
    local prepend="$4"
    local raw
    local output

    raw=$(raw_checkpoint "$dataset" "$backbone")
    output=$(converted_checkpoint "$dataset" "$backbone")
    validate_raw_checkpoint "$raw" "$expected_arch"

    if [ "$RUN_CONVERT" != "1" ]; then
        [ -s "$output" ] || die "Converted checkpoint missing: $output"
        validate_converted_checkpoint "$output" "$backbone"
        return
    fi
    if [ "$FORCE_CONVERT" != "1" ] && [ -s "$output" ]; then
        echo "Converted checkpoint exists: $output"
        validate_converted_checkpoint "$output" "$backbone"
        return
    fi

    echo "Convert teacher backbone: dataset=$dataset backbone=$backbone"
    python "$CONVERTER" \
        --checkpoint "$raw" \
        --source teacher \
        --prepend "$prepend" \
        --out "$output" \
        2>&1 | tee "$LOG_DIR/control100k_${dataset}_${backbone}_convert.log"

    [ -s "$output" ] || die "Conversion did not produce: $output"
    validate_converted_checkpoint "$output" "$backbone"
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

wait_for_gpu_group() {
    local gpu_ids="$1"
    local label="$2"
    if [ "$WAIT_FOR_GPUS" != "1" ]; then
        echo "WAIT_FOR_GPUS=$WAIT_FOR_GPUS, start $label without waiting."
        return
    fi

    local idle_rounds=0
    local gpu_array=()
    IFS=',' read -r -a gpu_array <<< "$gpu_ids"

    while true; do
        local all_idle=1
        local status=()
        for gpu in "${gpu_array[@]}"; do
            local state
            state=$(query_gpu_state "$gpu" || true)
            if [ -z "$state" ]; then
                status+=("gpu${gpu}=not_found")
                all_idle=0
                continue
            fi
            local mem util
            read -r mem util <<< "$state"
            status+=("gpu${gpu}=mem:${mem}MB,util:${util}%")
            if [ "$mem" -gt "$GPU_MAX_MEM_MB" ] || [ "$util" -gt "$GPU_MAX_UTIL" ]; then
                all_idle=0
            fi
        done
        echo "GPU status for $label: ${status[*]}"
        if [ "$all_idle" -eq 1 ]; then
            idle_rounds=$((idle_rounds + 1))
            if [ "$idle_rounds" -ge "$GPU_IDLE_CHECKS" ]; then
                echo "GPU group [$gpu_ids] is ready for $label."
                return
            fi
        else
            idle_rounds=0
        fi
        sleep "$GPU_WAIT_INTERVAL"
    done
}

validate_configs() {
    python - \
        "$R50_DET_CONFIG" "$R50_MASK_CONFIG" \
        "$VITS_DET_CONFIG" "$VITS_MASK_CONFIG" <<'PY'
import sys
from mmengine.config import Config

for path in sys.argv[1:]:
    cfg = Config.fromfile(path)
    train_cfg = cfg.train_cfg
    backbone = cfg.model.backbone
    schedulers = cfg.param_scheduler

    print('=' * 100)
    print('config:', path)
    print('max_epochs:', train_cfg.get('max_epochs'))
    print('val_interval:', train_cfg.get('val_interval'))
    print('backbone:', backbone.get('type'))
    print('optimizer:', cfg.optim_wrapper.get('optimizer'))
    print('scheduler:', schedulers)

    if train_cfg.get('max_epochs') != 24:
        raise SystemExit(f'{path}: expected max_epochs=24')
    if train_cfg.get('val_interval') != 1:
        raise SystemExit(f'{path}: expected val_interval=1')

    multistep = [
        item for item in schedulers
        if item.get('type') == 'MultiStepLR'
    ]
    if not multistep:
        raise SystemExit(f'{path}: MultiStepLR not found')
    if multistep[0].get('milestones') != [16, 22]:
        raise SystemExit(f'{path}: expected milestones=[16, 22]')
PY
}

validate_datasets() {
    for path in \
        "$RUOD_ROOT/annotations/instances_train.json" \
        "$RUOD_ROOT/annotations/instances_val.json" \
        "$RUOD_ROOT/train" \
        "$RUOD_ROOT/val" \
        "$UIIS_ROOT/annotations/instances_train.json" \
        "$UIIS_ROOT/annotations/instances_val.json" \
        "$UIIS_ROOT/train" \
        "$UIIS_ROOT/val"; do
        [ -e "$path" ] || die "Dataset path missing: $path"
    done
}

run_task() {
    local dataset="$1"
    local backbone="$2"
    local task="$3"
    local gpu_ids="$4"
    local port="$5"
    local config
    local data_root
    local save_best
    local suffix
    local checkpoint

    checkpoint=$(converted_checkpoint "$dataset" "$backbone")
    if [ "$task" = "det" ]; then
        data_root="$RUOD_ROOT"
        save_best="coco/bbox_mAP"
        suffix="ruod24e_det"
        if [ "$backbone" = "resnet50" ]; then
            config="$R50_DET_CONFIG"
        else
            config="$VITS_DET_CONFIG"
        fi
    else
        data_root="$UIIS_ROOT"
        save_best="coco/segm_mAP"
        suffix="uiis10k24e_mask"
        if [ "$backbone" = "resnet50" ]; then
            config="$R50_MASK_CONFIG"
        else
            config="$VITS_MASK_CONFIG"
        fi
    fi

    local name="control100k_${dataset}_dino_${backbone}_pre100e_${suffix}"
    local work_dir="$WORK_ROOT/$name"
    local log_file="$LOG_DIR/${name}.log"
    local test_log="$LOG_DIR/${name}_test.log"
    local num_gpus
    num_gpus=$(awk -F, '{print NF}' <<< "$gpu_ids")

    echo "================================================================================"
    echo "[$(timestamp)] Start $name"
    echo "config: $config"
    echo "checkpoint: $checkpoint"
    echo "gpu_ids: $gpu_ids"
    echo "work_dir: $work_dir"
    echo "================================================================================"

    mkdir -p "$work_dir"
    if [ "$RUN_TRAIN" = "1" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_ids" PORT="$port" \
        bash tools/dist_train.sh "$config" "$num_gpus" \
            --work-dir "$work_dir" \
            --cfg-options \
                load_from=None \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint="$checkpoint" \
                train_dataloader.dataset.data_root="$data_root/" \
                val_dataloader.dataset.data_root="$data_root/" \
                test_dataloader.dataset.data_root="$data_root/" \
                val_evaluator.ann_file="$data_root/annotations/instances_val.json" \
                test_evaluator.ann_file="$data_root/annotations/instances_val.json" \
                default_hooks.checkpoint.save_best="$save_best" \
                default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS" \
            2>&1 | tee "$log_file"
    fi

    if [ "$RUN_TEST" = "1" ]; then
        local best_checkpoint
        best_checkpoint=$(ls -t "$work_dir"/best_*.pth 2>/dev/null | head -1 || true)
        [ -n "$best_checkpoint" ] || die "Best checkpoint not found in $work_dir"

        CUDA_VISIBLE_DEVICES="$gpu_ids" PORT="$((port + 100))" \
        bash tools/dist_test.sh "$config" "$best_checkpoint" "$num_gpus" \
            --cfg-options \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint="$checkpoint" \
                test_dataloader.dataset.data_root="$data_root/" \
                test_evaluator.ann_file="$data_root/annotations/instances_val.json" \
            2>&1 | tee "$test_log"
    fi

    echo "[$(timestamp)] Complete $name"
}

run_dataset_pipeline() {
    local dataset="$1"
    dataset_info "$dataset"
    local gpu_ids="$GPU_GROUP"
    local base_port="$((BASE_PORT + DATASET_INDEX * 10))"

    wait_for_gpu_group "$gpu_ids" "$dataset downstream pipeline"
    run_task "$dataset" resnet50 det "$gpu_ids" "$base_port"
    run_task "$dataset" resnet50 mask "$gpu_ids" "$((base_port + 1))"
    run_task "$dataset" vits det "$gpu_ids" "$((base_port + 2))"
    run_task "$dataset" vits mask "$gpu_ids" "$((base_port + 3))"
}

for required in \
    "$CONVERTER" \
    "$R50_DET_CONFIG" "$R50_MASK_CONFIG" \
    "$VITS_DET_CONFIG" "$VITS_MASK_CONFIG"; do
    [ -s "$required" ] || die "Required file missing: $required"
done

validate_datasets
validate_configs

for dataset in imagenet100k realuw100k synthetic5_100k; do
    convert_one "$dataset" resnet50 resnet50 ""
    convert_one "$dataset" vits vit_small "backbone."
done

if [ "$VALIDATE_ONLY" = "1" ]; then
    echo "[$(timestamp)] VALIDATE_ONLY=1, stop before downstream training."
    exit 0
fi

declare -a pids=()
for dataset in imagenet100k realuw100k synthetic5_100k; do
    run_dataset_pipeline "$dataset" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    die "At least one controlled 100K downstream pipeline failed."
fi

echo "================================================================================"
echo "[$(timestamp)] All 12 controlled 100K downstream tasks completed."
echo "work_root: $WORK_ROOT"
echo "log_dir: $LOG_DIR"
echo "================================================================================"
