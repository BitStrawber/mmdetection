#!/usr/bin/env bash
# Train the four ImageNet-100K -> DFUI detector-backbone variants, then
# initialize standard 24-epoch RUOD Cascade R-CNN runs from their best backbones.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# The two architecture pipelines run concurrently.  Each pipeline processes
# the two DFUI data branches serially, so one two-GPU group owns one model.
R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-29680}"
MODEL_PREFIX="${MODEL_PREFIX:-imagenet100k}"
VARIANTS="${VARIANTS:-dfui_ruod,dfui_ruod_uiis}"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"

HF_ROOT="${HF_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output}"
DFUI_RUOD_ROOT="${DFUI_RUOD_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_EASY}"
DFUI_RUOD_UIIS_ROOT="${DFUI_RUOD_UIIS_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"

WORK_ROOT="${WORK_ROOT:-work_dirs/dfui_imagenet100k_100e}"
BACKBONE_ROOT="${BACKBONE_ROOT:-work_dirs/dfui_imagenet100k_backbones}"
LOG_ROOT="${LOG_ROOT:-logs/tri_pretrain/dfui_imagenet100k_100e}"
PRETRAIN_DIR="${PRETRAIN_DIR:-work_dirs/pretrained_weights/imagenet100k_100e_converted}"

# R50 uses the two exact, previously validated J10 stage-one experiments.
# The source configs are located dynamically because their final filename is
# experiment-specific, while the run-directory contract is stable.
J10_ROOT="${J10_ROOT:-work_dirs/j10_dino_official_source_compare}"
R50_DFUI_RUOD_J10_DIR="${R50_DFUI_RUOD_J10_DIR:-$J10_ROOT/j10_dino_official_dfui_ruod_easy_f1_lr000375_e48_s1}"
R50_DFUI_RUOD_UIIS_J10_DIR="${R50_DFUI_RUOD_UIIS_J10_DIR:-$J10_ROOT/j10_dino_official_dfui_ruod_uiis_easy_f1_lr000375_e48_s1}"
# ViT-S starts from the standard controlled-100K Cascade configuration; its
# DFUI stage is extended to 48 epochs below, with proportional LR milestones.
VITS_DFUI_CONFIG="${VITS_DFUI_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
R50_RUOD_CONFIG="${R50_RUOD_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py}"
VITS_RUOD_CONFIG="${VITS_RUOD_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"

DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
RUOD_EPOCHS="${RUOD_EPOCHS:-24}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME="${RESUME:-1}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"
CHECK_ONLY="${CHECK_ONLY:-0}"

DFUI_CLASSES="('holothurian','echinus','scallop','starfish','fish','corals','diver','cuttlefish','turtle','jellyfish','waterweeds')"

IMAGENET100K_R50_RAW="${IMAGENET100K_R50_RAW:-$HF_ROOT/PRETRAIN/Controlled100K/ImageNet/DINO_ResNet50_100e/checkpoint.pth}"
IMAGENET100K_VITS_RAW="${IMAGENET100K_VITS_RAW:-$HF_ROOT/PRETRAIN/Controlled100K/ImageNet/DINO_ViTS_100e/checkpoint.pth}"
IMAGENET100K_R50_INIT="$PRETRAIN_DIR/imagenet100k_dino_resnet50_100e_teacher_backbone.pth"
IMAGENET100K_VITS_INIT="$PRETRAIN_DIR/imagenet100k_dino_vits_100e_teacher_backbone.pth"

mkdir -p "$WORK_ROOT" "$BACKBONE_ROOT" "$LOG_ROOT" "$PRETRAIN_DIR"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
die() { echo "[$(timestamp)] ERROR: $*" >&2; exit 1; }

gpu_count() { awk -F, '{print NF}' <<< "$1"; }

wait_for_group() {
    local group="$1"
    local label="$2"
    [ "$WAIT_FOR_GPUS" = "1" ] || return 0
    while true; do
        local ready=1
        local gpu mem util
        IFS=',' read -r -a ids <<< "$group"
        for gpu in "${ids[@]}"; do
            read -r mem util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
            echo "[$(timestamp)] $label gpu${gpu}: memory=${mem}MiB util=${util}%"
            if [ "$mem" -gt "$GPU_MAX_MEM_MB" ] || [ "$util" -gt "$GPU_MAX_UTIL" ]; then
                ready=0
            fi
        done
        [ "$ready" -eq 1 ] && return 0
        sleep "$GPU_WAIT_INTERVAL"
    done
}

validate_dino_checkpoint() {
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

args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
arch = getattr(args, 'arch', None)
epoch = checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
teacher = checkpoint.get('teacher') if isinstance(checkpoint, dict) else None
print(f'raw={path} epoch={epoch} arch={arch} teacher_tensors={len(teacher) if isinstance(teacher, dict) else 0}')
if epoch != 100 or arch != expected_arch or not isinstance(teacher, dict) or not teacher:
    raise SystemExit('Source checkpoint is not the expected completed DINO-100e teacher checkpoint.')
PY
}

convert_teacher() {
    local raw="$1"
    local output="$2"
    local arch="$3"
    local prepend="$4"
    local temporary="${output}.partial.$$"
    validate_dino_checkpoint "$raw" "$arch"
    if [ -s "$output" ]; then
        if python - "$output" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else None
if not isinstance(state, dict) or not state:
    raise SystemExit('Converted checkpoint has no state_dict tensors.')
print(f'validated converted tensors={len(state)}')
PY
        then
            echo "[$(timestamp)] Reuse validated converted backbone: $output"
            return 0
        fi
        echo "[$(timestamp)] Remove invalid converted backbone: $output"
        rm -f -- "$output"
    fi
    rm -f -- "$temporary"
    python tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$raw" \
        --source teacher \
        --prepend "$prepend" \
        --out "$temporary"
    [ -s "$temporary" ] || die "DINO conversion did not produce $temporary"
    python - "$temporary" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else None
if not isinstance(state, dict) or not state:
    raise SystemExit('Converted checkpoint has no state_dict tensors.')
print(f'validated converted tensors={len(state)}')
PY
    mv -f -- "$temporary" "$output"
}

latest_best() {
    local work_dir="$1"
    find "$work_dir" -maxdepth 1 -type f -name 'best_coco_bbox_mAP_epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n 1 | cut -d' ' -f2-
}

export_detector_backbone() {
    local detector_checkpoint="$1"
    local output="$2"
    if [ -s "$output" ]; then
        echo "[$(timestamp)] Reuse exported detector backbone: $output"
        return 0
    fi
    python - "$detector_checkpoint" "$output" <<'PY'
import sys
from pathlib import Path
import torch

source, target = map(Path, sys.argv[1:])
try:
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(source, map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
if not isinstance(state, dict):
    raise SystemExit(f'No state dict found: {source}')

backbone = {
    key[len('backbone.'):]: value
    for key, value in state.items()
    if key.startswith('backbone.')
}
if not backbone:
    raise SystemExit(f'No backbone.* tensors found: {source}')

target.parent.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': backbone, 'meta': {'source': str(source)}}, target)
print(f'exported={target} tensors={len(backbone)}')
PY
}

common_options() {
    local init_checkpoint="$1"
    local classes="$2"
    local max_epochs="$3"
    cat <<EOF
load_from=None
model.backbone.init_cfg.type=Pretrained
model.backbone.init_cfg.checkpoint=$init_checkpoint
train_cfg.max_epochs=$max_epochs
default_hooks.checkpoint.save_best=coco/bbox_mAP
default_hooks.checkpoint.max_keep_ckpts=$MAX_KEEP_CKPTS
EOF
}

run_train() {
    local label="$1"
    local config="$2"
    local work_dir="$3"
    local group="$4"
    local port="$5"
    shift 5
    local -a options=("$@")
    local best
    local complete_marker="$work_dir/.pipeline_train_complete"
    best=$(latest_best "$work_dir" || true)
    if [ "$SKIP_COMPLETED" = "1" ] && [ -n "$best" ] && [ -f "$complete_marker" ]; then
        echo "[$(timestamp)] Reuse completed stage: $label best=$best"
        return 0
    fi

    local -a resume_args=()
    if [ "$RESUME" = "1" ] && [ -s "$work_dir/last_checkpoint" ]; then
        resume_args=(--resume)
    fi
    mkdir -p "$work_dir"
    echo "[$(timestamp)] START $label"
    echo "  config=$config"
    echo "  work_dir=$work_dir"
    echo "  gpu_group=$group"
    CUDA_VISIBLE_DEVICES="$group" PORT="$port" \
      bash tools/dist_train.sh "$config" "$(gpu_count "$group")" \
        --work-dir "$work_dir" \
        "${resume_args[@]}" \
        --cfg-options "${options[@]}"
    touch "$complete_marker"
    echo "[$(timestamp)] Training completed: $label marker=$complete_marker"
}

run_test() {
    local label="$1"
    local config="$2"
    local checkpoint="$3"
    local group="$4"
    local port="$5"
    shift 5
    local -a options=("$@")
    [ "$RUN_TEST" = "1" ] || return 0
    CUDA_VISIBLE_DEVICES="$group" PORT="$port" \
      bash tools/dist_test.sh "$config" "$checkpoint" "$(gpu_count "$group")" \
        --cfg-options "${options[@]}" \
      | tee "$LOG_ROOT/${label}_test.log"
}

set_bbox_classes() {
    local count="$1"
    printf '%s\n' \
      "model.roi_head.bbox_head.0.num_classes=$count" \
      "model.roi_head.bbox_head.1.num_classes=$count" \
      "model.roi_head.bbox_head.2.num_classes=$count"
}

set_dataset_classes() {
    local classes="$1"
    printf '%s\n' \
      "train_dataloader.dataset.metainfo.classes=$classes" \
      "val_dataloader.dataset.metainfo.classes=$classes" \
      "test_dataloader.dataset.metainfo.classes=$classes"
}

resolve_unique_config() {
    local directory="$1"
    local description="$2"
    local -a configs=()
    mapfile -t configs < <(find "$directory" -maxdepth 1 -type f -name '*.py' -print | sort)
    [ "${#configs[@]}" -eq 1 ] || die "$description must contain exactly one config (*.py), found ${#configs[@]} in $directory"
    printf '%s\n' "${configs[0]}"
}

run_architecture_pipeline() {
    local architecture="$1"
    local group="$2"
    local port="$3"
    local raw init ruod_config expected_arch prepend
    case "$architecture" in
      resnet50)
        raw="$IMAGENET100K_R50_RAW"
        init="$IMAGENET100K_R50_INIT"
        ruod_config="$R50_RUOD_CONFIG"
        expected_arch=resnet50
        prepend=""
        ;;
      vits)
        raw="$IMAGENET100K_VITS_RAW"
        init="$IMAGENET100K_VITS_INIT"
        ruod_config="$VITS_RUOD_CONFIG"
        expected_arch=vit_small
        prepend="backbone."
        ;;
      *) die "Unsupported architecture: $architecture" ;;
    esac

    [ -s "$ruod_config" ] || die "RUOD config missing: $ruod_config"
    [ -s "$raw" ] || die "Source DINO checkpoint missing: $raw"

    wait_for_group "$group" "$architecture pipeline"
    convert_teacher "$raw" "$init" "$expected_arch" "$prepend"

    local variant root
    for variant in "${VARIANT_LIST[@]}"; do
      local dfui_config
      if [ "$variant" = "dfui_ruod" ]; then
        root="$DFUI_RUOD_ROOT"
        if [ "$architecture" = "resnet50" ]; then
          dfui_config=$(resolve_unique_config "$R50_DFUI_RUOD_J10_DIR" "R50 DFUI+RUODEasy J10 configuration")
        else
          dfui_config="$VITS_DFUI_CONFIG"
        fi
      elif [ "$variant" = "dfui_ruod_uiis" ]; then
        root="$DFUI_RUOD_UIIS_ROOT"
        if [ "$architecture" = "resnet50" ]; then
          dfui_config=$(resolve_unique_config "$R50_DFUI_RUOD_UIIS_J10_DIR" "R50 DFUI+RUODEasy+UIISEasy J10 configuration")
        else
          dfui_config="$VITS_DFUI_CONFIG"
        fi
      else
        die "Unsupported VARIANTS entry: $variant (expected dfui_ruod or dfui_ruod_uiis)"
      fi
      [ -s "$dfui_config" ] || die "DFUI config missing: $dfui_config"
      for required in \
        "$root/annotations/instances_train.json" \
        "$root/annotations/instances_val.json" \
        "$root/images"; do
        [ -e "$required" ] || die "DFUI dataset requirement missing: $required"
      done

      local dfui_name="${MODEL_PREFIX}_${architecture}_${variant}_det48e"
      local dfui_work="$WORK_ROOT/$dfui_name"
      local -a dfui_opts=()
      while IFS= read -r item; do dfui_opts+=("$item"); done < <(common_options "$init" "$DFUI_CLASSES" "$DFUI_EPOCHS")
      while IFS= read -r item; do dfui_opts+=("$item"); done < <(set_bbox_classes 11)
      while IFS= read -r item; do dfui_opts+=("$item"); done < <(set_dataset_classes "$DFUI_CLASSES")
      dfui_opts+=(
        "train_dataloader.dataset.data_root=$root/"
        "train_dataloader.dataset.ann_file=$root/annotations/instances_train.json"
        "train_dataloader.dataset.data_prefix.img=$root/images/"
        "val_dataloader.dataset.data_root=$root/"
        "val_dataloader.dataset.ann_file=$root/annotations/instances_val.json"
        "val_dataloader.dataset.data_prefix.img=$root/images/"
        "test_dataloader.dataset.data_root=$root/"
        "test_dataloader.dataset.ann_file=$root/annotations/instances_val.json"
        "test_dataloader.dataset.data_prefix.img=$root/images/"
        "val_evaluator.ann_file=$root/annotations/instances_val.json"
        "test_evaluator.ann_file=$root/annotations/instances_val.json"
      )
      if [ "$architecture" = "vits" ]; then
        dfui_opts+=("param_scheduler.1.milestones=[32,44]")
      fi

      run_train "$dfui_name" "$dfui_config" "$dfui_work" "$group" "$port" "${dfui_opts[@]}"
      local dfui_best
      dfui_best=$(latest_best "$dfui_work" || true)
      [ -n "$dfui_best" ] || die "No best DFUI checkpoint after $dfui_name"
      run_test "$dfui_name" "$dfui_config" "$dfui_best" "$group" "$((port + 100))" "${dfui_opts[@]}"

      local exported="$BACKBONE_ROOT/${dfui_name}_best_backbone.pth"
      export_detector_backbone "$dfui_best" "$exported"

      for required in \
        "$RUOD_ROOT/annotations/instances_train.json" \
        "$RUOD_ROOT/annotations/instances_val.json" \
        "$RUOD_ROOT/train" \
        "$RUOD_ROOT/val"; do
        [ -e "$required" ] || die "RUOD dataset requirement missing: $required"
      done

      local ruod_name="${MODEL_PREFIX}_${architecture}_${variant}_backbone_ruod24e_det"
      local ruod_work="$WORK_ROOT/$ruod_name"
      local -a ruod_opts=()
      while IFS= read -r item; do ruod_opts+=("$item"); done < <(common_options "$exported" "" "$RUOD_EPOCHS")
      while IFS= read -r item; do ruod_opts+=("$item"); done < <(set_bbox_classes 10)
      ruod_opts+=(
        "train_dataloader.dataset.data_root=$RUOD_ROOT/"
        "train_dataloader.dataset.ann_file=$RUOD_ROOT/annotations/instances_train.json"
        "train_dataloader.dataset.data_prefix.img=$RUOD_ROOT/train/"
        "val_dataloader.dataset.data_root=$RUOD_ROOT/"
        "val_dataloader.dataset.ann_file=$RUOD_ROOT/annotations/instances_val.json"
        "val_dataloader.dataset.data_prefix.img=$RUOD_ROOT/val/"
        "test_dataloader.dataset.data_root=$RUOD_ROOT/"
        "test_dataloader.dataset.ann_file=$RUOD_ROOT/annotations/instances_val.json"
        "test_dataloader.dataset.data_prefix.img=$RUOD_ROOT/val/"
        "val_evaluator.ann_file=$RUOD_ROOT/annotations/instances_val.json"
        "test_evaluator.ann_file=$RUOD_ROOT/annotations/instances_val.json"
      )

      run_train "$ruod_name" "$ruod_config" "$ruod_work" "$group" "$((port + 10))" "${ruod_opts[@]}"
      local ruod_best
      ruod_best=$(latest_best "$ruod_work" || true)
      [ -n "$ruod_best" ] || die "No best RUOD checkpoint after $ruod_name"
      run_test "$ruod_name" "$ruod_config" "$ruod_best" "$group" "$((port + 110))" "${ruod_opts[@]}"
      port=$((port + 20))
    done
}

echo "============================================================"
echo "ImageNet-100K -> DFUI detection -> RUOD detection pipeline"
echo "R50 GPUs:  $R50_GPUS"
echo "ViT-S GPUs:$VITS_GPUS"
echo "Model prefix: $MODEL_PREFIX"
echo "Variants: $VARIANTS"
echo "DFUI data: $DFUI_RUOD_ROOT ; $DFUI_RUOD_UIIS_ROOT"
echo "R50 J10 configs: $R50_DFUI_RUOD_J10_DIR ; $R50_DFUI_RUOD_UIIS_J10_DIR"
echo "DFUI epochs: $DFUI_EPOCHS ; RUOD epochs: $RUOD_EPOCHS"
echo "Raw R50 checkpoint:  $IMAGENET100K_R50_RAW"
echo "Raw ViT-S checkpoint:$IMAGENET100K_VITS_RAW"
sha256sum "$IMAGENET100K_R50_RAW" "$IMAGENET100K_VITS_RAW"
echo "Pipeline log: $PIPELINE_LOG"
echo "============================================================"

for required in \
  tools/dist_train.sh \
  tools/dist_test.sh \
  tools/convert_ssl_backbone_to_mmdet.py \
  "$VITS_DFUI_CONFIG" "$R50_RUOD_CONFIG" "$VITS_RUOD_CONFIG"; do
  [ -s "$required" ] || die "Required file missing: $required"
done

if [ "$CHECK_ONLY" = "1" ]; then
  echo "R50 DFUI+RUODEasy config: $(resolve_unique_config "$R50_DFUI_RUOD_J10_DIR" "R50 DFUI+RUODEasy J10 configuration")"
  echo "R50 DFUI+RUODEasy+UIISEasy config: $(resolve_unique_config "$R50_DFUI_RUOD_UIIS_J10_DIR" "R50 DFUI+RUODEasy+UIISEasy J10 configuration")"
  for required in \
    "$DFUI_RUOD_ROOT/annotations/instances_train.json" \
    "$DFUI_RUOD_ROOT/annotations/instances_val.json" \
    "$DFUI_RUOD_ROOT/images" \
    "$DFUI_RUOD_UIIS_ROOT/annotations/instances_train.json" \
    "$DFUI_RUOD_UIIS_ROOT/annotations/instances_val.json" \
    "$DFUI_RUOD_UIIS_ROOT/images" \
    "$RUOD_ROOT/annotations/instances_train.json" \
    "$RUOD_ROOT/annotations/instances_val.json" \
    "$RUOD_ROOT/train" \
    "$RUOD_ROOT/val"; do
    [ -e "$required" ] || die "Dataset requirement missing: $required"
  done
  convert_teacher "$IMAGENET100K_R50_RAW" "$IMAGENET100K_R50_INIT" resnet50 ""
  convert_teacher "$IMAGENET100K_VITS_RAW" "$IMAGENET100K_VITS_INIT" vit_small "backbone."
  echo "CHECK_ONLY=1: source checkpoints, configs, and conversion passed."
  exit 0
fi

run_architecture_pipeline resnet50 "$R50_GPUS" "$BASE_PORT" &
pid_r50=$!
run_architecture_pipeline vits "$VITS_GPUS" "$((BASE_PORT + 200))" &
pid_vits=$!

status=0
wait "$pid_r50" || status=1
wait "$pid_vits" || status=1
[ "$status" -eq 0 ] || die "At least one architecture pipeline failed."

echo "[$(timestamp)] COMPLETE"
echo "backbones: $BACKBONE_ROOT"
echo "work dirs: $WORK_ROOT"
