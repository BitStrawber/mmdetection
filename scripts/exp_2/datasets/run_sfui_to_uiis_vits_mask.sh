#!/usr/bin/env bash
# Configurable SFUI Mask R-CNN adaptation followed by UIIS10K Mask R-CNN.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new experiment directory}"
SFUI_DATA_ROOT="${SFUI_DATA_ROOT:-}"
SFUI_TRAIN_ANN="${SFUI_TRAIN_ANN:-}"
MODEL_TAG="${MODEL_TAG:-vits}"
DINO_RAW="${DINO_RAW:-${VITS_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/ImageNet/DINO_ViTS_100e/checkpoint.pth}}"
SFUI_CONFIG="${SFUI_CONFIG:-configs/exp_2/tri_pretrain/mask-rcnn_vit-small_dino_fpn_24e_uiis10k_control100k.py}"
UIIS_CONFIG="${UIIS_CONFIG:-$SFUI_CONFIG}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"
CONVERTER="${CONVERTER:-tools/convert_ssl_backbone_to_mmdet.py}"
HELPER="tools/exp_2/prepare_sfui_uiis_mask_configs.py"

RUN_SFUI="${RUN_SFUI:-1}"
RUN_UIIS="${RUN_UIIS:-1}"
SFUI_CHECKPOINT="${SFUI_CHECKPOINT:-}"
SFUI_VAL_ANN="${SFUI_VAL_ANN:-}"
SFUI_TEST_ANN="${SFUI_TEST_ANN:-}"
SFUI_TRAIN_PREFIX="${SFUI_TRAIN_PREFIX:-train/images/}"
SFUI_VAL_PREFIX="${SFUI_VAL_PREFIX:-train/images/}"
SFUI_TEST_PREFIX="${SFUI_TEST_PREFIX:-$SFUI_TRAIN_PREFIX}"
SFUI_EPOCHS="${SFUI_EPOCHS:-48}"
SFUI_LR="${SFUI_LR:-}"
SFUI_LR_MILESTONES="${SFUI_LR_MILESTONES:-}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
UIIS_LR="${UIIS_LR:-}"
UIIS_LR_MILESTONES="${UIIS_LR_MILESTONES:-}"
UIIS_TRAIN_PREFIX="${UIIS_TRAIN_PREFIX:-train/}"
UIIS_VAL_PREFIX="${UIIS_VAL_PREFIX:-val/}"
SFUI_GPUS="${SFUI_GPUS:-6,7}"
UIIS_GPUS="${UIIS_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-39200}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-60}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
SFUI_RUN_TEST="${SFUI_RUN_TEST:-0}"
UIIS_RUN_TEST="${UIIS_RUN_TEST:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_ROOT="$OUTPUT_ROOT/work_dirs"
CONVERTED_ROOT="$OUTPUT_ROOT/converted"
BACKBONE_ROOT="$OUTPUT_ROOT/backbones"
LOG_ROOT="$OUTPUT_ROOT/logs"

die() { echo "ERROR: $*" >&2; exit 1; }
gpu_count() { awk -F, '{print NF}' <<< "$1"; }
best_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 1 | cut -d' ' -f2-
}

[[ "$RUN_SFUI" =~ ^[01]$ && "$RUN_UIIS" =~ ^[01]$ ]] || die "RUN_SFUI/RUN_UIIS must be 0 or 1"
[[ "$RUN_SFUI$RUN_UIIS" != "00" ]] || die "at least one stage must be enabled"
if [[ "$RUN_SFUI" == 1 ]]; then
  [[ -d "$SFUI_DATA_ROOT" && -s "$SFUI_TRAIN_ANN" ]] || die "SFUI data root or annotation is missing"
  [[ -s "$DINO_RAW" ]] || die "missing DINO checkpoint: $DINO_RAW"
  [[ -s "$SFUI_CONFIG" && -s "$CONVERTER" ]] || die "missing SFUI base config or DINO converter"
fi
[[ -s "$HELPER" ]] || die "missing config helper: $HELPER"
if [[ "$RUN_UIIS" == 1 ]]; then
  [[ -s "$UIIS_CONFIG" ]] || die "missing UIIS base config: $UIIS_CONFIG"
  [[ -s "$UIIS_ROOT/annotations/instances_train.json" && -s "$UIIS_ROOT/annotations/instances_val.json" ]] || die "invalid UIIS dataset root"
fi
[[ "$RUN_UIIS" != 1 || "$RUN_SFUI" == 1 || -s "$SFUI_CHECKPOINT" ]] || die "UIIS-only mode needs SFUI_CHECKPOINT"

mkdir -p "$WORK_ROOT" "$CONVERTED_ROOT" "$BACKBONE_ROOT" "$LOG_ROOT"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/sfui_uiis_pipeline_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

wait_for_gpus() {
  local group="$1" stage="$2"
  [[ "$WAIT_FOR_GPU" == 1 ]] || return 0
  while true; do
    local ready=1 gpu free
    IFS=',' read -r -a ids <<< "$group"
    for gpu in "${ids[@]}"; do
      free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
      echo "GPU=$gpu free_MiB=$free required_MiB=$MIN_FREE_MIB stage=$stage"
      [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_MIB" ]] || ready=0
    done
    [[ "$ready" == 1 ]] && return 0
    echo "Waiting for sufficient free GPU memory before $stage; interval=${GPU_WAIT_INTERVAL}s"
    sleep "$GPU_WAIT_INTERVAL"
  done
}

prepare_raw_backbone() {
  local output="$CONVERTED_ROOT/imagenet_${MODEL_TAG}_dino_teacher_backbone.pth"
  if [[ ! -s "$output" ]]; then
    local partial="$output.partial.$$"
    "$PYTHON_BIN" "$CONVERTER" --checkpoint "$DINO_RAW" --source teacher --prepend backbone. --out "$partial" \
      2>&1 | tee "$LOG_ROOT/convert_imagenet_${MODEL_TAG}.log"
    [[ -s "$partial" ]] || die "DINO backbone conversion did not produce weights"
    mv "$partial" "$output"
  fi
}

write_sfui_config() {
  local init="$1" work="$2"
  local runtime="$work/runtime_config.py" candidate="$work/runtime_config.candidate.py"
  local args=(prepare-sfui --base-config "$SFUI_CONFIG" --output "$candidate"
    --init-checkpoint "$init" --data-root "$SFUI_DATA_ROOT" --train-ann "$SFUI_TRAIN_ANN"
    --train-prefix "$SFUI_TRAIN_PREFIX" --test-prefix "$SFUI_TEST_PREFIX"
    --epochs "$SFUI_EPOCHS" --max-keep-ckpts "$MAX_KEEP_CKPTS"
    --milestones "$SFUI_LR_MILESTONES")
  [[ -z "$SFUI_LR" ]] || args+=(--lr "$SFUI_LR")
  [[ -z "$SFUI_VAL_ANN" ]] || args+=(--val-ann "$SFUI_VAL_ANN" --val-prefix "$SFUI_VAL_PREFIX")
  [[ -z "$SFUI_TEST_ANN" ]] || args+=(--test-ann "$SFUI_TEST_ANN")
  "$PYTHON_BIN" "$HELPER" "${args[@]}"
  guard_or_install_config "$candidate" "$runtime" "$work"
}

write_uiis_config() {
  local init="$1" work="$2"
  local runtime="$work/runtime_config.py" candidate="$work/runtime_config.candidate.py"
  local args=(prepare-uiis --base-config "$UIIS_CONFIG" --output "$candidate"
    --init-checkpoint "$init" --data-root "$UIIS_ROOT"
    --train-prefix "$UIIS_TRAIN_PREFIX" --val-prefix "$UIIS_VAL_PREFIX"
    --epochs "$UIIS_EPOCHS" --max-keep-ckpts "$MAX_KEEP_CKPTS"
    --milestones "$UIIS_LR_MILESTONES")
  [[ -z "$UIIS_LR" ]] || args+=(--lr "$UIIS_LR")
  "$PYTHON_BIN" "$HELPER" "${args[@]}"
  guard_or_install_config "$candidate" "$runtime" "$work"
}

configs_equivalent() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
from mmengine.config import Config

configs = [Config.fromfile(path).to_dict() for path in sys.argv[1:]]
for config in configs:
    config.pop("launcher", None)
    config.pop("work_dir", None)
raise SystemExit(0 if configs[0] == configs[1] else 1)
PY
}

guard_or_install_config() {
  local candidate="$1" runtime="$2" work="$3"
  if find "$work" -maxdepth 1 -type f \( -name 'latest.pth' -o -name 'epoch_*.pth' \) -print -quit | grep -q .; then
    [[ -s "$runtime" ]] || {
      rm -f "$candidate"
      die "checkpoints exist without a saved runtime config; refusing unverifiable resume: $work"
    }
    if ! configs_equivalent "$candidate" "$runtime"; then
      rm -f "$candidate"
      die "config differs from a run that already has checkpoints; use a fresh OUTPUT_ROOT/work directory to change settings: $work"
    fi
    rm -f "$candidate"
  else
    mv -f "$candidate" "$runtime"
  fi
}

train_stage() {
  local stage="$1" runtime="$2" work="$3" group="$4" port="$5" metric="$6" run_test="$7"
  local train_log="$LOG_ROOT/${stage}_train.log" best test_log
  best="$(best_checkpoint "$work" || true)"
  if [[ -s "$work/epoch_${SFUI_EPOCHS}.pth" && "$stage" == sfui ]]; then
    echo "REUSE completed SFUI final epoch checkpoint: $work/epoch_${SFUI_EPOCHS}.pth"
  elif [[ -s "$work/epoch_${UIIS_EPOCHS}.pth" && "$stage" == uiis ]]; then
    echo "REUSE completed UIIS final epoch checkpoint: $work/epoch_${UIIS_EPOCHS}.pth"
  else
    wait_for_gpus "$group" "$stage-train"
    echo "START stage=$stage gpus=$group runtime=$runtime work=$work"
    local -a resume_args=()
    if [[ -s "$work/latest.pth" || -s "$work/last_checkpoint" ]]; then
      resume_args+=(--resume)
    fi
    CUDA_VISIBLE_DEVICES="$group" PORT="$port" \
      bash tools/dist_train.sh "$runtime" "$(gpu_count "$group")" --work-dir "$work" "${resume_args[@]}" \
      2>&1 | tee "$train_log"
  fi

  if [[ "$stage" == sfui ]]; then
    best="$(best_checkpoint "$work" || true)"
    [[ -n "$best" ]] || best="$work/epoch_${SFUI_EPOCHS}.pth"
  else
    best="$(best_checkpoint "$work" || true)"
    [[ -n "$best" ]] || best="$work/epoch_${UIIS_EPOCHS}.pth"
  fi
  [[ -s "$best" ]] || die "$stage checkpoint not found: $best"
  if [[ "$run_test" == 1 ]]; then
    test_log="$LOG_ROOT/${stage}_test.log"
    wait_for_gpus "$group" "$stage-test"
    set +e
    CUDA_VISIBLE_DEVICES="$group" PORT="$((port + 100))" \
      bash tools/dist_test.sh "$runtime" "$best" "$(gpu_count "$group")" \
      2>&1 | tee "$test_log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [[ "$status" != 0 ]] && ! grep -q "${metric}:" "$test_log"; then
      die "$stage evaluation failed (exit=$status) without reporting $metric"
    fi
  fi
  RESULT_CHECKPOINT="$best"
}

echo "SFUI->UIIS Mask R-CNN pipeline model=$MODEL_TAG"
echo "output_root=$OUTPUT_ROOT sfui_gpus=$SFUI_GPUS uiis_gpus=$UIIS_GPUS"
echo "sfui_epochs=$SFUI_EPOCHS sfui_lr=${SFUI_LR:-CONFIG_DEFAULT} uiis_epochs=$UIIS_EPOCHS uiis_lr=${UIIS_LR:-CONFIG_DEFAULT}"
echo "sfui_train_ann=${SFUI_TRAIN_ANN:-NOT_USED} sfui_val_ann=${SFUI_VAL_ANN:-DISABLED} uiis_root=$UIIS_ROOT"
echo "pipeline_log=$PIPELINE_LOG"

if [[ "$CHECK_ONLY" == 1 ]]; then
  "$PYTHON_BIN" -m py_compile "$HELPER"
  if [[ "$RUN_SFUI" == 1 ]]; then
    sfui_work="$WORK_ROOT/sfui_${MODEL_TAG}_maskrcnn_${SFUI_EPOCHS}e"
    mkdir -p "$sfui_work"
    write_sfui_config "$CONVERTED_ROOT/imagenet_${MODEL_TAG}_dino_teacher_backbone.pth" "$sfui_work"
  fi
  if [[ "$RUN_UIIS" == 1 ]]; then
    uiis_work="$WORK_ROOT/uiis_${MODEL_TAG}_maskrcnn_${UIIS_EPOCHS}e"
    mkdir -p "$uiis_work"
    uiis_init="$BACKBONE_ROOT/sfui_${MODEL_TAG}_maskrcnn_backbone.pth"
    write_uiis_config "$uiis_init" "$uiis_work"
  fi
  echo "CHECK_ONLY=1 passed; no training or evaluation was started"
  exit 0
fi

if [[ "$RUN_SFUI" == 1 ]]; then
  prepare_raw_backbone
  model_init="$CONVERTED_ROOT/imagenet_${MODEL_TAG}_dino_teacher_backbone.pth"
  sfui_work="$WORK_ROOT/sfui_${MODEL_TAG}_maskrcnn_${SFUI_EPOCHS}e"
  sfui_runtime="$sfui_work/runtime_config.py"
  mkdir -p "$sfui_work"
  write_sfui_config "$model_init" "$sfui_work"
  train_stage sfui "$sfui_runtime" "$sfui_work" "$SFUI_GPUS" "$BASE_PORT" coco/segm_mAP "$SFUI_RUN_TEST"
  SFUI_CHECKPOINT="$RESULT_CHECKPOINT"
fi

if [[ "$RUN_UIIS" == 1 ]]; then
  [[ -s "$SFUI_CHECKPOINT" ]] || die "SFUI checkpoint is missing: $SFUI_CHECKPOINT"
  adapted="$BACKBONE_ROOT/sfui_${MODEL_TAG}_maskrcnn_backbone.pth"
  if [[ ! -s "$adapted" ]]; then
    "$PYTHON_BIN" "$HELPER" export-backbone --checkpoint "$SFUI_CHECKPOINT" --output "$adapted"
  fi
  uiis_work="$WORK_ROOT/uiis_${MODEL_TAG}_maskrcnn_${UIIS_EPOCHS}e"
  uiis_runtime="$uiis_work/runtime_config.py"
  mkdir -p "$uiis_work"
  write_uiis_config "$adapted" "$uiis_work"
  train_stage uiis "$uiis_runtime" "$uiis_work" "$UIIS_GPUS" "$((BASE_PORT + 10))" coco/segm_mAP "$UIIS_RUN_TEST"
  uiis_checkpoint="$RESULT_CHECKPOINT"
  echo "COMPLETE SFUI_CHECKPOINT=$SFUI_CHECKPOINT UIIS_CHECKPOINT=$uiis_checkpoint BACKBONE=$adapted"
else
  echo "COMPLETE SFUI_CHECKPOINT=$SFUI_CHECKPOINT"
fi
