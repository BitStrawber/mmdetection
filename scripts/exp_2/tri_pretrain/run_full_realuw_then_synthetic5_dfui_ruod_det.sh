#!/usr/bin/env bash
# Run the full-scale RealUW and corrected Synthetic5 DINO transfer experiments.
# Each architecture owns one GPU pair and processes RealUW before Synthetic5.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
RUN_NAME="${RUN_NAME:-full_realuw_then_synthetic5_corrected_dfuiuiis_s1_ruod_s2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/media/SSD1/XCX/exp_2/dfui_training_runs/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"

REALUW_R50_RAW="${REALUW_R50_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/RealUW/DINO_ResNet50_100e/checkpoint.pth}"
REALUW_VITS_RAW="${REALUW_VITS_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/RealUW/DINO_ViTS_100e/checkpoint.pth}"
SYNTHETIC5_R50_RAW="${SYNTHETIC5_R50_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/Synthetic5Merged_CorrectedWaterGAN/DINO_ResNet50_100e/checkpoint.pth}"
SYNTHETIC5_VITS_RAW="${SYNTHETIC5_VITS_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/Synthetic5Merged_CorrectedWaterGAN/DINO_ViTS_100e/checkpoint.pth}"

R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-6,7}"
R50_REALUW_PORT="${R50_REALUW_PORT:-30800}"
R50_SYNTHETIC5_PORT="${R50_SYNTHETIC5_PORT:-30900}"
VITS_REALUW_PORT="${VITS_REALUW_PORT:-31000}"
VITS_SYNTHETIC5_PORT="${VITS_SYNTHETIC5_PORT:-31100}"

DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
RUOD_EPOCHS="${RUOD_EPOCHS:-24}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME="${RESUME:-1}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-400}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"
CHECK_ONLY="${CHECK_ONLY:-0}"
TMPDIR="${TMPDIR:-/media/SSD1/tmp}"

PIPELINE="scripts/exp_2/tri_pretrain/run_imagenet1k_dfui_then_ruod_det.sh"
mkdir -p "$LOG_ROOT" "$TMPDIR"
export TMPDIR

MASTER_LOG="${MASTER_LOG:-$LOG_ROOT/master_pipeline.log}"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() { echo "ERROR: $*" >&2; exit 1; }

require_file() {
  [ -s "$1" ] || die "Required file is missing or empty: $1"
}

write_provenance() {
  local target="$RUN_ROOT/pretraining_provenance.tsv"
  printf '%s\n' 'source	backbone	raw_checkpoint	sha256' > "$target"
  printf '%s\t%s\t%s\t%s\n' \
    full_realuw resnet50 "$REALUW_R50_RAW" "$(sha256sum "$REALUW_R50_RAW" | awk '{print $1}')" \
    >> "$target"
  printf '%s\t%s\t%s\t%s\n' \
    full_realuw vits "$REALUW_VITS_RAW" "$(sha256sum "$REALUW_VITS_RAW" | awk '{print $1}')" \
    >> "$target"
  printf '%s\t%s\t%s\t%s\n' \
    synthetic5_merged_corrected_watergan resnet50 "$SYNTHETIC5_R50_RAW" "$(sha256sum "$SYNTHETIC5_R50_RAW" | awk '{print $1}')" \
    >> "$target"
  printf '%s\t%s\t%s\t%s\n' \
    synthetic5_merged_corrected_watergan vits "$SYNTHETIC5_VITS_RAW" "$(sha256sum "$SYNTHETIC5_VITS_RAW" | awk '{print $1}')" \
    >> "$target"
  echo "Pretraining provenance: $target"
}

run_stage() {
  local source_name="$1" architecture="$2" gpu_group="$3" raw_checkpoint="$4" port="$5"
  local source_root="$RUN_ROOT/$source_name"
  local architecture_log_root="$source_root/logs/$architecture"
  local -a raw_args=()

  case "$architecture" in
    resnet50) raw_args=("IMAGENET1K_R50_RAW=$raw_checkpoint") ;;
    vits) raw_args=("IMAGENET1K_VITS_RAW=$raw_checkpoint") ;;
    *) die "Unsupported architecture: $architecture" ;;
  esac

  echo "================================================================"
  echo "SOURCE=$source_name ARCHITECTURE=$architecture GPU_GROUP=$gpu_group"
  echo "s1: DFUI + RUOD Easy + UIIS Easy, $DFUI_EPOCHS epochs"
  echo "s2: RUOD, $RUOD_EPOCHS epochs"

  env \
    ARCHITECTURES="$architecture" \
    VARIANTS=dfui_ruod_uiis \
    MODEL_PREFIX="$source_name" \
    PRETRAIN_PREFIX="$source_name" \
    DATA_ROOT="$DATA_ROOT" \
    "${raw_args[@]}" \
    OUTPUT_ROOT="$source_root" \
    WORK_ROOT="$source_root/work_dirs" \
    BACKBONE_ROOT="$source_root/backbones" \
    PRETRAIN_DIR="$source_root/converted" \
    LOG_ROOT="$architecture_log_root" \
    PIPELINE_LOG="$architecture_log_root/pipeline.log" \
    R50_GPUS="$gpu_group" \
    VITS_GPUS="$gpu_group" \
    BASE_PORT="$port" \
    DFUI_EPOCHS="$DFUI_EPOCHS" \
    RUOD_EPOCHS="$RUOD_EPOCHS" \
    RUN_TEST="$RUN_TEST" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    RESUME="$RESUME" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
    GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    CHECK_ONLY="$CHECK_ONLY" \
    TMPDIR="$TMPDIR" \
    bash "$PIPELINE"
}

run_r50_chain() {
  run_stage full_realuw resnet50 "$R50_GPUS" "$REALUW_R50_RAW" "$R50_REALUW_PORT"
  run_stage synthetic5_merged_corrected_watergan resnet50 "$R50_GPUS" "$SYNTHETIC5_R50_RAW" "$R50_SYNTHETIC5_PORT"
}

run_vits_chain() {
  run_stage full_realuw vits "$VITS_GPUS" "$REALUW_VITS_RAW" "$VITS_REALUW_PORT"
  run_stage synthetic5_merged_corrected_watergan vits "$VITS_GPUS" "$SYNTHETIC5_VITS_RAW" "$VITS_SYNTHETIC5_PORT"
}

require_file "$PIPELINE"
require_file "$REALUW_R50_RAW"
require_file "$REALUW_VITS_RAW"
require_file "$SYNTHETIC5_R50_RAW"
require_file "$SYNTHETIC5_VITS_RAW"
write_provenance

echo "RUN_ROOT=$RUN_ROOT"
echo "R50 chain: $R50_GPUS; RealUW -> Synthetic5Merged_CorrectedWaterGAN"
echo "ViT-S chain: $VITS_GPUS; RealUW -> Synthetic5Merged_CorrectedWaterGAN"

run_r50_chain &
R50_PID=$!
run_vits_chain &
VITS_PID=$!

status=0
wait "$R50_PID" || status=1
wait "$VITS_PID" || status=1
[ "$status" -eq 0 ] || die "At least one architecture chain failed."

echo "COMPLETE: $RUN_ROOT"
