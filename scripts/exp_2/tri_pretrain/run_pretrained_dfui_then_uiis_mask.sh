#!/usr/bin/env bash
# Adapt a completed DINO ResNet-50 / ViT-S pair on DFUI detection, then
# transfer only the detector backbones to UIIS10K Mask R-CNN.
#
# Required inputs are MODEL_PREFIX, R50_RAW, and VITS_RAW.  The same entrypoint
# supports ImageNet, RealUW, Synthetic5, controlled-scale, and future sources.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PREFIX="${MODEL_PREFIX:?set MODEL_PREFIX, for example imagenet1k_full}"
R50_RAW="${R50_RAW:?set R50_RAW to a completed DINO ResNet-50 checkpoint.pth}"
VITS_RAW="${VITS_RAW:?set VITS_RAW to a completed DINO ViT-S checkpoint.pth}"

# Labels only identify artifacts and logs. They do not alter the selected raw
# checkpoint or downstream configuration.
SOURCE_LABEL="${SOURCE_LABEL:-$MODEL_PREFIX}"
SCALE_LABEL="${SCALE_LABEL:-full}"
VARIANTS="${VARIANTS:-dfui_ruod}"
DFUI_FOLLOWUPS="${DFUI_FOLLOWUPS:-mask}"
ALLOW_TARGET_DOMAIN_UIIS="${ALLOW_TARGET_DOMAIN_UIIS:-0}"

R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-30600}"
DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/SSD1/XCX/exp_2/dfui_uiis_mask_runs/${MODEL_PREFIX}}"
WORK_ROOT="${WORK_ROOT:-$OUTPUT_ROOT/work_dirs}"
BACKBONE_ROOT="${BACKBONE_ROOT:-$OUTPUT_ROOT/backbones}"
CONVERTED_ROOT="${CONVERTED_ROOT:-$OUTPUT_ROOT/converted}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log}"

DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

case ",$VARIANTS," in
  *,dfui_ruod_uiis,*)
    [ "$ALLOW_TARGET_DOMAIN_UIIS" = "1" ] || {
      echo "ERROR: dfui_ruod_uiis includes UIIS Easy images in the intermediate detector stage." >&2
      echo "Set ALLOW_TARGET_DOMAIN_UIIS=1 only after auditing UIIS10K split overlap." >&2
      exit 2
    }
    ;;
esac

case ",$DFUI_FOLLOWUPS," in
  *,mask,*) ;;
  *)
    echo "ERROR: DFUI_FOLLOWUPS must include mask for this entrypoint." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT" "$WORK_ROOT" "$BACKBONE_ROOT" "$CONVERTED_ROOT" "$LOG_ROOT"

{
  printf 'field\tvalue\n'
  printf 'model_prefix\t%s\n' "$MODEL_PREFIX"
  printf 'source_label\t%s\n' "$SOURCE_LABEL"
  printf 'scale_label\t%s\n' "$SCALE_LABEL"
  printf 'r50_raw\t%s\n' "$R50_RAW"
  printf 'vits_raw\t%s\n' "$VITS_RAW"
  printf 'dfui_variants\t%s\n' "$VARIANTS"
  printf 'dfui_epochs\t%s\n' "$DFUI_EPOCHS"
  printf 'uiis_mask_epochs\t24\n'
  printf 'r50_gpus\t%s\n' "$R50_GPUS"
  printf 'vits_gpus\t%s\n' "$VITS_GPUS"
} > "$OUTPUT_ROOT/run_provenance.tsv"

echo "============================================================"
echo "DINO -> DFUI detection -> UIIS10K Mask R-CNN pipeline"
echo "model_prefix=$MODEL_PREFIX"
echo "source_label=$SOURCE_LABEL scale_label=$SCALE_LABEL"
echo "variants=$VARIANTS"
echo "r50_raw=$R50_RAW"
echo "vits_raw=$VITS_RAW"
echo "output_root=$OUTPUT_ROOT"
echo "pipeline_log=$PIPELINE_LOG"
echo "============================================================"

env \
  SOURCE="$SOURCE_LABEL" \
  SCALE="$SCALE_LABEL" \
  R50_RAW="$R50_RAW" \
  VITS_RAW="$VITS_RAW" \
  R50_GPUS="$R50_GPUS" \
  VITS_GPUS="$VITS_GPUS" \
  BASE_PORT="$BASE_PORT" \
  WORK_ROOT="$WORK_ROOT" \
  BACKBONE_ROOT="$BACKBONE_ROOT" \
  CONVERTED_ROOT="$CONVERTED_ROOT" \
  LOG_ROOT="$LOG_ROOT" \
  PIPELINE_LOG="$PIPELINE_LOG" \
  RUOD_ROOT="${RUOD_ROOT:-$DATA_ROOT/RUOD/coco}" \
  UIIS_ROOT="${UIIS_ROOT:-$DATA_ROOT/UIIS10K/coco}" \
  DFUI_RUOD_ROOT="${DFUI_RUOD_ROOT:-$DATA_ROOT/DFUI_RUOD_EASY}" \
  DFUI_RUOD_UIIS_ROOT="${DFUI_RUOD_UIIS_ROOT:-$DATA_ROOT/DFUI_RUOD_UIIS_EASY}" \
  RUN_DIRECT=0 \
  RUN_DFUI=1 \
  DFUI_FOLLOWUPS="$DFUI_FOLLOWUPS" \
  DFUI_EPOCHS="$DFUI_EPOCHS" \
  MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
  RUN_TEST="$RUN_TEST" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  CHECK_ONLY="$CHECK_ONLY" \
  VARIANTS="$VARIANTS" \
  bash "$SCRIPT_DIR/run_nested_scale_downstream.sh"
