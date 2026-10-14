#!/usr/bin/env bash
# Configurable DINO ViT-S -> DFUI detector adaptation -> RUOD detector.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PREFIX="${MODEL_PREFIX:-dfui_vits}"
SOURCE_LABEL="${SOURCE_LABEL:-$MODEL_PREFIX}"
SCALE_LABEL="${SCALE_LABEL:-full}"
VITS_RAW="${VITS_RAW:?set VITS_RAW to a completed DINO ViT-S checkpoint.pth}"
DFUI_ROOT="${DFUI_ROOT:?set DFUI_ROOT to the DFUI COCO root}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/RUOD/coco}"
VITS_DFUI_CONFIG="${VITS_DFUI_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/SSD1/XCX/exp_2/dfui_ruod_runs/$MODEL_PREFIX}"

[[ -s "$VITS_RAW" ]] || { echo "ERROR: missing ViT-S checkpoint: $VITS_RAW" >&2; exit 1; }
[[ -s "$VITS_DFUI_CONFIG" ]] || { echo "ERROR: missing DFUI/RUOD config: $VITS_DFUI_CONFIG" >&2; exit 1; }
[[ -s "$DFUI_ROOT/annotations/instances_train.json" && -s "$DFUI_ROOT/annotations/instances_val.json" ]] || { echo "ERROR: invalid DFUI_ROOT: $DFUI_ROOT" >&2; exit 1; }
[[ -s "$RUOD_ROOT/annotations/instances_train.json" && -s "$RUOD_ROOT/annotations/instances_val.json" ]] || { echo "ERROR: invalid RUOD_ROOT: $RUOD_ROOT" >&2; exit 1; }
[[ -s "$REPO_ROOT/scripts/exp_2/tri_pretrain/run_nested_scale_downstream.sh" ]] || { echo "ERROR: missing shared DFUI/RUOD runner" >&2; exit 1; }

exec env \
  SOURCE="$SOURCE_LABEL" SCALE="$SCALE_LABEL" ARCHITECTURES=vits \
  R50_RAW="${R50_RAW:-$VITS_RAW}" VITS_RAW="$VITS_RAW" \
  VITS_DFUI_CONFIG="$VITS_DFUI_CONFIG" \
  DFUI_RUOD_ROOT="$DFUI_ROOT" RUOD_ROOT="$RUOD_ROOT" \
  RUN_DIRECT=0 RUN_DFUI=1 VARIANTS=dfui_ruod DFUI_FOLLOWUPS=ruod \
  OUTPUT_ROOT="$OUTPUT_ROOT" WORK_ROOT="${WORK_ROOT:-$OUTPUT_ROOT/work_dirs}" \
  BACKBONE_ROOT="${BACKBONE_ROOT:-$OUTPUT_ROOT/backbones}" \
  CONVERTED_ROOT="${CONVERTED_ROOT:-$OUTPUT_ROOT/converted}" \
  LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}" \
  VITS_GPUS="${VITS_GPUS:-6,7}" BASE_PORT="${BASE_PORT:-30600}" \
  DFUI_EPOCHS="${DFUI_EPOCHS:-48}" RUN_TEST="${RUN_TEST:-1}" \
  SKIP_COMPLETED="${SKIP_COMPLETED:-1}" CHECK_ONLY="${CHECK_ONLY:-0}" \
  MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}" \
  bash "$SCRIPT_DIR/run_nested_scale_downstream.sh"
