#!/usr/bin/env bash
# Configurable DINO ResNet-50 -> DFUI detector adaptation -> RUOD detector.
# DFUI_ROOT and DFUI_CONFIG are intentionally explicit so alternate DFUI
# variants can be compared without editing the shared nested runner.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PREFIX="${MODEL_PREFIX:-dfui_r50}"
SOURCE_LABEL="${SOURCE_LABEL:-$MODEL_PREFIX}"
SCALE_LABEL="${SCALE_LABEL:-full}"
R50_RAW="${R50_RAW:?set R50_RAW to a completed DINO ResNet-50 checkpoint.pth}"
DFUI_ROOT="${DFUI_ROOT:?set DFUI_ROOT to the DFUI COCO root}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/RUOD/coco}"
R50_DFUI_CONFIG="${R50_DFUI_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_easy_j10_scheme_c_s1.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/SSD1/XCX/exp_2/dfui_ruod_runs/$MODEL_PREFIX}"

[[ -s "$R50_RAW" ]] || { echo "ERROR: missing R50 checkpoint: $R50_RAW" >&2; exit 1; }
[[ -s "$R50_DFUI_CONFIG" ]] || { echo "ERROR: missing DFUI/RUOD config: $R50_DFUI_CONFIG" >&2; exit 1; }
[[ -s "$DFUI_ROOT/annotations/instances_train.json" && -s "$DFUI_ROOT/annotations/instances_val.json" ]] || { echo "ERROR: invalid DFUI_ROOT: $DFUI_ROOT" >&2; exit 1; }
[[ -s "$RUOD_ROOT/annotations/instances_train.json" && -s "$RUOD_ROOT/annotations/instances_val.json" ]] || { echo "ERROR: invalid RUOD_ROOT: $RUOD_ROOT" >&2; exit 1; }
[[ -s "$REPO_ROOT/scripts/exp_2/tri_pretrain/run_nested_scale_downstream.sh" ]] || { echo "ERROR: missing shared DFUI/RUOD runner" >&2; exit 1; }

exec env \
  SOURCE="$SOURCE_LABEL" SCALE="$SCALE_LABEL" ARCHITECTURES=resnet50 \
  R50_RAW="$R50_RAW" VITS_RAW="${VITS_RAW:-$R50_RAW}" \
  R50_DFUI_RUOD_CONFIG="$R50_DFUI_CONFIG" \
  DFUI_RUOD_ROOT="$DFUI_ROOT" RUOD_ROOT="$RUOD_ROOT" \
  RUN_DIRECT=0 RUN_DFUI=1 VARIANTS=dfui_ruod DFUI_FOLLOWUPS=ruod \
  OUTPUT_ROOT="$OUTPUT_ROOT" WORK_ROOT="${WORK_ROOT:-$OUTPUT_ROOT/work_dirs}" \
  BACKBONE_ROOT="${BACKBONE_ROOT:-$OUTPUT_ROOT/backbones}" \
  CONVERTED_ROOT="${CONVERTED_ROOT:-$OUTPUT_ROOT/converted}" \
  LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}" \
  R50_GPUS="${R50_GPUS:-4,5}" BASE_PORT="${BASE_PORT:-30600}" \
  DFUI_EPOCHS="${DFUI_EPOCHS:-48}" RUN_TEST="${RUN_TEST:-1}" \
  SKIP_COMPLETED="${SKIP_COMPLETED:-1}" CHECK_ONLY="${CHECK_ONLY:-0}" \
  MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}" \
  bash "$SCRIPT_DIR/run_nested_scale_downstream.sh"
