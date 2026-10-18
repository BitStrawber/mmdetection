#!/usr/bin/env bash
# Configurable ResNet-50 SFUI -> UIIS10K Mask R-CNN pipeline.
# Override SFUI_CONFIG, UIIS_CONFIG, SFUI_DATA_ROOT, SFUI_TRAIN_ANN,
# UIIS_ROOT, DINO_RAW, SFUI/UIIS epochs, GPU groups, and BASE_PORT as needed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BASE_RUN="${AQUA_EASY_RUN:-}"
SFUI_DATA_ROOT="${SFUI_DATA_ROOT:-${BASE_RUN:+$BASE_RUN/aquaov255_instance_coco}}"
SFUI_SPLIT_ROOT="${SFUI_SPLIT_ROOT:-${BASE_RUN:+$BASE_RUN/aquaov255_easy_dfui_split85_15_seed42}}"
SFUI_TRAIN_ANN="${SFUI_TRAIN_ANN:-${SFUI_SPLIT_ROOT:+$SFUI_SPLIT_ROOT/annotations/instances_train.json}}"
SFUI_VAL_ANN="${SFUI_VAL_ANN:-${SFUI_SPLIT_ROOT:+$SFUI_SPLIT_ROOT/annotations/instances_val.json}}"
SFUI_TEST_ANN="${SFUI_TEST_ANN:-$SFUI_VAL_ANN}"

DINO_RAW="${DINO_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/ImageNet/DINO_ResNet50_100e/checkpoint.pth}"
R50_CONFIG="${R50_CONFIG:-configs/exp_2/mask-rcnn_r50_dino-official_fpn_2x_uiis10k_mask.py}"
UIIS_CONFIG="${UIIS_CONFIG:-$R50_CONFIG}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"

SFUI_EPOCHS="${SFUI_EPOCHS:-48}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
SFUI_GPUS="${SFUI_GPUS:-6,7}"
UIIS_GPUS="${UIIS_GPUS:-6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
BASE_PORT="${BASE_PORT:-39800}"
RUN_NAME="${RUN_NAME:-sfui_r50_uiis_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/sfui_uiis_runs/$RUN_NAME}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -s "$DINO_RAW" ]] || die "missing DINO checkpoint: $DINO_RAW"
[[ -s "$R50_CONFIG" && -s "$UIIS_CONFIG" ]] || die "missing R50/UIIS config"
[[ -d "$SFUI_DATA_ROOT" && -s "$SFUI_TRAIN_ANN" ]] || die "missing SFUI data or annotation"
[[ -s "$UIIS_ROOT/annotations/instances_train.json" && -s "$UIIS_ROOT/annotations/instances_val.json" ]] || die "missing UIIS annotations"
[[ -d "$UIIS_ROOT/train" && -d "$UIIS_ROOT/val" ]] || die "missing UIIS train/val image directories"
[[ -s tools/exp_2/prepare_sfui_uiis_mask_configs.py && -s tools/convert_ssl_backbone_to_mmdet.py ]] || die "missing shared SFUI/UIIS tools"
[[ -s "$SCRIPT_DIR/run_sfui_to_uiis_vits_mask.sh" ]] || die "missing shared SFUI/UIIS runner"

mkdir -p "$OUTPUT_ROOT/logs"
echo "SFUI -> UIIS10K Mask R-CNN; model=resnet50"
echo "sfui_config=$R50_CONFIG"
echo "uiis_config=$UIIS_CONFIG"
echo "sfui_data_root=$SFUI_DATA_ROOT"
echo "sfui_train_ann=$SFUI_TRAIN_ANN"
echo "uiis_root=$UIIS_ROOT"
echo "dino_raw=$DINO_RAW"
echo "output_root=$OUTPUT_ROOT"

exec env \
  OUTPUT_ROOT="$OUTPUT_ROOT" MODEL_TAG=r50 DINO_RAW="$DINO_RAW" \
  SFUI_CONFIG="$R50_CONFIG" UIIS_CONFIG="$UIIS_CONFIG" \
  SFUI_DATA_ROOT="$SFUI_DATA_ROOT" SFUI_TRAIN_ANN="$SFUI_TRAIN_ANN" \
  SFUI_VAL_ANN="$SFUI_VAL_ANN" SFUI_TEST_ANN="$SFUI_TEST_ANN" \
  SFUI_TRAIN_PREFIX="${SFUI_TRAIN_PREFIX:-train/images/}" \
  SFUI_VAL_PREFIX="${SFUI_VAL_PREFIX:-train/images/}" \
  SFUI_TEST_PREFIX="${SFUI_TEST_PREFIX:-train/images/}" \
  UIIS_ROOT="$UIIS_ROOT" SFUI_EPOCHS="$SFUI_EPOCHS" UIIS_EPOCHS="$UIIS_EPOCHS" \
  SFUI_GPUS="$SFUI_GPUS" UIIS_GPUS="$UIIS_GPUS" MIN_FREE_MIB="$MIN_FREE_MIB" \
  BASE_PORT="$BASE_PORT" SFUI_RUN_TEST="${SFUI_RUN_TEST:-0}" \
  UIIS_RUN_TEST="${UIIS_RUN_TEST:-1}" CHECK_ONLY="${CHECK_ONLY:-0}" \
  bash "$SCRIPT_DIR/run_sfui_to_uiis_vits_mask.sh"
