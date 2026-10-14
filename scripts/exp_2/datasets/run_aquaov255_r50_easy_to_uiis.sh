#!/usr/bin/env bash
# Train Mask R-CNN on AquaOV255 Easy, then fine-tune/evaluate on UIIS10K.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BASE_RUN="${AQUA_EASY_RUN:-/media/HDD2/XCX/exp_2/aquaov255_maskrcnn_runs/aquaov255_reuse_ab_thr070_gpu45_20261010_134254}"
SFUI_DATA_ROOT="${SFUI_DATA_ROOT:-$BASE_RUN/aquaov255_instance_coco}"
SFUI_SPLIT_ROOT="${SFUI_SPLIT_ROOT:-$BASE_RUN/aquaov255_easy_dfui_split85_15_seed42}"
SFUI_TRAIN_ANN="${SFUI_TRAIN_ANN:-$SFUI_SPLIT_ROOT/annotations/instances_train.json}"
SFUI_VAL_ANN="${SFUI_VAL_ANN:-$SFUI_SPLIT_ROOT/annotations/instances_val.json}"
SFUI_TEST_ANN="${SFUI_TEST_ANN:-$SFUI_VAL_ANN}"

DINO_RAW="${DINO_RAW:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/ImageNet/DINO_ResNet50_100e/checkpoint.pth}"
R50_CONFIG="${R50_CONFIG:-configs/exp_2/mask-rcnn_r50_dino-official_fpn_2x_uiis10k_mask.py}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"

SFUI_EPOCHS="${SFUI_EPOCHS:-48}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
SFUI_GPUS="${SFUI_GPUS:-6,7}"
UIIS_GPUS="${UIIS_GPUS:-6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
BASE_PORT="${BASE_PORT:-39800}"
RUN_NAME="${RUN_NAME:-imagenet_r50_aquaov255_thr070_easy_uiis_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/sfui_uiis_runs/$RUN_NAME}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -s "$DINO_RAW" ]] || die "ImageNet DINO ResNet-50 checkpoint not found: $DINO_RAW"
[[ -s "$R50_CONFIG" ]] || die "ResNet-50 Mask R-CNN config not found: $R50_CONFIG"
[[ -s "$SFUI_TRAIN_ANN" && -s "$SFUI_VAL_ANN" ]] || die "AquaOV255 Easy train/val annotation missing"
[[ -d "$SFUI_DATA_ROOT/train/images" ]] || die "AquaOV255 image root missing: $SFUI_DATA_ROOT/train/images"
[[ -s "$UIIS_ROOT/annotations/instances_train.json" ]] || die "UIIS train annotation missing"
[[ -s "$UIIS_ROOT/annotations/instances_val.json" ]] || die "UIIS val annotation missing"
[[ -d "$UIIS_ROOT/train" && -d "$UIIS_ROOT/val" ]] || die "UIIS train/val image directory missing"
[[ -s tools/exp_2/prepare_sfui_uiis_mask_configs.py ]] || die "SFUI/UIIS config helper missing"
[[ -s tools/convert_ssl_backbone_to_mmdet.py ]] || die "DINO checkpoint converter missing"
[[ -s scripts/exp_2/datasets/run_sfui_to_uiis_vits_mask.sh ]] || die "shared SFUI/UIIS runner missing"

mkdir -p "$OUTPUT_ROOT/logs"

echo "AquaOV255 Easy -> UIIS10K Mask R-CNN; backbone=ResNet-50"
echo "ImageNet DINO checkpoint: $DINO_RAW"
echo "R50 config: $R50_CONFIG"
echo "Aqua Easy train/val: $SFUI_TRAIN_ANN / $SFUI_VAL_ANN"
echo "UIIS root: $UIIS_ROOT"
echo "epochs: Aqua=$SFUI_EPOCHS UIIS=$UIIS_EPOCHS; GPUs: Aqua=$SFUI_GPUS UIIS=$UIIS_GPUS"
echo "output_root=$OUTPUT_ROOT"

exec env \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_TAG=r50 \
  DINO_RAW="$DINO_RAW" \
  SFUI_CONFIG="$R50_CONFIG" \
  UIIS_CONFIG="$R50_CONFIG" \
  SFUI_DATA_ROOT="$SFUI_DATA_ROOT" \
  SFUI_TRAIN_ANN="$SFUI_TRAIN_ANN" \
  SFUI_VAL_ANN="$SFUI_VAL_ANN" \
  SFUI_TEST_ANN="$SFUI_TEST_ANN" \
  SFUI_TRAIN_PREFIX=train/images/ \
  SFUI_VAL_PREFIX=train/images/ \
  SFUI_TEST_PREFIX=train/images/ \
  UIIS_ROOT="$UIIS_ROOT" \
  SFUI_EPOCHS="$SFUI_EPOCHS" \
  UIIS_EPOCHS="$UIIS_EPOCHS" \
  SFUI_GPUS="$SFUI_GPUS" \
  UIIS_GPUS="$UIIS_GPUS" \
  MIN_FREE_MIB="$MIN_FREE_MIB" \
  BASE_PORT="$BASE_PORT" \
  SFUI_RUN_TEST="${SFUI_RUN_TEST:-0}" \
  UIIS_RUN_TEST="${UIIS_RUN_TEST:-1}" \
  CHECK_ONLY="${CHECK_ONLY:-0}" \
  bash scripts/exp_2/datasets/run_sfui_to_uiis_vits_mask.sh
