#!/bin/bash
set -euo pipefail

# Extract random10 feature maps:
#   - ImageNet images with torchvision ImageNet-supervised ResNet-50.
#   - RUOD images with a supervised Cascade R-CNN ResNet-50 backbone checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

IMAGENET_ROOT="${IMAGENET_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K/imagefolder/train}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD}"
CASCADE_CONFIG="${CASCADE_CONFIG:-}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-}"
OUT_DIR="${OUT_DIR:-work_dirs/exp_2/feature_maps/random10_imagenet_ruod}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
RUOD_MAX_SIDE="${RUOD_MAX_SIDE:-1333}"

if [ -z "$CASCADE_CONFIG" ] || [ -z "$CASCADE_CHECKPOINT" ]; then
    echo "Error: CASCADE_CONFIG and CASCADE_CHECKPOINT are required."
    echo
    echo "Example:"
    echo "  CASCADE_CONFIG=configs/exp_2/.../your_cascade_ruod.py \\"
    echo "  CASCADE_CHECKPOINT=work_dirs/.../best_coco_bbox_mAP_epoch_24.pth \\"
    echo "  bash scripts/exp_2/features/extract_random10_resnet50_feature_maps.sh"
    exit 1
fi

python tools/exp_2/extract_random10_resnet50_feature_maps.py \
    --imagenet-root "$IMAGENET_ROOT" \
    --ruod-root "$RUOD_ROOT" \
    --cascade-config "$CASCADE_CONFIG" \
    --cascade-checkpoint "$CASCADE_CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --ruod-max-side "$RUOD_MAX_SIDE"
