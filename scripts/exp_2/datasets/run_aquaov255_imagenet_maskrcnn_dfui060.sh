#!/usr/bin/env bash
# Train an ImageNet-initialized Mask R-CNN on AquaOV255's training partition,
# then create a DFUI-like high-confidence pseudo-instance set from the held-out
# partition only.  Ground-truth validation masks are never read by selection.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

AQUA_ROOT="${AQUA_ROOT:?set AQUA_ROOT to the prepared AquaOV255 root containing images/ and masks/}"
MASK_CONFIG="${MASK_CONFIG:?set MASK_CONFIG to an MMDetection ResNet-50 Mask R-CNN config}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new experiment output directory}"
GPU_GROUP="${GPU_GROUP:-2,3}"
BASE_PORT="${BASE_PORT:-38400}"
EPOCHS="${EPOCHS:-24}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.60}"
VAL_RATIO="${VAL_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-20261006}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# MMDetection's standard ImageNet-1K ResNet-50 initialization identifier.
IMAGENET_INIT="${IMAGENET_INIT:-torchvision://resnet50}"

die() { echo "ERROR: $*" >&2; exit 1; }
gpu_count() { awk -F, '{print NF}' <<< "$1"; }
best_checkpoint() { find "$1" -maxdepth 1 -type f -name 'best_coco_segm_mAP_epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-; }

[ -d "$AQUA_ROOT/images" ] || die "missing AquaOV255 images: $AQUA_ROOT/images"
[ -d "$AQUA_ROOT/masks" ] || die "missing AquaOV255 masks: $AQUA_ROOT/masks"
[ -s "$MASK_CONFIG" ] || die "missing Mask R-CNN config: $MASK_CONFIG"
[ -x tools/dist_train.sh ] || die "missing tools/dist_train.sh"
[ -x tools/dist_test.sh ] || die "missing tools/dist_test.sh"
case "$EPOCHS" in ''|*[!0-9]*) die "EPOCHS must be a positive integer" ;; esac
case "$SCORE_THRESHOLD" in '') die "SCORE_THRESHOLD must be set" ;; esac

COCO_ROOT="$OUTPUT_ROOT/aquaov255_instance_coco"
WORK_DIR="$OUTPUT_ROOT/work_dirs/maskrcnn_r50_imagenet_aquaov255_${EPOCHS}e"
RUNTIME_CONFIG="$WORK_DIR/runtime_config.py"
PSEUDO_ROOT="$OUTPUT_ROOT/aquaov255_dfui_score${SCORE_THRESHOLD/./p}"
LOG_ROOT="$OUTPUT_ROOT/logs"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo "AquaOV255 ImageNet Mask R-CNN -> held-out DFUI-style score filter"
echo "aqua_root=$AQUA_ROOT"
echo "mask_config=$MASK_CONFIG"
echo "imagenet_init=$IMAGENET_INIT"
echo "gpu_group=$GPU_GROUP"
echo "epochs=$EPOCHS"
echo "score_threshold=$SCORE_THRESHOLD"
echo "val_ratio=$VAL_RATIO seed=$SPLIT_SEED"
echo "output_root=$OUTPUT_ROOT"
echo "============================================================"

python tools/exp_2/build_aquaov255_instance_coco.py \
  --images "$AQUA_ROOT/images" \
  --masks "$AQUA_ROOT/masks" \
  --categories "$AQUA_ROOT/category.txt" \
  --output "$COCO_ROOT" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SPLIT_SEED" \
  --check-only

if [ "$CHECK_ONLY" = 1 ]; then
  echo "CHECK_ONLY=1 passed"
  exit 0
fi

if [ ! -f "$COCO_ROOT/.complete" ]; then
  python tools/exp_2/build_aquaov255_instance_coco.py \
    --images "$AQUA_ROOT/images" \
    --masks "$AQUA_ROOT/masks" \
    --categories "$AQUA_ROOT/category.txt" \
    --output "$COCO_ROOT" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SPLIT_SEED" \
    --storage-mode symlink
fi

python - "$MASK_CONFIG" "$RUNTIME_CONFIG" "$COCO_ROOT" "$IMAGENET_INIT" "$EPOCHS" "$MAX_KEEP_CKPTS" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

source, destination, root, init, epochs, max_keep = sys.argv[1:]
source = Path(source)
destination = Path(destination)
root = Path(root)
cfg = Config.fromfile(source)

categories = __import__('json').loads(
    (root / 'train' / 'annotations' / 'instances_train.json').read_text(encoding='utf-8')
)['categories']
classes = tuple(item['name'] for item in categories)
if len(classes) != 254:
    raise SystemExit(f'Expected 254 AquaOV255 foreground classes, got {len(classes)}')

def configure_dataset(dataset, split):
    if 'dataset' in dataset:
        configure_dataset(dataset['dataset'], split)
        return
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            configure_dataset(child, split)
        return
    dataset['data_root'] = f'{root}/'
    dataset['ann_file'] = str(root / split / 'annotations' / f'instances_{split}.json')
    dataset['data_prefix'] = dict(img=f'{split}/images/')
    dataset['metainfo'] = dict(classes=classes)

configure_dataset(cfg.train_dataloader['dataset'], 'train')
configure_dataset(cfg.val_dataloader['dataset'], 'val')
configure_dataset(cfg.test_dataloader['dataset'], 'val')
for name in ('val_evaluator', 'test_evaluator'):
    evaluator = cfg.get(name)
    if evaluator is not None:
        evaluator['ann_file'] = str(root / 'val' / 'annotations' / 'instances_val.json')

roi_head = cfg.model.roi_head
roi_head.bbox_head['num_classes'] = len(classes)
roi_head.mask_head['num_classes'] = len(classes)
cfg.model.backbone['init_cfg'] = dict(type='Pretrained', checkpoint=init)
cfg.load_from = None
cfg.train_cfg.max_epochs = int(epochs)
cfg.train_cfg.val_interval = 1
cfg.default_hooks.checkpoint['save_best'] = 'coco/segm_mAP'
cfg.default_hooks.checkpoint['max_keep_ckpts'] = int(max_keep)
destination.parent.mkdir(parents=True, exist_ok=True)
cfg.dump(destination)
print(f'Wrote runtime config: {destination}')
print(f'classes={len(classes)} init={init}')
PY

BEST="$(best_checkpoint "$WORK_DIR" || true)"
if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$WORK_DIR/.complete" ] && [ -n "$BEST" ]; then
  echo "REUSE completed training: $BEST"
else
  mkdir -p "$WORK_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_GROUP" PORT="$BASE_PORT" \
    bash tools/dist_train.sh "$RUNTIME_CONFIG" "$(gpu_count "$GPU_GROUP")" --work-dir "$WORK_DIR" \
    2>&1 | tee "$LOG_ROOT/maskrcnn_train.log"
  BEST="$(best_checkpoint "$WORK_DIR" || true)"
  [ -n "$BEST" ] || die "No best coco/segm_mAP checkpoint was created"
  if [ "$RUN_TEST" = 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPU_GROUP" PORT="$((BASE_PORT + 100))" \
      bash tools/dist_test.sh "$RUNTIME_CONFIG" "$BEST" "$(gpu_count "$GPU_GROUP")" \
      2>&1 | tee "$LOG_ROOT/maskrcnn_val_test.log"
  fi
  touch "$WORK_DIR/.complete"
fi

if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$PSEUDO_ROOT/.complete" ]; then
  echo "REUSE completed held-out pseudo-label set: $PSEUDO_ROOT"
else
  # The builder sees the held-out image directory only.  It never accepts the
  # ground-truth COCO JSON, so native validation masks cannot influence filtering.
  CUDA_VISIBLE_DEVICES="${GPU_GROUP%%,*}" \
    python tools/exp_2/build_aquaov255_maskrcnn_pseudolabels.py \
      --images "$COCO_ROOT/val/images" \
      --config "$RUNTIME_CONFIG" \
      --checkpoint "$BEST" \
      --output "$PSEUDO_ROOT" \
      --score-thr "$SCORE_THRESHOLD" \
      --device cuda:0 \
      --storage-mode symlink \
      2>&1 | tee "$LOG_ROOT/heldout_score_filter.log"
fi

printf 'field\tvalue\n' > "$OUTPUT_ROOT/run_provenance.tsv"
printf 'aqua_root\t%s\n' "$AQUA_ROOT" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'mask_config\t%s\n' "$MASK_CONFIG" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'imagenet_init\t%s\n' "$IMAGENET_INIT" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'epochs\t%s\n' "$EPOCHS" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'score_threshold\t%s\n' "$SCORE_THRESHOLD" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'teacher_checkpoint\t%s\n' "$BEST" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'selection_source\theld-out AquaOV255 validation partition only\n' >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'ground_truth_masks_used_for_selection\tfalse\n' >> "$OUTPUT_ROOT/run_provenance.tsv"

echo "COMPLETE output_root=$OUTPUT_ROOT"
