#!/usr/bin/env bash
# RUOD-style 1:1 A/B cross-filtering with Mask R-CNN, followed by Cascade Mask R-CNN.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

AQUA_ROOT="${AQUA_ROOT:?set AQUA_ROOT to AquaOV255 root containing images/, masks/, category.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to an experiment output directory}"
GPU_GROUP="${GPU_GROUP:-0,1}"
BASE_PORT="${BASE_PORT:-38600}"
MASK_CONFIG="${MASK_CONFIG:-configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py}"
CASCADE_CONFIG="${CASCADE_CONFIG:-configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py}"
IMAGENET_INIT="${IMAGENET_INIT:-torchvision://resnet50}"
MASK_EPOCHS="${MASK_EPOCHS:-24}"
CASCADE_EPOCHS="${CASCADE_EPOCHS:-24}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.60}"
VAL_RATIO="${VAL_RATIO:-0.10}"
EASY_VAL_RATIO="${EASY_VAL_RATIO:-0.10}"
EASY_SPLIT_SEED="${EASY_SPLIT_SEED:-20261007}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-20261006}"
AB_SEED="${AB_SEED:-42}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECK_ONLY="${CHECK_ONLY:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_TEST="${RUN_TEST:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }
gpu_count() { awk -F, '{print NF}' <<< "$1"; }
best_checkpoint() {
  find "$1" -maxdepth 1 -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 1 | cut -d' ' -f2-
}

[[ -d "$AQUA_ROOT/images" ]] || die "missing images directory: $AQUA_ROOT/images"
[[ -d "$AQUA_ROOT/masks" ]] || die "missing masks directory: $AQUA_ROOT/masks"
[[ -s "$AQUA_ROOT/category.txt" ]] || die "missing category.txt: $AQUA_ROOT/category.txt"
[[ -s "$MASK_CONFIG" ]] || die "missing Mask R-CNN config: $MASK_CONFIG"
[[ -s "$CASCADE_CONFIG" ]] || die "missing Cascade Mask R-CNN config: $CASCADE_CONFIG"
[[ -x tools/dist_train.sh && -x tools/dist_test.sh ]] || die "MMDetection distributed train/test scripts are missing"

COCO_ROOT="$OUTPUT_ROOT/aquaov255_instance_coco"
AB_ROOT="$OUTPUT_ROOT/ab_split"
EASY_ROOT="$OUTPUT_ROOT/aquaov255_easy"
EASY_SPLIT_ROOT="$OUTPUT_ROOT/aquaov255_easy_split10"
WORK_ROOT="$OUTPUT_ROOT/work_dirs"
LOG_ROOT="$OUTPUT_ROOT/logs"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "============================================================"
echo "AquaOV255 A/B cross-filter -> two Cascade Mask R-CNN variants"
echo "aqua_root=$AQUA_ROOT"
echo "gpu_group=$GPU_GROUP"
echo "mask_config=$MASK_CONFIG epochs=$MASK_EPOCHS"
echo "cascade_mask_config=$CASCADE_CONFIG epochs=$CASCADE_EPOCHS"
echo "imagenet_init=$IMAGENET_INIT"
echo "per_image_bbox_map_threshold=$SCORE_THRESHOLD"
echo "source_validation_ratio=$VAL_RATIO data_seed=$DATA_SPLIT_SEED ab_seed=$AB_SEED"
echo "easy_validation_ratio=$EASY_VAL_RATIO easy_split_seed=$EASY_SPLIT_SEED"
echo "output_root=$OUTPUT_ROOT"
echo "============================================================"

python tools/exp_2/build_aquaov255_instance_coco.py \
  --images "$AQUA_ROOT/images" \
  --masks "$AQUA_ROOT/masks" \
  --categories "$AQUA_ROOT/category.txt" \
  --output "$COCO_ROOT" \
  --val-ratio "$VAL_RATIO" \
  --seed "$DATA_SPLIT_SEED" \
  --check-only

if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "CHECK_ONLY=1 passed; no training or inference was started"
  exit 0
fi

if [[ ! -f "$COCO_ROOT/.complete" ]]; then
  python tools/exp_2/build_aquaov255_instance_coco.py \
    --images "$AQUA_ROOT/images" \
    --masks "$AQUA_ROOT/masks" \
    --categories "$AQUA_ROOT/category.txt" \
    --output "$COCO_ROOT" \
    --val-ratio "$VAL_RATIO" \
    --seed "$DATA_SPLIT_SEED" \
    --storage-mode symlink
fi

if [[ ! -f "$AB_ROOT/.complete" ]]; then
  python tools/exp_2/aquaov255_cross_easy.py split \
    --annotation "$COCO_ROOT/train/annotations/instances_train.json" \
    --output "$AB_ROOT" \
    --seed "$AB_SEED"
fi

write_runtime_config() {
  local kind="$1" source_config="$2" output_config="$3" train_ann="$4" val_ann="$5" \
    train_prefix="$6" val_prefix="$7" epochs="$8" save_best="$9" validation_enabled="${10}" \
    test_ann="${11}" test_prefix="${12}" test_enabled="${13}"
  python - "$kind" "$source_config" "$output_config" "$COCO_ROOT" "$IMAGENET_INIT" \
    "$train_ann" "$val_ann" "$train_prefix" "$val_prefix" "$epochs" "$save_best" "$MAX_KEEP_CKPTS" \
    "$validation_enabled" "$test_ann" "$test_prefix" "$test_enabled" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

(kind, source, output, coco_root, init, train_ann, val_ann,
 train_prefix, val_prefix, epochs, save_best, max_keep, validation_enabled,
 test_ann, test_prefix, test_enabled) = sys.argv[1:]
validation_enabled = validation_enabled == '1'
test_enabled = test_enabled == '1'
source, output, coco_root = Path(source), Path(output), Path(coco_root)
cfg = Config.fromfile(source)
categories = json.loads((coco_root / 'train/annotations/instances_train.json').read_text(encoding='utf-8'))['categories']
classes = tuple(item['name'] for item in categories)
if len(classes) != 254:
    raise SystemExit(f'Expected 254 AquaOV255 classes, found {len(classes)}')

def set_dataset(dataset, ann_file, image_prefix):
    if 'dataset' in dataset:
        set_dataset(dataset['dataset'], ann_file, image_prefix)
        return
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            set_dataset(child, ann_file, image_prefix)
        return
    dataset['data_root'] = str(coco_root) + '/'
    dataset['ann_file'] = str(Path(ann_file).resolve())
    dataset['data_prefix'] = dict(img=image_prefix)
    dataset['metainfo'] = dict(classes=classes)

set_dataset(cfg.train_dataloader['dataset'], train_ann, train_prefix)
if validation_enabled:
    set_dataset(cfg.val_dataloader['dataset'], val_ann, val_prefix)
    evaluator = cfg.get('val_evaluator')
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            item['ann_file'] = str(Path(val_ann).resolve())
    elif evaluator is not None:
        evaluator['ann_file'] = str(Path(val_ann).resolve())
else:
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None

if test_enabled:
    set_dataset(cfg.test_dataloader['dataset'], test_ann, test_prefix)
    evaluator = cfg.get('test_evaluator')
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            item['ann_file'] = str(Path(test_ann).resolve())
    elif evaluator is not None:
        evaluator['ann_file'] = str(Path(test_ann).resolve())
else:
    cfg.test_dataloader = None
    cfg.test_evaluator = None

cfg.model.backbone['init_cfg'] = dict(type='Pretrained', checkpoint=init)
cfg.load_from = None
cfg.train_cfg.max_epochs = int(epochs)
cfg.train_cfg.val_interval = 1
if validation_enabled:
    cfg.default_hooks.checkpoint['save_best'] = save_best
else:
    cfg.default_hooks.checkpoint.pop('save_best', None)
cfg.default_hooks.checkpoint['max_keep_ckpts'] = int(max_keep)

if kind == 'mask':
    cfg.model.roi_head.bbox_head['num_classes'] = len(classes)
    cfg.model.roi_head.mask_head['num_classes'] = len(classes)
else:
    heads = cfg.model.roi_head.bbox_head
    if not isinstance(heads, (list, tuple)) or len(heads) != 3:
        raise SystemExit('Expected a three-stage Cascade R-CNN bbox_head list')
    for head in heads:
        head['num_classes'] = len(classes)
    mask_head = cfg.model.roi_head.get('mask_head')
    if mask_head is None:
        raise SystemExit('Cascade Mask R-CNN config has no roi_head.mask_head')
    mask_head['num_classes'] = len(classes)
    schedulers = cfg.param_scheduler
    multi_step = [item for item in schedulers if item.get('type') == 'MultiStepLR']
    if len(multi_step) != 1:
        raise SystemExit(f'Expected one MultiStepLR scheduler, got {len(multi_step)}')
    multi_step[0]['milestones'] = [16, 22]

output.parent.mkdir(parents=True, exist_ok=True)
cfg.dump(output)
print(f'config={output} kind={kind} classes={len(classes)} epochs={epochs} init={init}')
PY
}

run_mask_fold() {
  local train_fold="$1" val_fold="$2" port="$3"
  local name="maskrcnn_train_${train_fold}_val_${val_fold}"
  local work="$WORK_ROOT/$name"
  local runtime="$work/runtime_config.py"
  local train_ann="$AB_ROOT/train_${train_fold}.json"
  local val_ann="$AB_ROOT/train_${val_fold}.json"
  local best marker
  marker="$work/.complete"
  best="$(best_checkpoint "$work" 'best_coco_bbox_mAP_epoch_*.pth' || true)"

  if [[ "$SKIP_COMPLETED" == 1 && -f "$marker" && -n "$best" ]]; then
    echo "REUSE completed $name: $best"
    return
  fi
  if [[ -s "$work/epoch_${MASK_EPOCHS}.pth" && -n "$best" ]]; then
    touch "$marker"
    echo "RECOVER completed $name from final epoch checkpoint: $best"
    return
  fi
  mkdir -p "$work"
  write_runtime_config mask "$MASK_CONFIG" "$runtime" "$train_ann" "$val_ann" \
    "train/images/" "train/images/" "$MASK_EPOCHS" "coco/bbox_mAP" 1 \
    "$val_ann" "train/images/" 1
  echo "START $name train=$train_fold val=$val_fold GPUs=$GPU_GROUP"
  local -a resume_args=()
  [[ -s "$work/latest.pth" ]] && resume_args+=(--resume)
  CUDA_VISIBLE_DEVICES="$GPU_GROUP" PORT="$port" \
    bash tools/dist_train.sh "$runtime" "$(gpu_count "$GPU_GROUP")" --work-dir "$work" "${resume_args[@]}" \
    2>&1 | tee "$LOG_ROOT/$name.log"
  best="$(best_checkpoint "$work" 'best_coco_bbox_mAP_epoch_*.pth' || true)"
  [[ -n "$best" ]] || die "$name did not produce a best bbox AP checkpoint"
  [[ -s "$work/epoch_${MASK_EPOCHS}.pth" ]] || die "$name did not reach epoch $MASK_EPOCHS"
  touch "$marker"
  printf '%s\n' "$best" > "$work/best_checkpoint.txt"
}

run_mask_fold A B "$BASE_PORT"
run_mask_fold B A "$((BASE_PORT + 1))"

filter_fold() {
  local train_fold="$1" filter_fold_name="$2"
  local name="maskrcnn_train_${train_fold}_val_${filter_fold_name}"
  local work="$WORK_ROOT/$name"
  local best="$(best_checkpoint "$work" 'best_coco_bbox_mAP_epoch_*.pth' || true)"
  local out="$OUTPUT_ROOT/easy_${filter_fold_name}"
  local -a reuse_args=()
  [[ -n "$best" ]] || die "missing teacher checkpoint for $name"
  [[ "$SKIP_COMPLETED" == 1 ]] && reuse_args+=(--skip-existing)
  CUDA_VISIBLE_DEVICES="${GPU_GROUP%%,*}" \
    python tools/exp_2/aquaov255_cross_easy.py filter \
      --images "$COCO_ROOT/train/images" \
      --annotation "$AB_ROOT/train_${filter_fold_name}.json" \
      --config "$work/runtime_config.py" \
      --checkpoint "$best" \
      --output "$out" \
      --threshold "$SCORE_THRESHOLD" \
      --device cuda:0 \
      "${reuse_args[@]}" \
      2>&1 | tee "$LOG_ROOT/filter_${filter_fold_name}.log"
}

# Each fold is scored only by the Mask R-CNN trained on the opposite fold.
filter_fold A B
filter_fold B A

EASY_ANN="$EASY_ROOT/annotations/instances_train.json"
if [[ ! -s "$EASY_ANN" ]]; then
  python tools/exp_2/aquaov255_cross_easy.py merge \
    --annotations "$OUTPUT_ROOT/easy_A/instances_easy.json" "$OUTPUT_ROOT/easy_B/instances_easy.json" \
    --output "$EASY_ANN"
  touch "$EASY_ROOT/.complete"
fi

if [[ ! -f "$EASY_SPLIT_ROOT/.complete" ]]; then
  python tools/exp_2/aquaov255_cross_easy.py partition \
    --annotation "$EASY_ANN" \
    --output "$EASY_SPLIT_ROOT" \
    --val-ratio "$EASY_VAL_RATIO" \
    --seed "$EASY_SPLIT_SEED"
fi

run_cascade_variant() {
  local variant="$1" train_ann="$2" val_ann="$3" has_val="$4" port_offset="$5"
  local work="$WORK_ROOT/cascade_mask_rcnn_aquaov255_easy_${variant}_${CASCADE_EPOCHS}e"
  local runtime="$work/runtime_config.py"
  local test_log="$LOG_ROOT/cascade_mask_rcnn_${variant}_test.log"
  local marker="$work/.complete"
  local best
  if [[ "$has_val" == 1 ]]; then
    best="$(best_checkpoint "$work" 'best_coco_segm_mAP_epoch_*.pth' || true)"
  else
    best="$work/epoch_${CASCADE_EPOCHS}.pth"
  fi
  if [[ "$SKIP_COMPLETED" == 1 && -f "$marker" && -s "$best" ]]; then
    echo "REUSE completed Cascade Mask R-CNN variant=$variant checkpoint=$best"
    return
  fi

  mkdir -p "$work"
  local save_best=""
  [[ "$has_val" != 1 ]] || save_best="coco/segm_mAP"
  local test_ann test_prefix test_description
  if [[ "$has_val" == 1 ]]; then
    test_ann="$COCO_ROOT/val/annotations/instances_val.json"
    test_prefix="val/images/"
    test_description="independent original AquaOV255 holdout"
  else
    test_ann="$EASY_ANN"
    test_prefix="train/images/"
    test_description="self-evaluation on the same all_easy training set"
  fi
  write_runtime_config cascade "$CASCADE_CONFIG" "$runtime" "$train_ann" "$val_ann" \
    "train/images/" "train/images/" "$CASCADE_EPOCHS" "$save_best" "$has_val" \
    "$test_ann" "$test_prefix" 1
  if [[ ! -s "$work/epoch_${CASCADE_EPOCHS}.pth" ]]; then
    local -a resume_args=()
    [[ -s "$work/latest.pth" ]] && resume_args+=(--resume)
    CUDA_VISIBLE_DEVICES="$GPU_GROUP" PORT="$((BASE_PORT + port_offset))" \
      bash tools/dist_train.sh "$runtime" "$(gpu_count "$GPU_GROUP")" \
      --work-dir "$work" "${resume_args[@]}" \
      2>&1 | tee "$LOG_ROOT/cascade_mask_rcnn_${variant}_train.log"
  fi
  [[ -s "$work/epoch_${CASCADE_EPOCHS}.pth" ]] || die "Cascade Mask R-CNN $variant did not reach epoch $CASCADE_EPOCHS"
  if [[ "$has_val" == 1 ]]; then
    best="$(best_checkpoint "$work" 'best_coco_segm_mAP_epoch_*.pth' || true)"
    [[ -n "$best" ]] || die "Cascade Mask R-CNN split10 best segm AP checkpoint was not created"
  else
    best="$work/epoch_${CASCADE_EPOCHS}.pth"
    echo "No Easy validation loop for all_easy: all selected Easy images are used for training."
  fi
  if [[ "$RUN_TEST" == 1 ]]; then
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_GROUP" PORT="$((BASE_PORT + port_offset + 100))" \
      bash tools/dist_test.sh "$runtime" "$best" "$(gpu_count "$GPU_GROUP")" \
      2>&1 | tee "$test_log"
    local test_status="${PIPESTATUS[0]}"
    set -e
    if [[ "$test_status" -ne 0 ]] && ! grep -qF 'coco/segm_mAP:' "$test_log"; then
      die "Cascade Mask R-CNN $variant test failed without segm AP (exit=$test_status)"
    fi
  fi

  printf 'field\tvalue\n' > "$work/run_provenance.tsv"
  printf 'variant\t%s\n' "$variant" >> "$work/run_provenance.tsv"
  printf 'train_annotation\t%s\n' "$train_ann" >> "$work/run_provenance.tsv"
  printf 'validation_annotation\t%s\n' "${val_ann:-NONE}" >> "$work/run_provenance.tsv"
  printf 'best_or_final_checkpoint\t%s\n' "$best" >> "$work/run_provenance.tsv"
  printf 'training_validation_metric\t%s\n' "$([[ "$has_val" == 1 ]] && echo coco/segm_mAP || echo NONE)" >> "$work/run_provenance.tsv"
  printf 'test_annotation\t%s\n' "$test_ann" >> "$work/run_provenance.tsv"
  printf 'test_protocol\t%s\n' "$test_description" >> "$work/run_provenance.tsv"
  printf 'test_metric\tcoco/segm_mAP; interpret according to test_protocol\n' >> "$work/run_provenance.tsv"
  touch "$marker"
  echo "COMPLETE Cascade Mask R-CNN variant=$variant checkpoint=$best"
  if [[ "$RUN_TEST" == 1 ]]; then
    grep -E 'coco/(segm|bbox)_mAP(:|_50:|_75:|_s:|_m:|_l:)' "$test_log" | tail -n 6 || true
  fi
}

run_cascade_variant split10 \
  "$EASY_SPLIT_ROOT/instances_train.json" \
  "$EASY_SPLIT_ROOT/instances_val.json" 1 10
run_cascade_variant all_easy "$EASY_ANN" "" 0 20

printf 'field\tvalue\n' > "$OUTPUT_ROOT/run_provenance.tsv"
printf 'aqua_root\t%s\n' "$AQUA_ROOT" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'gpu_group\t%s\n' "$GPU_GROUP" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'ab_rule\tRUOD-style shuffle seed %s; equal halves; Mask R-CNN filters opposite fold\n' "$AB_SEED" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'filter_metric\tper-image custom bbox AP averaged over IoU 0.50:0.95; threshold >= %s\n' "$SCORE_THRESHOLD" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'easy_annotation\t%s\n' "$EASY_ANN" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'cascade_split10\tEasy random split %.3f; evaluate segm AP\n' "$EASY_VAL_RATIO" >> "$OUTPUT_ROOT/run_provenance.tsv"
printf 'cascade_all_easy\tall A_easy+B_easy images used for training and self-evaluation; not a generalization estimate\n' >> "$OUTPUT_ROOT/run_provenance.tsv"
echo "COMPLETE output_root=$OUTPUT_ROOT"
echo "EASY_ANNOTATION=$EASY_ANN"
