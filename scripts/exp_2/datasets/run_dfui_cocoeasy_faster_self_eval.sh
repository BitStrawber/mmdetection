#!/usr/bin/env bash
# Train and test Faster R-CNN on the same sampled DFUI_COCOEasy set.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

SOURCE_RUN="${SOURCE_RUN:-/media/HDD2/XCX/exp_2/coco_easy_runs/coco_full_ab_easy_sample_selfeval_gpu67_restart_20261016_132628}"
SAMPLE_ROOT="${SAMPLE_ROOT:-$SOURCE_RUN/DFUI_COCOEasy}"
TRAIN_ANN="$SAMPLE_ROOT/annotations/instances_train.json"
VAL_ANN="$SAMPLE_ROOT/annotations/instances_val.json"
CONFIG="${CONFIG:-configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/coco_easy_runs/faster_rcnn_dfui_cocoeasy_all_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1}"
EPOCHS="${EPOCHS:-96}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MILESTONES="${MILESTONES:-64,88}"
BASE_PORT="${BASE_PORT:-39920}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -s "$TRAIN_ANN" && -s "$VAL_ANN" && -d "$SAMPLE_ROOT/images" && -s "$CONFIG" ]] ||
  die "missing sampled train/val annotations, images, or Faster R-CNN config"
[[ -x tools/dist_train.sh && -x tools/dist_test.sh ]] || die "missing distributed train/test scripts"
[[ "$GPUS" =~ ^[0-9]+,[0-9]+$ ]] || die "GPUS must contain two GPU indices, e.g. 0,1"
IFS=, read -r GPU_A GPU_B <<< "$GPUS"
[[ "$GPU_A" != "$GPU_B" ]] || die "GPU indices must differ"
[[ "$EPOCHS" -gt 0 && "$BATCH_SIZE" -gt 0 ]] || die "epochs and batch size must be positive"

"$PYTHON_BIN" - "$TRAIN_ANN" "$VAL_ANN" "$SAMPLE_ROOT/images" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from mmengine.config import Config
from mmdet.datasets import CocoDataset

train_ann, val_ann, image_root, config = map(Path, sys.argv[1:])
train = json.loads(train_ann.read_text(encoding='utf-8'))
val = json.loads(val_ann.read_text(encoding='utf-8'))
categories = train['categories']
if categories != val['categories']:
    raise SystemExit('Sampled train/val categories differ')
images = train['images'] + val['images']
annotations = train['annotations'] + val['annotations']
if len(train['images']) != 9770 or len(val['images']) != 1724:
    raise SystemExit('Expected the 9770/1724-image sampled DFUI_COCOEasy split')
ids = {int(image['id']) for image in images}
if len(ids) != len(images) or len(categories) != 80:
    raise SystemExit('Expected disjoint image IDs and 80 COCO categories')
if tuple(category['name'] for category in categories) != CocoDataset.METAINFO['classes']:
    raise SystemExit('Easy category names or order differ from COCO metainfo')
if len({int(ann['id']) for ann in annotations}) != len(annotations):
    raise SystemExit('Duplicate annotation IDs across sampled train/val')
if any(int(ann['image_id']) not in ids for ann in annotations):
    raise SystemExit('Annotation references an image outside sampled DFUI_COCOEasy')
missing = [image['file_name'] for image in images
           if not (image_root / image['file_name']).is_file()]
if missing:
    raise SystemExit(f'Missing sampled images: {missing[:5]}')
cfg = Config.fromfile(str(config))
if cfg.model.type != 'FasterRCNN':
    raise SystemExit(f'Expected FasterRCNN, got {cfg.model.type}')
print(f'train_images={len(train["images"])} val_images={len(val["images"])} '
      f'all_images={len(images)} annotations={len(annotations)} categories={len(categories)}')
print('protocol=train_all_sampled_no_validation_test_same_all_sampled')
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F',' -v a="$GPU_A" -v b="$GPU_B" -v min="$MIN_FREE_MIB" '
      {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2)}
      $1 == a || $1 == b {
        print "GPU=" $1 " free_MiB=" $2
        seen[$1] = 1
        if ($2 + 0 < min) failed = 1
      }
      END {if (!seen[a] || !seen[b] || failed) exit 1}
    ' || die "selected GPUs do not meet MIN_FREE_MIB=$MIN_FREE_MIB"
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "CHECK_ONLY=1 passed; no files written and no training started"
  exit 0
fi

[[ ! -e "$OUTPUT_ROOT" ]] || die "output already exists: $OUTPUT_ROOT"
WORK="$OUTPUT_ROOT/work_dirs/faster_rcnn_dfui_cocoeasy_${EPOCHS}e"
TEST_WORK="$OUTPUT_ROOT/work_dirs/faster_rcnn_dfui_cocoeasy_test"
ALL_ANN="$OUTPUT_ROOT/annotations/instances_all.json"
mkdir -p "$WORK" "$TEST_WORK" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/annotations"

"$PYTHON_BIN" - "$TRAIN_ANN" "$VAL_ANN" "$ALL_ANN" <<'PY'
import json
import sys
from pathlib import Path

train_path, val_path, output_path = map(Path, sys.argv[1:])
train = json.loads(train_path.read_text(encoding='utf-8'))
val = json.loads(val_path.read_text(encoding='utf-8'))
merged = dict(train)
merged['images'] = train['images'] + val['images']
merged['annotations'] = train['annotations'] + val['annotations']
output_path.write_text(json.dumps(merged, ensure_ascii=False), encoding='utf-8')
print(f'combined_annotation={output_path} images={len(merged["images"])}')
PY

"$PYTHON_BIN" - "$CONFIG" "$SAMPLE_ROOT" "$ALL_ANN" "$WORK" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$MILESTONES" <<'PY'
import sys
from pathlib import Path
from mmengine.config import Config
from mmdet.datasets import CocoDataset

config, sample_root, annotation, work, epochs, batch_size, workers, milestones = sys.argv[1:]
cfg = Config.fromfile(config)
classes = CocoDataset.METAINFO['classes']
for key in ('train_dataloader', 'test_dataloader'):
    dataset = cfg[key].dataset
    dataset.data_root = str(Path(sample_root).resolve()) + '/'
    dataset.ann_file = str(Path(annotation).resolve())
    dataset.data_prefix = dict(img='images/')
    dataset.metainfo = dict(classes=classes)
cfg.train_dataloader.batch_size = int(batch_size)
cfg.train_dataloader.num_workers = int(workers)
cfg.val_dataloader = None
cfg.val_evaluator = None
cfg.val_cfg = None
cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = int(workers)
cfg.test_evaluator.ann_file = str(Path(annotation).resolve())
cfg.test_evaluator.metric = 'bbox'
cfg.train_cfg.max_epochs = int(epochs)
schedulers = [item for item in cfg.param_scheduler if item.get('type') == 'MultiStepLR']
if len(schedulers) != 1:
    raise SystemExit('Expected exactly one MultiStepLR scheduler')
schedulers[0].end = int(epochs)
schedulers[0].milestones = [int(item) for item in milestones.split(',')]
cfg.default_hooks.checkpoint.pop('save_best', None)
cfg.default_hooks.checkpoint.interval = int(epochs)
cfg.default_hooks.checkpoint.save_last = True
cfg.default_hooks.checkpoint.max_keep_ckpts = 1
cfg.load_from = None
cfg.resume = False
cfg.work_dir = work
runtime = Path(work) / 'runtime_config.py'
cfg.dump(str(runtime))
print(f'runtime_config={runtime} epochs={epochs} batch_size_per_gpu={batch_size}')
PY

PORT="$BASE_PORT" CUDA_VISIBLE_DEVICES="$GPUS" bash tools/dist_train.sh \
  "$WORK/runtime_config.py" 2 --work-dir "$WORK" \
  2>&1 | tee "$OUTPUT_ROOT/logs/train.log"

FINAL_CKPT="$WORK/epoch_${EPOCHS}.pth"
[[ -s "$FINAL_CKPT" ]] || die "final epoch checkpoint missing: $FINAL_CKPT"
PORT="$((BASE_PORT + 1))" CUDA_VISIBLE_DEVICES="$GPUS" bash tools/dist_test.sh \
  "$WORK/runtime_config.py" "$FINAL_CKPT" 2 --work-dir "$TEST_WORK" \
  2>&1 | tee "$OUTPUT_ROOT/logs/test.log"

{
  printf 'field\tvalue\n'
  printf 'source_run\t%s\n' "$SOURCE_RUN"
  printf 'sample_root\t%s\n' "$SAMPLE_ROOT"
  printf 'sampled_train_annotation\t%s\n' "$TRAIN_ANN"
  printf 'sampled_val_annotation\t%s\n' "$VAL_ANN"
  printf 'combined_annotation\t%s\n' "$ALL_ANN"
  printf 'gpu_group\t%s\n' "$GPUS"
  printf 'epochs\t%s\n' "$EPOCHS"
  printf 'final_checkpoint\t%s\n' "$FINAL_CKPT"
  printf 'validation_annotation\tNONE\n'
  printf 'test_annotation\t%s\n' "$ALL_ANN"
  printf 'test_protocol\ttrain_and_test_on_same_sampled_dfui_cocoeasy_not_independent_generalization\n'
} > "$OUTPUT_ROOT/run_provenance.tsv"
touch "$OUTPUT_ROOT/.complete"
echo "RESULT=PASS_FASTER_RCNN_DFUI_COCOEASY_SELF_EVAL"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
