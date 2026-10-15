#!/usr/bin/env bash
# Cross-filter the full COCO train2017 set, then sample a DFUI-sized COCO-Easy dataset.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

COCO_ROOT="${COCO_ROOT:-/media/HDD2/XCX/exp_2/COCO2017}"
COCO_ANN="${COCO_ANN:-$COCO_ROOT/annotations/instances_train2017.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$COCO_ROOT/train2017}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/coco_easy_runs/coco_full_ab_easy_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py}"
SELF_EVAL_CONFIG="${SELF_EVAL_CONFIG:-configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py}"
GPUS="${GPUS:-6,7}"
NUM_GPUS="${NUM_GPUS:-2}"
BASE_PORT="${BASE_PORT:-39860}"
EPOCHS="${EPOCHS:-24}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
LR="${LR:-0.015}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
LR_MILESTONES="${LR_MILESTONES:-16,22}"
LR_GAMMA="${LR_GAMMA:-0.1}"
SELF_EVAL_EPOCHS="${SELF_EVAL_EPOCHS:-96}"
SELF_EVAL_BATCH_SIZE="${SELF_EVAL_BATCH_SIZE:-4}"
SELF_EVAL_MILESTONES="${SELF_EVAL_MILESTONES:-64,88}"
SELF_EVAL_NUM_WORKERS="${SELF_EVAL_NUM_WORKERS:-2}"
THRESHOLD="${THRESHOLD:-0.60}"
AB_SEED="${AB_SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-20261016}"
SAMPLE_COUNT="${SAMPLE_COUNT:-11494}"
TRAIN_COUNT="${TRAIN_COUNT:-9770}"
VAL_COUNT="${VAL_COUNT:-1724}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -s "$COCO_ANN" && -d "$IMAGE_ROOT" ]] || die "missing COCO train2017 annotation or image directory"
[[ -s tools/exp_2/coco_cross_easy.py ]] || die "missing COCO cross-filter helper"
[[ -s "$CONFIG" ]] || die "missing detector config: $CONFIG"
[[ -s "$SELF_EVAL_CONFIG" ]] || die "missing Faster R-CNN self-eval config: $SELF_EVAL_CONFIG"
[[ "$NUM_GPUS" -eq 2 && "$GPUS" == "6,7" ]] || die "this launch profile is fixed to GPUs 6,7"
[[ "$SAMPLE_COUNT" -eq $((TRAIN_COUNT + VAL_COUNT)) ]] || die "sample/train/val counts do not add up"
[[ "$EPOCHS" -eq 24 ]] || die "RUOD-compatible COCO Easy schedule requires EPOCHS=24"
"$PYTHON_BIN" - "$THRESHOLD" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("THRESHOLD must be numeric")
if not 0.0 <= value <= 1.0:
    raise SystemExit("THRESHOLD must be in [0, 1]")
PY
[[ ! -e "$OUTPUT_ROOT" ]] || die "output already exists: $OUTPUT_ROOT"

echo "===== COCO source counts ====="
"$PYTHON_BIN" - "$COCO_ANN" "$IMAGE_ROOT" <<'PY'
import json, sys
from pathlib import Path
ann = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
images = ann.get('images', [])
missing = sum(not (root / image['file_name']).is_file() for image in images)
print(f"images={len(images)} annotations={len(ann.get('annotations', []))} categories={len(ann.get('categories', []))} missing_images={missing}")
if missing:
    raise SystemExit("COCO image validation failed")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
  awk -F', *' -v min="$MIN_FREE_MIB" '($1==6 || $1==7) {print "GPU=" $1 " free_MiB=" $2; if ($2<min) bad=1; seen[$1]=1} END {if (!seen[6] || !seen[7] || bad) exit 1}' \
    || die "GPU 6 and GPU 7 must each have at least ${MIN_FREE_MIB} MiB free"
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
  "$PYTHON_BIN" - "$CONFIG" "$EPOCHS" "$TRAIN_BATCH_SIZE" "$LR" "$WARMUP_ITERS" "$LR_MILESTONES" "$LR_GAMMA" <<'PY'
import sys
from mmengine.config import Config
from mmdet.datasets import CocoDataset

path, epochs, batch_size, lr, warmup, milestones, gamma = sys.argv[1:]
cfg = Config.fromfile(path)
classes = cfg.train_dataloader.dataset.get("metainfo", {}).get(
    "classes", CocoDataset.METAINFO["classes"])
assert cfg.model.type == "CascadeRCNN", cfg.model.type
assert len(classes) == 80, f"Expected 80 COCO classes, got {len(classes)}"
assert int(epochs) == 24
assert int(batch_size) == 6
milestones = [int(value) for value in milestones.split(",")]
assert milestones == [16, 22], milestones
print(f"base_config={path}")
print(f"model={cfg.model.type} classes={len(classes)}")
print(f"schedule={epochs}e batch_size_per_gpu={batch_size} lr={lr} warmup_iters={warmup} milestones={milestones} gamma={gamma}")
PY
  "$PYTHON_BIN" - "$SELF_EVAL_CONFIG" "$SELF_EVAL_EPOCHS" "$SELF_EVAL_BATCH_SIZE" "$SELF_EVAL_MILESTONES" <<'PY'
import sys
from mmengine.config import Config

path, epochs, batch_size, milestones = sys.argv[1:]
cfg = Config.fromfile(path)
assert cfg.model.type == "FasterRCNN", cfg.model.type
milestones = [int(value) for value in milestones.split(",")]
assert int(epochs) == 96 and milestones == [64, 88], (epochs, milestones)
print(f"self_eval_config={path}")
print(f"self_eval_model={cfg.model.type} epochs={epochs} batch_size_per_gpu={batch_size} milestones={milestones} metric=bbox")
print("self_eval_protocol=train and evaluate on the same sampled COCO-Easy set; not an independent generalization estimate")
PY
  echo "CHECK_ONLY=1 passed; no data was written and no training was started"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/work_dirs"
AB_ROOT="$OUTPUT_ROOT/ab_split"
EASY_ROOT="$OUTPUT_ROOT/easy_full"
SAMPLE_ROOT="$OUTPUT_ROOT/DFUI_COCOEasy"
"$PYTHON_BIN" tools/exp_2/coco_cross_easy.py split --annotation "$COCO_ANN" --output "$AB_ROOT" --seed "$AB_SEED" \
  2>&1 | tee "$OUTPUT_ROOT/logs/split.log"

write_runtime_config() {
  local fold="$1" ann_train="$2" ann_val="$3" work="$4" port="$5"
  "$PYTHON_BIN" - "$CONFIG" "$COCO_ROOT" "$ann_train" "$ann_val" "$work" "$EPOCHS" "$TRAIN_BATCH_SIZE" "$LR" "$WARMUP_ITERS" "$LR_MILESTONES" "$LR_GAMMA" <<'PY'
import sys
from mmengine.config import Config
from mmdet.datasets import CocoDataset
cfg = Config.fromfile(sys.argv[1])
root, train_ann, val_ann, work, epochs, batch_size, lr, warmup, milestones, gamma = sys.argv[2:]
classes = cfg.train_dataloader.dataset.get("metainfo", {}).get(
    "classes", CocoDataset.METAINFO["classes"])
if cfg.model.type != "CascadeRCNN" or len(classes) != 80:
    raise SystemExit(f"Expected an 80-class CascadeRCNN COCO config; got {cfg.model.type} with {len(classes)} classes")
for key, ann in (("train_dataloader", train_ann), ("val_dataloader", val_ann), ("test_dataloader", val_ann)):
    ds = cfg[key].dataset
    ds.data_root = root.rstrip('/') + '/'
    ds.ann_file = ann
    ds.data_prefix = dict(img='train2017/')
    ds.metainfo = dict(classes=classes)
cfg.train_dataloader.batch_size = int(batch_size)
for key in ("val_evaluator", "test_evaluator"):
    cfg[key].ann_file = val_ann
cfg.train_cfg.max_epochs = int(epochs)
cfg.train_cfg.val_interval = 1
cfg.optim_wrapper.optimizer.lr = float(lr)
milestones = [int(value) for value in milestones.split(',')]
cfg.param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=int(warmup)),
    dict(type='MultiStepLR', begin=0, end=int(epochs), by_epoch=True,
         milestones=milestones, gamma=float(gamma)),
]
cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
cfg.default_hooks.checkpoint.max_keep_ckpts = 3
cfg.load_from = None
cfg.work_dir = work
cfg.dump(work + "/runtime_config.py")
print(f"runtime_config={work}/runtime_config.py classes={len(classes)} epochs={epochs} batch_size_per_gpu={batch_size} lr={lr} warmup_iters={warmup} milestones={milestones} gamma={gamma}")
PY
}

train_fold() {
  local train_fold="$1" val_fold="$2" stage="train_${1}_val_${2}"
  local work="$OUTPUT_ROOT/work_dirs/$stage" port
  if [[ "$train_fold" == A ]]; then port="$BASE_PORT"; else port="$((BASE_PORT + 1))"; fi
  mkdir -p "$work"
  write_runtime_config "$train_fold" "$AB_ROOT/train_${train_fold}.json" "$AB_ROOT/train_${val_fold}.json" "$work" "$port"
  PORT="$port" CUDA_VISIBLE_DEVICES="$GPUS" bash tools/dist_train.sh "$work/runtime_config.py" "$NUM_GPUS" \
    --work-dir "$work" 2>&1 | tee "$OUTPUT_ROOT/logs/$stage.log"
  local ckpt
  ckpt="$(find "$work" -maxdepth 1 -type f -name 'best_coco_bbox_mAP*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
  [[ -s "$ckpt" ]] || die "no best bbox mAP checkpoint produced for $stage"
  printf '%s\n' "$ckpt" > "$work/best_checkpoint.txt"
}

filter_fold() {
  local train_fold="$1" target_fold="$2" work="$OUTPUT_ROOT/work_dirs/train_${train_fold}_val_${target_fold}"
  local ckpt
  ckpt="$(cat "$work/best_checkpoint.txt")"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON_BIN" tools/exp_2/coco_cross_easy.py filter \
    --annotation "$AB_ROOT/train_${target_fold}.json" --image-root "$IMAGE_ROOT" \
    --config "$work/runtime_config.py" --checkpoint "$ckpt" --output "$EASY_ROOT/easy_${target_fold}" \
    --threshold "$THRESHOLD" --device cuda:0 2>&1 | tee "$OUTPUT_ROOT/logs/filter_${target_fold}.log"
}

train_fold A B
filter_fold A B
train_fold B A
filter_fold B A

"$PYTHON_BIN" - "$EASY_ROOT/easy_A/instances_easy.json" "$EASY_ROOT/easy_B/instances_easy.json" "$EASY_ROOT/instances_easy_merged.json" <<'PY'
import json, sys
from pathlib import Path
a, b = [json.loads(Path(p).read_text()) for p in sys.argv[1:3]]
ids_a = {int(x['id']) for x in a['images']}
ids_b = {int(x['id']) for x in b['images']}
if ids_a & ids_b:
    raise SystemExit('A/B easy outputs overlap')
merged = dict(a)
merged['images'] = a['images'] + b['images']
merged['annotations'] = a['annotations'] + b['annotations']
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps(merged), encoding='utf-8')
print(f"easy_images={len(merged['images'])} easy_annotations={len(merged['annotations'])}")
PY

"$PYTHON_BIN" tools/exp_2/coco_cross_easy.py sample --easy-annotation "$EASY_ROOT/instances_easy_merged.json" \
  --image-root "$IMAGE_ROOT" --output "$SAMPLE_ROOT" --count "$SAMPLE_COUNT" \
  --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --seed "$SAMPLE_SEED" \
  2>&1 | tee "$OUTPUT_ROOT/logs/sample.log"

SELF_EVAL_ANN="$SAMPLE_ROOT/annotations/instances_all.json"
SELF_EVAL_WORK="$OUTPUT_ROOT/work_dirs/faster_rcnn_cocoeasy_self_eval"
SELF_EVAL_TEST_WORK="$OUTPUT_ROOT/work_dirs/faster_rcnn_cocoeasy_self_eval_test"
"$PYTHON_BIN" - "$SAMPLE_ROOT" "$SELF_EVAL_ANN" "$SAMPLE_COUNT" <<'PY'
import json, sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
parts = [json.loads((root / "annotations" / f"instances_{split}.json").read_text()) for split in ("train", "val")]
categories = parts[0].get("categories", [])
if any(part.get("categories", []) != categories for part in parts[1:]):
    raise SystemExit("COCO Easy train/val category definitions differ")
images, annotations, image_ids, annotation_ids = [], [], set(), set()
for part in parts:
    for image in part.get("images", []):
        image_id = int(image["id"])
        if image_id in image_ids:
            raise SystemExit(f"Duplicate image id in sampled self-eval set: {image_id}")
        image_ids.add(image_id)
        images.append(image)
    for ann in part.get("annotations", []):
        ann_id = int(ann["id"])
        if ann_id in annotation_ids:
            raise SystemExit(f"Duplicate annotation id in sampled self-eval set: {ann_id}")
        annotation_ids.add(ann_id)
        annotations.append(ann)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}), encoding="utf-8")
print(f"self_eval_images={len(images)} self_eval_annotations={len(annotations)} output={output}")
if len(images) != int(sys.argv[3]):
    raise SystemExit("Unexpected self-eval image count")
PY

mkdir -p "$SELF_EVAL_WORK" "$SELF_EVAL_TEST_WORK"
"$PYTHON_BIN" - "$SELF_EVAL_CONFIG" "$SAMPLE_ROOT" "$SELF_EVAL_ANN" "$SELF_EVAL_WORK" "$SELF_EVAL_EPOCHS" "$SELF_EVAL_BATCH_SIZE" "$SELF_EVAL_NUM_WORKERS" "$SELF_EVAL_MILESTONES" <<'PY'
import sys
from mmengine.config import Config
from mmdet.datasets import CocoDataset

config, root, annotation, work, epochs, batch_size, workers, milestones = sys.argv[1:]
cfg = Config.fromfile(config)
if cfg.model.type != "FasterRCNN":
    raise SystemExit(f"Expected FasterRCNN config; got {cfg.model.type}")
classes = CocoDataset.METAINFO["classes"]
if len(classes) != 80:
    raise SystemExit(f"Expected 80 built-in COCO classes; got {len(classes)}")
dataset = dict(type="CocoDataset", data_root=root.rstrip("/") + "/",
               ann_file=annotation, data_prefix=dict(img="images/"),
               metainfo=dict(classes=classes))
for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
    cfg[key].dataset = dict(dataset)
    cfg[key].num_workers = int(workers)
cfg.train_dataloader.batch_size = int(batch_size)
cfg.val_dataloader.batch_size = 1
cfg.train_cfg.max_epochs = int(epochs)
cfg.train_cfg.val_interval = 1
milestones = [int(value) for value in milestones.split(",")]
cfg.param_scheduler[1].end = int(epochs)
cfg.param_scheduler[1].milestones = milestones
for key in ("val_evaluator", "test_evaluator"):
    cfg[key].ann_file = annotation
    cfg[key].metric = "bbox"
cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
cfg.default_hooks.checkpoint.max_keep_ckpts = 5
cfg.load_from = None
cfg.work_dir = work
cfg.dump(work + "/runtime_config.py")
print(f"self_eval_runtime_config={work}/runtime_config.py classes={len(classes)} epochs={epochs} batch_size_per_gpu={batch_size} milestones={milestones}")
PY
SELF_TRAIN_PORT="$((BASE_PORT + 2))"
SELF_TEST_PORT="$((BASE_PORT + 3))"
PORT="$SELF_TRAIN_PORT" CUDA_VISIBLE_DEVICES="$GPUS" bash tools/dist_train.sh "$SELF_EVAL_WORK/runtime_config.py" "$NUM_GPUS" \
  --work-dir "$SELF_EVAL_WORK" 2>&1 | tee "$OUTPUT_ROOT/logs/faster_rcnn_self_eval_train.log"
SELF_CKPT="$(find "$SELF_EVAL_WORK" -maxdepth 1 -type f -name 'best_coco_bbox_mAP*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -s "$SELF_CKPT" ]] || die "Faster R-CNN self-eval did not produce a best bbox mAP checkpoint"
printf '%s\n' "$SELF_CKPT" > "$SELF_EVAL_WORK/best_checkpoint.txt"
PORT="$SELF_TEST_PORT" CUDA_VISIBLE_DEVICES="$GPUS" bash tools/dist_test.sh "$SELF_EVAL_WORK/runtime_config.py" "$SELF_CKPT" "$NUM_GPUS" \
  --work-dir "$SELF_EVAL_TEST_WORK" 2>&1 | tee "$OUTPUT_ROOT/logs/faster_rcnn_self_eval_test.log"

printf 'threshold\t%s\nab_seed\t%s\nsample_seed\t%s\neasy_pool\t%s\nsample_root\t%s\nself_eval_checkpoint\t%s\nself_eval_annotation\t%s\nself_eval_protocol\ttrain_and_test_on_same_sampled_easy_images_not_independent_test\n' \
  "$THRESHOLD" "$AB_SEED" "$SAMPLE_SEED" "$EASY_ROOT/instances_easy_merged.json" "$SAMPLE_ROOT" "$SELF_CKPT" "$SELF_EVAL_ANN" \
  > "$OUTPUT_ROOT/run_provenance.tsv"
touch "$OUTPUT_ROOT/.complete"
echo "RESULT=PASS_COCO_FULL_AB_EASY_DFUI_SAMPLE_AND_FASTER_SELF_EVAL"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
