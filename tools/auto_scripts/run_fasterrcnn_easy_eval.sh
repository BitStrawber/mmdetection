#!/usr/bin/env bash
# Reusable Faster R-CNN legacy self-evaluation runner for an Easy COCO dataset.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

usage() {
  cat <<'EOF'
Usage:
  run_fasterrcnn_easy_eval.sh --name NAME --easy-root PATH --gpus 0,1 [options]
  run_fasterrcnn_easy_eval.sh --name NAME --image-root DIR \
    --train-ann TRAIN.json --val-ann VAL.json --gpus 0,1

Options:
  --output-root PATH       Default: work_dirs/auto_faster_eval/NAME
  --config PATH            Faster R-CNN base config
  --epochs N               Default: 96
  --batch-size N           Per-GPU batch size (default: 4)
  --num-workers N          Dataloader workers (default: 2)
  --lr FLOAT               Override optimizer learning rate
  --milestones A,B         Scheduler milestones (default: 64,88)
  --base-port N            Default: 36000
  --start-stage STAGE      all|prepare|train|test
  --force-stage STAGE      all|prepare|train|test
  --no-resume              Do not pass --resume to unfinished training
  --allow-input-change     Reuse output root with different inputs
  --min-free-mib N         Minimum free GPU memory (default: 8192)
  --check-only             Validate and print the plan without writing
  -h, --help               Show this help

Default --easy-root layout:
  annotations/instances_train.json
  annotations/instances_val.json
  images/

Train and val annotations are merged into instances_all.json. All images are
used for training and the same set is used for final testing. The final epoch
checkpoint is used. This is self-evaluation, not an independent test protocol.
EOF
}

require_file() { [ -s "$1" ] || die "missing or empty file: $1"; }
require_dir() { [ -d "$1" ] || die "missing directory: $1"; }

pick_repo_file() {
  local relative="$1"
  local path
  for path in "$REPO_ROOT/$relative" "$REPO_ROOT/.publish_repo/$relative"; do
    [ -s "$path" ] && { printf '%s\n' "$path"; return 0; }
  done
  return 1
}

gpu_preflight() {
  local min_free="$1"
  local gpu free wanted seen=0 failed=0
  local -a requested
  IFS=',' read -r -a requested <<< "$GPUS"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  while IFS=',' read -r gpu free; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    for wanted in "${requested[@]}"; do
      if [ "$gpu" = "$wanted" ]; then
        echo "GPU=$gpu free_MiB=$free required_MiB=$min_free"
        seen=$((seen + 1))
        [ "$free" -ge "$min_free" ] || failed=1
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [ "$seen" -eq "${#requested[@]}" ] || failed=1
  [ "$failed" -eq 0 ] || die "selected GPUs do not meet MIN_FREE_MIB=$min_free"
}

marker() { printf '%s\n' "$STATE_DIR/$1.complete"; }

stage_done() {
  local stage="$1"
  local path
  path="$(marker "$stage")"
  [ -f "$path" ] && [ "$FORCE_STAGE" != "all" ] && [ "$FORCE_STAGE" != "$stage" ]
}

forced_stage() {
  [ "$FORCE_STAGE" = "all" ] || [ "$FORCE_STAGE" = "$1" ]
}

retire_directory() {
  local path="$1"
  local parent name
  [ -e "$path" ] || return 0
  parent="$(dirname "$path")"
  name="$(basename "$path")"
  mv -- "$path" "$parent/${name}.retired.$(date +%Y%m%d_%H%M%S)"
}

complete_stage() {
  local stage="$1"
  mkdir -p "$STATE_DIR"
  date '+%F %T' > "$(marker "$stage")"
  log "COMPLETE stage=$stage"
}

write_runtime_config() {
  "$PYTHON_BIN" - \
    "$CONFIG_PATH" "$IMAGE_ROOT" "$ALL_ANN" "$WORK" "$RUNTIME" \
    "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$MILESTONES" "$LR" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

(
    source, image_root, all_ann, work, runtime,
    epochs_value, batch_value, workers_value, milestones_value, lr_value,
) = sys.argv[1:]

source = Path(source)
image_root = Path(image_root).resolve()
all_ann = Path(all_ann).resolve()
work = Path(work).resolve()
runtime = Path(runtime).resolve()
epochs = int(epochs_value)

cfg = Config.fromfile(str(source))
if cfg.model.type != "FasterRCNN":
    raise SystemExit(f"expected FasterRCNN, got {cfg.model.type}")

data = json.loads(all_ann.read_text(encoding="utf-8"))
categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
classes = tuple(item["name"] for item in categories)
missing = [
    image["file_name"]
    for image in data.get("images", [])
    if not (image_root / image["file_name"]).is_file()
]
if not data.get("images"):
    raise SystemExit("combined annotation contains no images")
if missing:
    raise SystemExit(f"missing {len(missing)} images; examples={missing[:5]}")

def configure(dataset, ann_file):
    if "dataset" in dataset:
        configure(dataset["dataset"], ann_file)
        return
    if "datasets" in dataset:
        for child in dataset["datasets"]:
            configure(child, ann_file)
        return
    dataset["data_root"] = str(image_root).rstrip("/") + "/"
    dataset["ann_file"] = str(ann_file)
    dataset["data_prefix"] = dict(img="")
    dataset["metainfo"] = dict(classes=classes)

configure(cfg.train_dataloader.dataset, all_ann)
configure(cfg.test_dataloader.dataset, all_ann)

cfg.train_dataloader.batch_size = int(batch_value)
cfg.train_dataloader.num_workers = int(workers_value)
cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = int(workers_value)
cfg.train_cfg.max_epochs = epochs
cfg.val_dataloader = None
cfg.val_evaluator = None
cfg.val_cfg = None

if "roi_head" not in cfg.model or "bbox_head" not in cfg.model.roi_head:
    raise SystemExit("Faster R-CNN config has no roi_head.bbox_head")
cfg.model.roi_head.bbox_head["num_classes"] = len(classes)

cfg.val_evaluator.ann_file = str(val_ann)
cfg.test_evaluator.ann_file = str(test_ann)
cfg.val_evaluator.metric = "bbox"
cfg.test_evaluator.metric = "bbox"

if lr_value:
    cfg.optim_wrapper.optimizer["lr"] = float(lr_value)
milestones = [int(item) for item in milestones_value.split(",") if item]
multi_step = [
    scheduler
    for scheduler in cfg.param_scheduler
    if scheduler.get("type") == "MultiStepLR"
]
if len(multi_step) != 1:
    raise SystemExit(f"expected one MultiStepLR, got {len(multi_step)}")
multi_step[0]["end"] = epochs
multi_step[0]["milestones"] = milestones

cfg.default_hooks.checkpoint.pop("save_best", None)
cfg.default_hooks.checkpoint.interval = epochs
cfg.default_hooks.checkpoint.save_last = True
cfg.default_hooks.checkpoint.max_keep_ckpts = 1
cfg.load_from = None
cfg.resume = False
cfg.work_dir = str(work)
work.mkdir(parents=True, exist_ok=True)
cfg.dump(str(runtime))
print(
    f"runtime={runtime} classes={len(classes)} "
    f"all_train={len(data['images'])} all_test={len(data['images'])} "
    f"epochs={epochs} protocol=train_all_test_same_all"
)
PY
}

stage_prepare() {
  mkdir -p "$OUTPUT_ROOT/annotations" "$WORK"
  "$PYTHON_BIN" - "$TRAIN_ANN" "$VAL_ANN" "$ALL_ANN" <<'PY'
import json
import sys
from pathlib import Path

train_path, val_path, output_path = map(Path, sys.argv[1:])
train = json.loads(train_path.read_text(encoding="utf-8"))
val = json.loads(val_path.read_text(encoding="utf-8"))

train_categories = sorted(train.get("categories", []), key=lambda item: int(item["id"]))
val_categories = sorted(val.get("categories", []), key=lambda item: int(item["id"]))
if [(x["id"], x["name"]) for x in train_categories] != [
    (x["id"], x["name"]) for x in val_categories
]:
    raise SystemExit("train/val category definitions differ")

images = list(train.get("images", [])) + list(val.get("images", []))
annotations = list(train.get("annotations", [])) + list(val.get("annotations", []))
image_ids = {int(item["id"]) for item in images}
annotation_ids = {int(item["id"]) for item in annotations}
if len(image_ids) != len(images):
    raise SystemExit("duplicate image IDs across train/val")
if len(annotation_ids) != len(annotations):
    raise SystemExit("duplicate annotation IDs across train/val")

merged = dict(train)
merged["images"] = images
merged["annotations"] = annotations
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
print(f"combined_annotation={output_path} images={len(images)} annotations={len(annotations)}")
PY
  write_runtime_config
}

stage_train() {
  local log_file="$OUTPUT_ROOT/logs/train.log"
  local final_checkpoint
  local -a resume_args=()
  if forced_stage train; then
    retire_directory "$WORK"
  fi
  mkdir -p "$WORK" "$OUTPUT_ROOT/logs"
  stage_prepare
  final_checkpoint="$WORK/epoch_${EPOCHS}.pth"
  if ! forced_stage train && [ "$RESUME" = "1" ] && \
     [ -s "$final_checkpoint" ]; then
    log "RECOVER train stage final=$final_checkpoint"
    return 0
  fi
  if ! forced_stage train && [ "$RESUME" = "1" ] && [ -s "$WORK/last_checkpoint" ]; then
    resume_args+=(--resume)
  fi
  log "START Faster R-CNN train gpus=$GPUS port=$BASE_PORT"
  PORT="$BASE_PORT" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TRAIN" \
    "$RUNTIME" "$NUM_GPUS" --work-dir "$WORK" "${resume_args[@]}" \
    2>&1 | tee "$log_file"
  [ -s "$final_checkpoint" ] || die "final checkpoint missing: $final_checkpoint"
  log "TRAINED Faster R-CNN final=$final_checkpoint"
}

stage_test() {
  local log_file="$OUTPUT_ROOT/logs/test.log"
  local test_work="$OUTPUT_ROOT/test"
  local final_checkpoint
  [ -s "$RUNTIME" ] && [ -s "$ALL_ANN" ] || stage_prepare
  final_checkpoint="$WORK/epoch_${EPOCHS}.pth"
  [ -s "$final_checkpoint" ] || die "final checkpoint missing: $final_checkpoint"
  mkdir -p "$test_work" "$OUTPUT_ROOT/logs"
  if ! forced_stage test && [ "$RESUME" = "1" ] && \
     grep -q 'coco/bbox_mAP:' "$log_file" 2>/dev/null; then
    log "RECOVER test metrics from $log_file"
    return 0
  fi
  log "TEST Faster R-CNN checkpoint=$final_checkpoint"
  PORT="$((BASE_PORT + 1))" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TEST" \
    "$RUNTIME" "$final_checkpoint" "$NUM_GPUS" --work-dir "$test_work" \
    2>&1 | tee "$log_file"
  grep -q 'coco/bbox_mAP:' "$log_file" || die "test did not report coco/bbox_mAP"
}

NAME=""
EASY_ROOT=""
TRAIN_ANN=""
VAL_ANN=""
IMAGE_ROOT=""
OUTPUT_ROOT=""
CONFIG="configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
GPUS="0,1"
EPOCHS=96
BATCH_SIZE=4
NUM_WORKERS=2
LR=""
MILESTONES="64,88"
BASE_PORT=36000
START_STAGE="all"
FORCE_STAGE=""
RESUME=1
ALLOW_INPUT_CHANGE=0
MIN_FREE_MIB=8192
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) NAME="${2:-}"; shift 2 ;;
    --easy-root) EASY_ROOT="${2:-}"; shift 2 ;;
    --train-ann) TRAIN_ANN="${2:-}"; shift 2 ;;
    --val-ann) VAL_ANN="${2:-}"; shift 2 ;;
    --image-root) IMAGE_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:-}"; shift 2 ;;
    --lr) LR="${2:-}"; shift 2 ;;
    --milestones) MILESTONES="${2:-}"; shift 2 ;;
    --base-port) BASE_PORT="${2:-}"; shift 2 ;;
    --start-stage) START_STAGE="${2:-}"; shift 2 ;;
    --force-stage) FORCE_STAGE="${2:-}"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --allow-input-change) ALLOW_INPUT_CHANGE=1; shift ;;
    --min-free-mib) MIN_FREE_MIB="${2:-}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$NAME" ] || die "--name is required"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid name"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --gpus"
[ "$(tr ',' '\n' <<< "$GPUS" | sort -u | wc -l)" -eq "$(awk -F, '{print NF}' <<< "$GPUS")" ] ||
  die "GPU indices must be unique"
case "$START_STAGE" in all|prepare|train|test) ;; *) die "invalid --start-stage" ;; esac
case "$FORCE_STAGE" in ""|all|prepare|train|test) ;; *) die "invalid --force-stage" ;; esac

if [ -z "$EASY_ROOT" ] && [ -z "$TRAIN_ANN" ] && [ -z "$VAL_ANN" ] && [ -z "$IMAGE_ROOT" ]; then
  die "--easy-root is required unless explicit annotation and image paths are provided"
fi
if [ -n "$EASY_ROOT" ]; then
  [ -n "$TRAIN_ANN" ] || TRAIN_ANN="$EASY_ROOT/annotations/instances_train.json"
  [ -n "$VAL_ANN" ] || VAL_ANN="$EASY_ROOT/annotations/instances_val.json"
  [ -n "$IMAGE_ROOT" ] || IMAGE_ROOT="$EASY_ROOT/images"
fi
[ -n "$TRAIN_ANN" ] || TRAIN_ANN="$EASY_ROOT/annotations/instances_train.json"
[ -n "$VAL_ANN" ] || VAL_ANN="$EASY_ROOT/annotations/instances_val.json"
[ -n "$IMAGE_ROOT" ] || IMAGE_ROOT="$EASY_ROOT/images"

require_file "$TRAIN_ANN"
require_file "$VAL_ANN"
require_dir "$IMAGE_ROOT"

CONFIG_PATH="$(pick_repo_file "$CONFIG" || true)"
DIST_TRAIN="$(pick_repo_file tools/dist_train.sh || true)"
DIST_TEST="$(pick_repo_file tools/dist_test.sh || true)"
[ -n "$CONFIG_PATH" ] || die "Faster R-CNN config not found: $CONFIG"
[ -n "$DIST_TRAIN" ] || die "tools/dist_train.sh not found"
[ -n "$DIST_TEST" ] || die "tools/dist_test.sh not found"

[ -n "$OUTPUT_ROOT" ] || OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_faster_eval/$NAME"
STATE_DIR="$OUTPUT_ROOT/.state"
WORK="$OUTPUT_ROOT/work_dirs/faster_rcnn_${EPOCHS}e"
RUNTIME="$WORK/runtime_config.py"
ALL_ANN="$OUTPUT_ROOT/annotations/instances_all.json"
NUM_GPUS="$(awk -F, '{print NF}' <<< "$GPUS")"
INPUT_FINGERPRINT="$(
  printf '%s\n' "$NAME|$TRAIN_ANN|$VAL_ANN|$IMAGE_ROOT|$CONFIG|$EPOCHS|$BATCH_SIZE|$NUM_WORKERS|$MILESTONES|$LR" |
    sha256sum | cut -d' ' -f1
)"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK_ONLY Faster R-CNN Easy evaluation"
  echo "name=$NAME gpus=$GPUS"
  echo "train_ann=$TRAIN_ANN"
  echo "val_ann=$VAL_ANN"
  echo "image_root=$IMAGE_ROOT"
  echo "output=$OUTPUT_ROOT"
  echo "protocol=train_all_test_same_all"
  echo "stages=prepare,train,test"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/work_dirs" "$STATE_DIR"
if [ -s "$STATE_DIR/input_fingerprint" ]; then
  existing="$(cat "$STATE_DIR/input_fingerprint")"
  if [ "$existing" != "$INPUT_FINGERPRINT" ] && [ "$ALLOW_INPUT_CHANGE" != "1" ]; then
    die "core inputs changed for existing output root; use a new directory or --allow-input-change"
  fi
else
  printf '%s\n' "$INPUT_FINGERPRINT" > "$STATE_DIR/input_fingerprint"
fi
if [ ! -s "$OUTPUT_ROOT/status.tsv" ]; then
  printf 'stage\tstatus\tstarted\n' > "$OUTPUT_ROOT/status.tsv"
fi

gpu_preflight "$MIN_FREE_MIB"

stage_rank() {
  case "$1" in
    all) echo 0 ;;
    prepare) echo 1 ;;
    train) echo 2 ;;
    test) echo 3 ;;
  esac
}

run_from() {
  local stage="$1"
  [ "$(stage_rank "$START_STAGE")" -le "$(stage_rank "$stage")" ] || return 0
  if stage_done "$stage"; then
    log "SKIP completed stage=$stage"
    return 0
  fi
  printf '%s\tRUNNING\t%s\n' "$stage" "$(date '+%F %T')" >> "$OUTPUT_ROOT/status.tsv"
  "stage_${stage}" || {
    printf '%s\tFAILED\t%s\n' "$stage" "$(date '+%F %T')" >> "$OUTPUT_ROOT/status.tsv"
    die "stage failed: $stage. Rerun the same command to resume from this stage."
  }
  complete_stage "$stage"
  printf '%s\tCOMPLETE\t%s\n' "$stage" "$(date '+%F %T')" >> "$OUTPUT_ROOT/status.tsv"
}

log "Faster R-CNN Easy evaluation name=$NAME gpus=$GPUS output=$OUTPUT_ROOT"
run_from prepare
run_from train
run_from test

FINAL_CKPT="$WORK/epoch_${EPOCHS}.pth"
{
  printf 'field\tvalue\n'
  printf 'name\t%s\n' "$NAME"
  printf 'train_ann\t%s\n' "$TRAIN_ANN"
  printf 'val_ann\t%s\n' "$VAL_ANN"
  printf 'combined_annotation\t%s\n' "$ALL_ANN"
  printf 'image_root\t%s\n' "$IMAGE_ROOT"
  printf 'gpus\t%s\n' "$GPUS"
  printf 'final_checkpoint\t%s\n' "$FINAL_CKPT"
  printf 'validation_annotation\tNONE\n'
  printf 'test_annotation\t%s\n' "$ALL_ANN"
  printf 'protocol\ttrain_all_test_same_all\n'
} > "$OUTPUT_ROOT/run_provenance.tsv"

echo "RESULT=PASS_FASTERRCNN_EASY_EVAL"
echo "FINAL_CHECKPOINT=$FINAL_CKPT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
