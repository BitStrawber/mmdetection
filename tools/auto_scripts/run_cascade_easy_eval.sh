#!/usr/bin/env bash
# Reusable Cascade Mask R-CNN train/validation/test runner for an Easy dataset.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

usage() {
  cat <<'EOF'
Usage:
  run_cascade_easy_eval.sh --name NAME --easy-root PATH --gpus 0,1 [options]
  run_cascade_easy_eval.sh --name NAME --image-root DIR \
    --train-ann TRAIN.json --val-ann VAL.json [--test-ann TEST.json] --gpus 0,1

Options:
  --output-root PATH       Default: work_dirs/auto_cascade_eval/NAME
  --config PATH            Cascade Mask R-CNN base config
  --epochs N               Default: 24
  --batch-size N           Per-GPU batch size (default: 2)
  --num-workers N          Dataloader workers (default: 2)
  --lr FLOAT               Override optimizer learning rate
  --milestones A,B         Scheduler milestones (default: 16,22)
  --metric bbox|segm       Best-checkpoint and test metric (default: segm)
  --image-root DIR         Shared image root for all splits
  --train-image-root DIR   Optional train image root
  --val-image-root DIR     Optional validation image root
  --test-image-root DIR    Optional independent test image root
  --all-easy-ann PATH      Use one annotation for train, validation, and test
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
  images/train/
  images/val/

With --all-easy-ann, every Easy image is used for training, checkpoint
validation, and final testing. This is the intended self-evaluation protocol.
Without it, the best checkpoint is selected by validation coco/segm_mAP or
coco/bbox_mAP and the selected checkpoint is tested on --test-ann.
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

best_checkpoint() {
  local work="$1"
  find "$work" -maxdepth 1 -type f -name "best_coco_${METRIC}_mAP*.pth" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

marker() { printf '%s\n' "$STATE_DIR/$1.complete"; }

stage_done() {
  local stage="$1"
  local path
  path="$(marker "$stage")"
  [ -f "$path" ] && [ "$FORCE_STAGE" != "all" ] && [ "$FORCE_STAGE" != "$stage" ]
}

complete_stage() {
  local stage="$1"
  mkdir -p "$STATE_DIR"
  date '+%F %T' > "$(marker "$stage")"
  log "COMPLETE stage=$stage"
}

write_runtime_config() {
  "$PYTHON_BIN" - \
    "$CONFIG_PATH" "$TRAIN_IMAGE_ROOT" "$VAL_IMAGE_ROOT" "$TEST_IMAGE_ROOT" \
    "$TRAIN_ANN" "$VAL_ANN" "$TEST_ANN" \
    "$WORK" "$RUNTIME" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" \
    "$MILESTONES" "$LR" "$METRIC" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

(
    source, train_image_root, val_image_root, test_image_root,
    train_ann, val_ann, test_ann, work, runtime,
    epochs_value, batch_value, workers_value, milestones_value, lr_value,
    metric,
) = sys.argv[1:]

source = Path(source)
train_image_root = Path(train_image_root).resolve()
val_image_root = Path(val_image_root).resolve()
test_image_root = Path(test_image_root).resolve()
train_ann = Path(train_ann).resolve()
val_ann = Path(val_ann).resolve()
test_ann = Path(test_ann).resolve()
work = Path(work).resolve()
runtime = Path(runtime).resolve()
epochs = int(epochs_value)

cfg = Config.fromfile(str(source))
if cfg.model.type != "CascadeRCNN":
    raise SystemExit(f"expected CascadeRCNN, got {cfg.model.type}")
if metric not in {"bbox", "segm"}:
    raise SystemExit(f"unsupported metric: {metric}")

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

train = load(train_ann)
val = load(val_ann)
test = load(test_ann)
categories = sorted(train.get("categories", []), key=lambda item: int(item["id"]))
for name, data in (("val", val), ("test", test)):
    current = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
    if [(x["id"], x["name"]) for x in current] != [
        (x["id"], x["name"]) for x in categories
    ]:
        raise SystemExit(f"{name} category definitions differ from train")
classes = tuple(item["name"] for item in categories)

for split, ann_path, image_root, data in (
    ("train", train_ann, train_image_root, train),
    ("val", val_ann, val_image_root, val),
    ("test", test_ann, test_image_root, test),
):
    missing = [
        image["file_name"]
        for image in data.get("images", [])
        if not (image_root / image["file_name"]).is_file()
    ]
    if not data.get("images"):
        raise SystemExit(f"{split}: annotation contains no images")
    if missing:
        raise SystemExit(f"{split}: missing {len(missing)} images; examples={missing[:5]}")

def configure(dataset, ann_file, image_root):
    if "dataset" in dataset:
        configure(dataset["dataset"], ann_file, image_root)
        return
    if "datasets" in dataset:
        for child in dataset["datasets"]:
            configure(child, ann_file, image_root)
        return
    dataset["data_root"] = str(image_root).rstrip("/") + "/"
    dataset["ann_file"] = str(ann_file)
    dataset["data_prefix"] = dict(img="")
    dataset["metainfo"] = dict(classes=classes)

configure(cfg.train_dataloader.dataset, train_ann, train_image_root)
configure(cfg.val_dataloader.dataset, val_ann, val_image_root)
configure(cfg.test_dataloader.dataset, test_ann, test_image_root)

cfg.train_dataloader.batch_size = int(batch_value)
cfg.train_dataloader.num_workers = int(workers_value)
cfg.val_dataloader.batch_size = 1
cfg.val_dataloader.num_workers = int(workers_value)
cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = int(workers_value)
cfg.train_cfg.max_epochs = epochs
cfg.train_cfg.val_interval = 1

if "roi_head" not in cfg.model or "bbox_head" not in cfg.model.roi_head:
    raise SystemExit("Cascade config has no roi_head.bbox_head")
heads = cfg.model.roi_head.bbox_head
if not isinstance(heads, (list, tuple)) or len(heads) != 3:
    raise SystemExit("expected three Cascade R-CNN bbox heads")
for head in heads:
    head["num_classes"] = len(classes)
if "mask_head" not in cfg.model.roi_head:
    raise SystemExit("Cascade config has no roi_head.mask_head")
cfg.model.roi_head.mask_head["num_classes"] = len(classes)

cfg.val_evaluator.ann_file = str(val_ann)
cfg.test_evaluator.ann_file = str(test_ann)
cfg.val_evaluator.metric = metric
cfg.test_evaluator.metric = metric

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

cfg.default_hooks.checkpoint.save_best = f"coco/{metric}_mAP"
cfg.default_hooks.checkpoint.max_keep_ckpts = 5
cfg.load_from = None
cfg.resume = False
cfg.work_dir = str(work)
work.mkdir(parents=True, exist_ok=True)
cfg.dump(str(runtime))
print(
    f"runtime={runtime} classes={len(classes)} "
    f"train={len(train['images'])} val={len(val['images'])} "
    f"test={len(test['images'])} epochs={epochs}"
)
PY
}

stage_prepare() {
  write_runtime_config
}

stage_train() {
  local log_file="$OUTPUT_ROOT/logs/train.log"
  local best
  local -a resume_args=()
  mkdir -p "$WORK" "$OUTPUT_ROOT/logs"
  stage_prepare
  best="$(best_checkpoint "$WORK" || true)"
  if [ "$RESUME" = "1" ] && [ -s "$WORK/epoch_${EPOCHS}.pth" ] && [ -n "$best" ]; then
    log "RECOVER train stage best=$best"
    return 0
  fi
  if [ "$RESUME" = "1" ] && [ -s "$WORK/last_checkpoint" ]; then
    resume_args+=(--resume)
  fi
  log "START Cascade Mask R-CNN train gpus=$GPUS port=$BASE_PORT"
  PORT="$BASE_PORT" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TRAIN" \
    "$RUNTIME" "$NUM_GPUS" --work-dir "$WORK" "${resume_args[@]}" \
    2>&1 | tee "$log_file"
  best="$(best_checkpoint "$WORK" || true)"
  [ -n "$best" ] || die "Cascade Mask R-CNN training produced no best checkpoint"
  log "TRAINED Cascade Mask R-CNN best=$best"
}

stage_test() {
  local log_file="$OUTPUT_ROOT/logs/test.log"
  local test_work="$OUTPUT_ROOT/test"
  local best
  [ -s "$RUNTIME" ] || stage_prepare
  best="$(best_checkpoint "$WORK" || true)"
  [ -n "$best" ] || die "no best checkpoint available for testing"
  mkdir -p "$test_work" "$OUTPUT_ROOT/logs"
  if [ "$RESUME" = "1" ] && grep -q "coco/${METRIC}_mAP:" "$log_file" 2>/dev/null; then
    log "RECOVER test metrics from $log_file"
    return 0
  fi
  log "TEST Cascade Mask R-CNN metric=$METRIC checkpoint=$best"
  PORT="$((BASE_PORT + 1))" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TEST" \
    "$RUNTIME" "$best" "$NUM_GPUS" --work-dir "$test_work" \
    2>&1 | tee "$log_file"
  grep -q "coco/${METRIC}_mAP:" "$log_file" ||
    die "test did not report coco/${METRIC}_mAP"
}

NAME=""
EASY_ROOT=""
TRAIN_ANN=""
VAL_ANN=""
TEST_ANN=""
ALL_EASY_ANN=""
IMAGE_ROOT=""
TRAIN_IMAGE_ROOT=""
VAL_IMAGE_ROOT=""
TEST_IMAGE_ROOT=""
OUTPUT_ROOT=""
CONFIG="configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
GPUS="0,1"
EPOCHS=24
BATCH_SIZE=2
NUM_WORKERS=2
LR=0.005
MILESTONES="16,22"
METRIC="segm"
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
    --test-ann) TEST_ANN="${2:-}"; shift 2 ;;
    --all-easy-ann) ALL_EASY_ANN="${2:-}"; shift 2 ;;
    --image-root) IMAGE_ROOT="${2:-}"; shift 2 ;;
    --train-image-root) TRAIN_IMAGE_ROOT="${2:-}"; shift 2 ;;
    --val-image-root) VAL_IMAGE_ROOT="${2:-}"; shift 2 ;;
    --test-image-root) TEST_IMAGE_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:-}"; shift 2 ;;
    --lr) LR="${2:-}"; shift 2 ;;
    --milestones) MILESTONES="${2:-}"; shift 2 ;;
    --metric) METRIC="${2:-}"; shift 2 ;;
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
[[ "$METRIC" == bbox || "$METRIC" == segm ]] || die "--metric must be bbox or segm"
case "$START_STAGE" in all|prepare|train|test) ;; *) die "invalid --start-stage" ;; esac
case "$FORCE_STAGE" in ""|all|prepare|train|test) ;; *) die "invalid --force-stage" ;; esac

if [ -z "$EASY_ROOT" ] && [ -z "$TRAIN_ANN" ] && [ -z "$VAL_ANN" ] &&
   [ -z "$ALL_EASY_ANN" ] &&
   [ -z "$IMAGE_ROOT" ] && [ -z "$TRAIN_IMAGE_ROOT" ]; then
  die "--easy-root is required unless explicit annotation and image paths are provided"
fi
if [ -n "$ALL_EASY_ANN" ]; then
  [ -z "$TRAIN_ANN$VAL_ANN$TEST_ANN" ] ||
    die "--all-easy-ann cannot be combined with separate train/val/test annotations"
  require_file "$ALL_EASY_ANN"
  TRAIN_ANN="$ALL_EASY_ANN"
  VAL_ANN="$ALL_EASY_ANN"
  TEST_ANN="$ALL_EASY_ANN"
  if [ -z "$TRAIN_IMAGE_ROOT" ]; then
    if [ -z "$IMAGE_ROOT" ]; then
      [ -n "$EASY_ROOT" ] || die "--all-easy-ann needs --image-root or --easy-root"
      IMAGE_ROOT="$EASY_ROOT/images"
    fi
    TRAIN_IMAGE_ROOT="$IMAGE_ROOT"
  fi
  [ -n "$VAL_IMAGE_ROOT" ] || VAL_IMAGE_ROOT="${IMAGE_ROOT:-$TRAIN_IMAGE_ROOT}"
  [ -n "$TEST_IMAGE_ROOT" ] || TEST_IMAGE_ROOT="${IMAGE_ROOT:-$VAL_IMAGE_ROOT}"
else
  if [ -n "$EASY_ROOT" ]; then
    [ -n "$TRAIN_ANN" ] || TRAIN_ANN="$EASY_ROOT/annotations/instances_train.json"
    [ -n "$VAL_ANN" ] || VAL_ANN="$EASY_ROOT/annotations/instances_val.json"
    if [ -z "$IMAGE_ROOT" ] && [ -z "$TRAIN_IMAGE_ROOT" ]; then
      if [ -d "$EASY_ROOT/images/train" ] && [ -d "$EASY_ROOT/images/val" ]; then
        TRAIN_IMAGE_ROOT="$EASY_ROOT/images/train"
        VAL_IMAGE_ROOT="$EASY_ROOT/images/val"
      else
        IMAGE_ROOT="$EASY_ROOT/images"
      fi
    fi
  fi
  [ -n "$TRAIN_ANN" ] || TRAIN_ANN="$EASY_ROOT/annotations/instances_train.json"
  [ -n "$VAL_ANN" ] || VAL_ANN="$EASY_ROOT/annotations/instances_val.json"
  [ -n "$TRAIN_IMAGE_ROOT" ] || TRAIN_IMAGE_ROOT="$IMAGE_ROOT"
  [ -n "$VAL_IMAGE_ROOT" ] || VAL_IMAGE_ROOT="${IMAGE_ROOT:-$TRAIN_IMAGE_ROOT}"
  [ -n "$TEST_IMAGE_ROOT" ] || TEST_IMAGE_ROOT="$VAL_IMAGE_ROOT"
  [ -n "$TEST_ANN" ] || TEST_ANN="$VAL_ANN"
fi

require_file "$TRAIN_ANN"
require_file "$VAL_ANN"
require_file "$TEST_ANN"
require_dir "$TRAIN_IMAGE_ROOT"
require_dir "$VAL_IMAGE_ROOT"
require_dir "$TEST_IMAGE_ROOT"
if [ -n "$ALL_EASY_ANN" ]; then
  PROTOCOL="all_easy_train_val_test_same"
else
  PROTOCOL="split_or_independent_test"
fi

CONFIG_PATH="$(pick_repo_file "$CONFIG" || true)"
DIST_TRAIN="$(pick_repo_file tools/dist_train.sh || true)"
DIST_TEST="$(pick_repo_file tools/dist_test.sh || true)"
[ -n "$CONFIG_PATH" ] || die "Cascade Mask R-CNN config not found: $CONFIG"
[ -n "$DIST_TRAIN" ] || die "tools/dist_train.sh not found"
[ -n "$DIST_TEST" ] || die "tools/dist_test.sh not found"

[ -n "$OUTPUT_ROOT" ] || OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_cascade_eval/$NAME"
STATE_DIR="$OUTPUT_ROOT/.state"
WORK="$OUTPUT_ROOT/work_dirs/cascade_mask_rcnn_${EPOCHS}e"
RUNTIME="$WORK/runtime_config.py"
NUM_GPUS="$(awk -F, '{print NF}' <<< "$GPUS")"
INPUT_FINGERPRINT="$(
  printf '%s\n' "$PROTOCOL|$NAME|$TRAIN_ANN|$VAL_ANN|$TEST_ANN|$TRAIN_IMAGE_ROOT|$VAL_IMAGE_ROOT|$TEST_IMAGE_ROOT|$CONFIG|$EPOCHS|$BATCH_SIZE|$NUM_WORKERS|$MILESTONES|$LR|$METRIC" |
    sha256sum | cut -d' ' -f1
)"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK_ONLY Cascade Mask R-CNN Easy evaluation"
  echo "name=$NAME gpus=$GPUS"
  echo "train_ann=$TRAIN_ANN"
  echo "val_ann=$VAL_ANN"
  echo "test_ann=$TEST_ANN"
  echo "train_image_root=$TRAIN_IMAGE_ROOT"
  echo "val_image_root=$VAL_IMAGE_ROOT"
  echo "test_image_root=$TEST_IMAGE_ROOT"
  echo "metric=$METRIC"
  echo "protocol=$PROTOCOL"
  echo "output=$OUTPUT_ROOT"
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

log "Cascade Mask R-CNN Easy evaluation name=$NAME gpus=$GPUS output=$OUTPUT_ROOT"
run_from prepare
run_from train
run_from test

BEST="$(best_checkpoint "$WORK" || true)"
{
  printf 'field\tvalue\n'
  printf 'name\t%s\n' "$NAME"
  printf 'train_ann\t%s\n' "$TRAIN_ANN"
  printf 'val_ann\t%s\n' "$VAL_ANN"
  printf 'test_ann\t%s\n' "$TEST_ANN"
  printf 'train_image_root\t%s\n' "$TRAIN_IMAGE_ROOT"
  printf 'val_image_root\t%s\n' "$VAL_IMAGE_ROOT"
  printf 'test_image_root\t%s\n' "$TEST_IMAGE_ROOT"
  printf 'protocol\t%s\n' "$PROTOCOL"
  printf 'gpus\t%s\n' "$GPUS"
  printf 'best_checkpoint\t%s\n' "$BEST"
  printf 'validation_metric\tcoco/%s_mAP\n' "$METRIC"
} > "$OUTPUT_ROOT/run_provenance.tsv"

echo "RESULT=PASS_CASCADE_MASK_RCNN_EASY_EVAL"
echo "BEST_CHECKPOINT=$BEST"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
