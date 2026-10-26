#!/usr/bin/env bash
# Reusable Cascade R-CNN A/B cross-filtering pipeline.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

usage() {
  cat <<'EOF'
Usage:
  run_cascade_easy_abba.sh --name NAME --annotation COCO_JSON \
    --image-root IMAGE_DIR --gpus 0,1 [options]

Options:
  --output-root PATH        Default: work_dirs/auto_easy/NAME
  --config PATH             Cascade R-CNN base config
  --epochs N                Epochs for each A/B direction (default: 24)
  --batch-size N            Per-GPU batch size (default: 6)
  --lr FLOAT                Learning rate (default: 0.015)
  --milestones A,B          Scheduler milestones (default: 16,22)
  --threshold FLOAT         Per-image bbox AP threshold (default: 0.60)
  --ab-seed N               A/B split seed (default: 42)
  --sample-seed N           Final sampling seed (default: 20261016)
  --sample-count N          Default: 11494
  --train-count N           Default: 9770
  --val-count N             Default: 1724
  --base-port N             Default: 34000
  --start-stage STAGE       all|split|train_ab|filter_b|train_ba|filter_a|merge|sample
  --force-stage STAGE       all|split|train_ab|filter_b|train_ba|filter_a|merge|sample
  --no-sample               Stop after merging the Easy pool
  --no-resume               Do not pass --resume to unfinished training
  --allow-input-change      Reuse an output directory with different core inputs
  --min-free-mib N          Minimum free GPU memory (default: 8192)
  --check-only              Validate and print the plan without writing
  -h, --help                Show this help
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
  find "$work" -maxdepth 1 -type f -name 'best_coco_bbox_mAP*.pth' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
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

complete_stage() {
  local stage="$1"
  mkdir -p "$STATE_DIR"
  date '+%F %T' > "$(marker "$stage")"
  log "COMPLETE stage=$stage"
}

retire_directory() {
  local path="$1"
  local parent
  local name
  [ -e "$path" ] || return 0
  parent="$(dirname "$path")"
  name="$(basename "$path")"
  mv -- "$path" "$parent/${name}.retired.$(date +%Y%m%d_%H%M%S)"
}

write_runtime_config() {
  local ann_train="$1"
  local ann_val="$2"
  local work="$3"
  local runtime="$4"
  "$PYTHON_BIN" - \
    "$CONFIG_PATH" "$IMAGE_ROOT" "$ann_train" "$ann_val" "$work" "$runtime" \
    "$EPOCHS" "$BATCH_SIZE" "$LR" "$MILESTONES" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

(
    source, image_root, train_ann, val_ann, work, runtime,
    epochs_value, batch_value, lr_value, milestones_value,
) = sys.argv[1:]

source = Path(source)
image_root = Path(image_root).resolve()
train_ann = Path(train_ann).resolve()
val_ann = Path(val_ann).resolve()
work = Path(work).resolve()
runtime = Path(runtime).resolve()
epochs = int(epochs_value)

cfg = Config.fromfile(str(source))
if cfg.model.type != "CascadeRCNN":
    raise SystemExit(f"expected CascadeRCNN, got {cfg.model.type}")

train = json.loads(train_ann.read_text(encoding="utf-8"))
val = json.loads(val_ann.read_text(encoding="utf-8"))
categories = sorted(train.get("categories", []), key=lambda item: int(item["id"]))
val_categories = sorted(val.get("categories", []), key=lambda item: int(item["id"]))
if [(x["id"], x["name"]) for x in categories] != [
    (x["id"], x["name"]) for x in val_categories
]:
    raise SystemExit("A/B category definitions differ")
classes = tuple(item["name"] for item in categories)

for split, ann_path, data in (("train", train_ann, train), ("val", val_ann, val)):
    missing = [
        image["file_name"]
        for image in data.get("images", [])
        if not (image_root / image["file_name"]).is_file()
    ]
    if missing:
        raise SystemExit(f"{split}: missing {len(missing)} images; examples={missing[:5]}")

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

configure(cfg.train_dataloader.dataset, train_ann)
configure(cfg.val_dataloader.dataset, val_ann)
configure(cfg.test_dataloader.dataset, val_ann)
cfg.train_dataloader.batch_size = int(batch_value)
cfg.train_cfg.max_epochs = epochs
cfg.train_cfg.val_interval = 1
cfg.optim_wrapper.optimizer.lr = float(lr_value)

heads = cfg.model.roi_head.bbox_head
if not isinstance(heads, (list, tuple)) or len(heads) != 3:
    raise SystemExit("expected three Cascade R-CNN bbox heads")
for head in heads:
    head["num_classes"] = len(classes)

for evaluator_name in ("val_evaluator", "test_evaluator"):
    cfg[evaluator_name].ann_file = str(val_ann)

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

cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
cfg.default_hooks.checkpoint.max_keep_ckpts = 3
cfg.load_from = None
cfg.resume = False
cfg.work_dir = str(work)
work.mkdir(parents=True, exist_ok=True)
cfg.dump(str(runtime))
print(
    f"runtime={runtime} classes={len(classes)} "
    f"train={len(train['images'])} val={len(val['images'])} epochs={epochs}"
)
PY
}

train_direction() {
  local train_fold="$1"
  local val_fold="$2"
  local stage="train_${train_fold}_${val_fold}"
  local work="$OUTPUT_ROOT/work_dirs/$stage"
  local runtime="$work/runtime_config.py"
  local log_file="$OUTPUT_ROOT/logs/$stage.log"
  local best
  local port="$BASE_PORT"
  local -a resume_args=()

  [ "$train_fold" = "A" ] || port="$((BASE_PORT + 1))"
  if forced_stage "$stage"; then
    retire_directory "$work"
  fi
  write_runtime_config "$AB_ROOT/train_${train_fold}.json" "$AB_ROOT/train_${val_fold}.json" \
    "$work" "$runtime"

  best="$(best_checkpoint "$work" || true)"
  if ! forced_stage "$stage" && [ "$RESUME" = "1" ] && \
     [ -s "$work/epoch_${EPOCHS}.pth" ] && [ -n "$best" ]; then
    log "RECOVER stage=$stage best=$best"
    printf '%s\n' "$best" > "$work/best_checkpoint.txt"
    return 0
  fi
  if ! forced_stage "$stage" && [ "$RESUME" = "1" ] && [ -s "$work/last_checkpoint" ]; then
    resume_args+=(--resume)
  fi

  log "START stage=$stage gpus=$GPUS port=$port"
  PORT="$port" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TRAIN" \
    "$runtime" "$NUM_GPUS" --work-dir "$work" "${resume_args[@]}" \
    2>&1 | tee "$log_file"
  best="$(best_checkpoint "$work" || true)"
  [ -n "$best" ] || die "$stage did not produce a best checkpoint"
  printf '%s\n' "$best" > "$work/best_checkpoint.txt"
  log "TRAINED stage=$stage best=$best"
}

filter_direction() {
  local train_fold="$1"
  local target_fold="$2"
  local stage="filter_${target_fold}"
  local work="$OUTPUT_ROOT/work_dirs/train_${train_fold}_${target_fold}"
  local best_file="$work/best_checkpoint.txt"
  local output="$EASY_ROOT/easy_${target_fold}"
  local log_file="$OUTPUT_ROOT/logs/filter_${target_fold}.log"
  local checkpoint

  require_file "$best_file"
  checkpoint="$(cat "$best_file")"
  require_file "$checkpoint"
  log "FILTER stage=$stage teacher=$checkpoint"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON_BIN" "$COCO_HELPER" filter \
    --annotation "$AB_ROOT/train_${target_fold}.json" \
    --image-root "$IMAGE_ROOT" \
    --config "$work/runtime_config.py" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --threshold "$THRESHOLD" \
    --device cuda:0 2>&1 | tee "$log_file"

  "$PYTHON_BIN" - "$output/instances_easy.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
count = len(data.get("images", []))
print(f"easy_images={count} annotation={path}")
if count == 0:
    raise SystemExit("Easy filter retained zero images")
PY
}

merge_easy() {
  "$PYTHON_BIN" - \
    "$EASY_ROOT/easy_A/instances_easy.json" \
    "$EASY_ROOT/easy_B/instances_easy.json" \
    "$EASY_ROOT/instances_easy_merged.json" <<'PY'
import json
import sys
from pathlib import Path

a_path, b_path, output = map(Path, sys.argv[1:])
a = json.loads(a_path.read_text(encoding="utf-8"))
b = json.loads(b_path.read_text(encoding="utf-8"))
ids_a = {int(item["id"]) for item in a.get("images", [])}
ids_b = {int(item["id"]) for item in b.get("images", [])}
if ids_a & ids_b:
    raise SystemExit("A/B Easy pools overlap")
merged = dict(a)
merged["images"] = a.get("images", []) + b.get("images", [])
merged["annotations"] = a.get("annotations", []) + b.get("annotations", [])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
print(f"easy_images={len(merged['images'])} easy_annotations={len(merged['annotations'])}")
PY
}

stage_split() {
  if forced_stage split; then
    retire_directory "$AB_ROOT"
  fi
  if [ -s "$AB_ROOT/train_A.json" ] && [ -s "$AB_ROOT/train_B.json" ]; then
    log "REUSE A/B split=$AB_ROOT"
    return 0
  fi
  "$PYTHON_BIN" "$COCO_HELPER" split \
    --annotation "$ANNOTATION" --output "$AB_ROOT" --seed "$AB_SEED" \
    2>&1 | tee "$OUTPUT_ROOT/logs/split.log"
}

stage_train_ab() { train_direction A B; }
stage_filter_b() { filter_direction A B; }
stage_train_ba() { train_direction B A; }
stage_filter_a() { filter_direction B A; }
stage_merge() { merge_easy; }

stage_sample() {
  [ "$NO_SAMPLE" = "0" ] || {
    log "SKIP sample because --no-sample was requested"
    return 0
  }
  retire_directory "$SAMPLE_ROOT"
  "$PYTHON_BIN" "$COCO_HELPER" sample \
    --easy-annotation "$EASY_ROOT/instances_easy_merged.json" \
    --image-root "$IMAGE_ROOT" \
    --output "$SAMPLE_ROOT" \
    --count "$SAMPLE_COUNT" \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --seed "$SAMPLE_SEED" \
    2>&1 | tee "$OUTPUT_ROOT/logs/sample.log"
}

NAME=""
ANNOTATION=""
IMAGE_ROOT=""
OUTPUT_ROOT=""
CONFIG="configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py"
GPUS="0,1"
EPOCHS=24
BATCH_SIZE=6
LR=0.015
MILESTONES="16,22"
THRESHOLD=0.60
AB_SEED=42
SAMPLE_SEED=20261016
SAMPLE_COUNT=11494
TRAIN_COUNT=9770
VAL_COUNT=1724
BASE_PORT=34000
START_STAGE="all"
FORCE_STAGE=""
NO_SAMPLE=0
RESUME=1
ALLOW_INPUT_CHANGE=0
MIN_FREE_MIB=8192
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) NAME="${2:-}"; shift 2 ;;
    --annotation) ANNOTATION="${2:-}"; shift 2 ;;
    --image-root) IMAGE_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --lr) LR="${2:-}"; shift 2 ;;
    --milestones) MILESTONES="${2:-}"; shift 2 ;;
    --threshold) THRESHOLD="${2:-}"; shift 2 ;;
    --ab-seed) AB_SEED="${2:-}"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="${2:-}"; shift 2 ;;
    --sample-count) SAMPLE_COUNT="${2:-}"; shift 2 ;;
    --train-count) TRAIN_COUNT="${2:-}"; shift 2 ;;
    --val-count) VAL_COUNT="${2:-}"; shift 2 ;;
    --base-port) BASE_PORT="${2:-}"; shift 2 ;;
    --start-stage) START_STAGE="${2:-}"; shift 2 ;;
    --force-stage) FORCE_STAGE="${2:-}"; shift 2 ;;
    --no-sample) NO_SAMPLE=1; shift ;;
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
[ -n "$ANNOTATION" ] || die "--annotation is required"
[ -n "$IMAGE_ROOT" ] || die "--image-root is required"
require_file "$ANNOTATION"
require_dir "$IMAGE_ROOT"
[[ "$GPUS" =~ ^[0-9]+,[0-9]+$ ]] || die "--gpus must contain exactly two GPU indices"
[ "${GPUS%%,*}" != "${GPUS##*,}" ] || die "GPU indices must differ"
[ "$SAMPLE_COUNT" -eq $((TRAIN_COUNT + VAL_COUNT)) ] ||
  die "sample count must equal train count + val count"
case "$START_STAGE" in
  all|split|train_ab|filter_b|train_ba|filter_a|merge|sample) ;;
  *) die "invalid --start-stage: $START_STAGE" ;;
esac
case "$FORCE_STAGE" in
  ""|all|split|train_ab|filter_b|train_ba|filter_a|merge|sample) ;;
  *) die "invalid --force-stage: $FORCE_STAGE" ;;
esac

CONFIG_PATH="$(pick_repo_file "$CONFIG" || true)"
COCO_HELPER="$(pick_repo_file tools/exp_2/coco_cross_easy.py || true)"
DIST_TRAIN="$(pick_repo_file tools/dist_train.sh || true)"
[ -n "$CONFIG_PATH" ] || die "Cascade config not found: $CONFIG"
[ -n "$COCO_HELPER" ] || die "tools/exp_2/coco_cross_easy.py not found"
[ -n "$DIST_TRAIN" ] || die "tools/dist_train.sh not found"

[ -n "$OUTPUT_ROOT" ] || OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_easy/$NAME"
STATE_DIR="$OUTPUT_ROOT/.state"
NUM_GPUS=2
AB_ROOT="$OUTPUT_ROOT/ab_split"
EASY_ROOT="$OUTPUT_ROOT/easy_full"
SAMPLE_ROOT="$OUTPUT_ROOT/DFUI_COCOEasy"
INPUT_FINGERPRINT="$(
  printf '%s\n' "$NAME|$ANNOTATION|$IMAGE_ROOT|$CONFIG|$EPOCHS|$BATCH_SIZE|$LR|$MILESTONES|$THRESHOLD|$AB_SEED|$SAMPLE_SEED|$SAMPLE_COUNT|$TRAIN_COUNT|$VAL_COUNT" |
    sha256sum | cut -d' ' -f1
)"

"$PYTHON_BIN" - "$ANNOTATION" "$IMAGE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

annotation, image_root = map(Path, sys.argv[1:])
data = json.loads(annotation.read_text(encoding="utf-8"))
images = data.get("images", [])
if len(images) < 2:
    raise SystemExit("annotation must contain at least two images")
missing = [item["file_name"] for item in images if not (image_root / item["file_name"]).is_file()]
print(f"images={len(images)} annotations={len(data.get('annotations', []))} "
      f"categories={len(data.get('categories', []))} missing={len(missing)}")
if missing:
    raise SystemExit(f"missing images: {missing[:5]}")
PY

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK_ONLY Cascade Easy AB/BA"
  echo "name=$NAME gpus=$GPUS"
  echo "annotation=$ANNOTATION"
  echo "image_root=$IMAGE_ROOT"
  echo "output=$OUTPUT_ROOT"
  echo "stages=split,train_ab,filter_b,train_ba,filter_a,merge,sample"
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
    split) echo 1 ;;
    train_ab) echo 2 ;;
    filter_b) echo 3 ;;
    train_ba) echo 4 ;;
    filter_a) echo 5 ;;
    merge) echo 6 ;;
    sample) echo 7 ;;
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

log "Cascade Easy AB/BA name=$NAME gpus=$GPUS output=$OUTPUT_ROOT"
run_from split
run_from train_ab
run_from filter_b
run_from train_ba
run_from filter_a
run_from merge
run_from sample

{
  printf 'field\tvalue\n'
  printf 'name\t%s\n' "$NAME"
  printf 'annotation\t%s\n' "$ANNOTATION"
  printf 'image_root\t%s\n' "$IMAGE_ROOT"
  printf 'gpus\t%s\n' "$GPUS"
  printf 'threshold\t%s\n' "$THRESHOLD"
  printf 'ab_seed\t%s\n' "$AB_SEED"
  printf 'easy_pool\t%s\n' "$EASY_ROOT/instances_easy_merged.json"
  printf 'sample_root\t%s\n' "$SAMPLE_ROOT"
} > "$OUTPUT_ROOT/run_provenance.tsv"

echo "RESULT=PASS_CASCADE_EASY_ABBA"
echo "EASY_POOL=$EASY_ROOT/instances_easy_merged.json"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
