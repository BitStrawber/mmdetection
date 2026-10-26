#!/usr/bin/env bash
# Reusable pretrained DINO -> DFUI detector -> RUOD detector pipeline.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

usage() {
  cat <<'EOF'
Usage:
  run_dfui_ruod_pipeline.sh --experiment-id ID --arch r50|vits \
    --pretrained CHECKPOINT --dfui-root DFUI_COCO --gpus 0,1 [options]

Options:
  --ruod-root PATH          RUOD COCO root (default: /media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco)
  --output-root PATH        Experiment output directory
  --dfui-config PATH        DFUI detector base config
  --ruod-config PATH        RUOD detector base config
  --dfui-work PATH          Existing DFUI work dir to reuse
  --dfui-best PATH          Existing DFUI best checkpoint to export
  --dfui-epochs N           DFUI epochs (default: 48)
  --ruod-epochs N           RUOD epochs (default: 24)
  --dfui-lr FLOAT           Override DFUI learning rate
  --ruod-lr FLOAT           Override RUOD learning rate
  --dfui-milestones A,B     DFUI scheduler milestones (default: 32,44)
  --ruod-milestones A,B     RUOD scheduler milestones (default: 16,22)
  --base-port N             First distributed master port (default: 32000)
  --start-stage STAGE       all|convert|dfui_train|dfui_test|export|ruod_train|ruod_test
  --force-stage STAGE       all|convert|dfui_train|dfui_test|export|ruod_train|ruod_test
  --no-resume               Do not pass --resume to unfinished training stages
  --allow-input-change      Reuse output root even when core inputs changed
  --min-free-mib N          Minimum free memory required per GPU (default: 8192)
  --max-keep-ckpts N        Checkpoints retained by MMEngine (default: 5)
  --check-only              Validate inputs and print the plan without writing
  -h, --help                Show this help

Stages are restartable. A failed stage has no .complete marker, so rerunning the
same command resumes that stage or retries it. Completed stages are skipped.
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
  local gpu_group="$1"
  local min_free="$2"
  local gpu
  local free
  local seen=0
  local failed=0
  local -a requested
  IFS=',' read -r -a requested <<< "$gpu_group"
  [ "${#requested[@]}" -gt 0 ] || die "GPU list is empty"
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
  find "$work" -maxdepth 1 -type f -name 'best_*.pth' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

marker() { printf '%s\n' "$STATE_DIR/$1.complete"; }

stage_done() {
  local stage="$1"
  local path
  path="$(marker "$stage")"
  [ -f "$path" ] && [ "${FORCE_STAGE:-}" != "all" ] && [ "$FORCE_STAGE" != "$stage" ]
}

forced_stage() {
  [ "$FORCE_STAGE" = "all" ] || [ "$FORCE_STAGE" = "$1" ]
}

retire_path() {
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

prepare_config() {
  local mode="$1"
  local source_config="$2"
  local root="$3"
  local init_checkpoint="$4"
  local work="$5"
  local runtime="$6"
  local epochs="$7"
  local milestones="$8"
  local lr="$9"

  "$PYTHON_BIN" - \
    "$source_config" "$root" "$init_checkpoint" "$work" "$runtime" \
    "$epochs" "$milestones" "$lr" "$mode" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

(
    source, root_value, init_checkpoint, work_value, runtime,
    epochs_value, milestones_value, lr_value, mode,
) = sys.argv[1:]

source = Path(source)
root = Path(root_value).resolve()
work = Path(work_value).resolve()
runtime = Path(runtime).resolve()
epochs = int(epochs_value)
work.mkdir(parents=True, exist_ok=True)
runtime.parent.mkdir(parents=True, exist_ok=True)

cfg = Config.fromfile(str(source))

def load_split(split: str):
    ann_path = root / "annotations" / f"instances_{split}.json"
    if not ann_path.is_file():
        raise SystemExit(f"missing annotation: {ann_path}")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    return ann_path, data

train_path, train = load_split("train")
val_path, val = load_split("val")

train_categories = sorted(train.get("categories", []), key=lambda item: int(item["id"]))
val_categories = sorted(val.get("categories", []), key=lambda item: int(item["id"]))
if [(x["id"], x["name"]) for x in train_categories] != [
    (x["id"], x["name"]) for x in val_categories
]:
    raise SystemExit("train/val category definitions differ")
classes = tuple(item["name"] for item in train_categories)
if not classes or len(set(classes)) != len(classes):
    raise SystemExit("category names must be non-empty and unique")

def resolve_split(split: str, ann_path: Path, data: dict, preferred_dir: str):
    images = data.get("images", [])
    names = [str(item.get("file_name", "")) for item in images]
    sample = [name for name in names if name][:64]

    candidates = (
        ("", sample),
        ("images/", [str(Path(name).name) for name in sample]),
        (f"{preferred_dir}/", [str(Path(name).name) for name in sample]),
    )
    prefix = None
    for candidate_prefix, candidate_names in candidates:
        if sample and all((root / candidate_prefix / name).is_file() for name in candidate_names):
            prefix = candidate_prefix
            break
    if prefix is None:
        raise SystemExit(
            f"cannot resolve {split} images under {root}; checked root, images/, {preferred_dir}/"
        )

    if prefix == "":
        normalized_path = ann_path
        normalized = data
    else:
        normalized = dict(data)
        normalized_images = []
        for image in images:
            item = dict(image)
            item["file_name"] = Path(str(item["file_name"])).name
            normalized_images.append(item)
        normalized["images"] = normalized_images
        normalized_path = runtime.parent / f"instances_{split}_normalized.json"
        normalized_path.write_text(json.dumps(normalized), encoding="utf-8")

    missing = [
        item["file_name"]
        for item in normalized.get("images", [])
        if not (root / prefix / item["file_name"]).is_file()
    ]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} missing images; examples={missing[:5]}")
    return normalized_path, prefix, normalized

train_ann, train_prefix, train_data = resolve_split("train", train_path, train, "train")
val_ann, val_prefix, val_data = resolve_split("val", val_path, val, "val")

def configure_dataset(dataset, ann_file: Path, prefix: str):
    if "dataset" in dataset:
        configure_dataset(dataset["dataset"], ann_file, prefix)
        return
    if "datasets" in dataset:
        for child in dataset["datasets"]:
            configure_dataset(child, ann_file, prefix)
        return
    dataset["data_root"] = str(root).rstrip("/") + "/"
    dataset["ann_file"] = str(ann_file)
    dataset["data_prefix"] = dict(img=prefix)
    dataset["metainfo"] = dict(classes=classes)

configure_dataset(cfg.train_dataloader.dataset, train_ann, train_prefix)
configure_dataset(cfg.val_dataloader.dataset, val_ann, val_prefix)
configure_dataset(cfg.test_dataloader.dataset, val_ann, val_prefix)

for name in ("val_evaluator", "test_evaluator"):
    evaluator = cfg.get(name)
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            item["ann_file"] = str(val_ann)
    elif evaluator is not None:
        evaluator["ann_file"] = str(val_ann)

heads = cfg.model.roi_head.bbox_head
if not isinstance(heads, (list, tuple)) or len(heads) != 3:
    raise SystemExit("expected a three-stage Cascade R-CNN bbox_head list")
for head in heads:
    head["num_classes"] = len(classes)

cfg.model.backbone.init_cfg = dict(
    type="Pretrained",
    checkpoint=str(Path(init_checkpoint).resolve()),
)
cfg.load_from = None
cfg.resume = False
cfg.train_cfg.max_epochs = epochs
cfg.train_cfg.val_interval = 1
cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
cfg.default_hooks.checkpoint.max_keep_ckpts = int(__import__("os").environ["MAX_KEEP_CKPTS"])
cfg.work_dir = str(work)

if lr_value:
    if "optimizer" not in cfg.optim_wrapper:
        raise SystemExit("base config has no optim_wrapper.optimizer")
    cfg.optim_wrapper.optimizer["lr"] = float(lr_value)

milestones = [int(item) for item in milestones_value.split(",") if item]
multi_step = [
    scheduler
    for scheduler in cfg.param_scheduler
    if scheduler.get("type") == "MultiStepLR"
]
if len(multi_step) != 1:
    raise SystemExit(f"expected one MultiStepLR scheduler, got {len(multi_step)}")
multi_step[0]["begin"] = 0
multi_step[0]["end"] = epochs
multi_step[0]["milestones"] = milestones

cfg.dump(str(runtime))
print(
    f"mode={mode} root={root} classes={len(classes)} "
    f"train={len(train_data['images'])} val={len(val_data['images'])} "
    f"train_prefix={train_prefix or '<root>'} val_prefix={val_prefix or '<root>'} "
    f"epochs={epochs} init={init_checkpoint}"
)
PY
}

run_train_stage() {
  local stage="$1"
  local runtime="$2"
  local work="$3"
  local epochs="$4"
  local port="$5"
  local log_file="$OUTPUT_ROOT/logs/${stage}.log"
  local best
  local -a resume_args=()

  mkdir -p "$work" "$OUTPUT_ROOT/logs"
  best="$(best_checkpoint "$work" || true)"
  if ! forced_stage "$stage" && [ "${RESUME}" = "1" ] && \
     [ -s "$work/epoch_${epochs}.pth" ] && [ -n "$best" ]; then
    log "RECOVER stage=$stage final=$work/epoch_${epochs}.pth best=$best"
    return 0
  fi
  if ! forced_stage "$stage" && [ "${RESUME}" = "1" ] && [ -s "$work/last_checkpoint" ]; then
    resume_args+=(--resume)
  fi

  log "START stage=$stage gpus=$GPUS port=$port"
  PORT="$port" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TRAIN" \
    "$runtime" "$NUM_GPUS" --work-dir "$work" "${resume_args[@]}" \
    2>&1 | tee "$log_file"

  best="$(best_checkpoint "$work" || true)"
  [ -n "$best" ] || die "$stage completed without a best checkpoint; see $log_file"
  log "TRAINED stage=$stage best=$best"
}

run_test_stage() {
  local stage="$1"
  local log_name="$2"
  local runtime="$3"
  local work="$4"
  local port="$5"
  local log_file="$OUTPUT_ROOT/logs/${log_name}_test.log"
  local test_work="$OUTPUT_ROOT/test/$stage"
  local best

  best="$(best_checkpoint "$work" || true)"
  [ -n "$best" ] || die "$stage has no best checkpoint"
  mkdir -p "$test_work" "$OUTPUT_ROOT/logs"
  if ! forced_stage "$stage" && [ "${RESUME}" = "1" ] && \
     grep -q 'coco/bbox_mAP:' "$log_file" 2>/dev/null; then
    log "RECOVER stage=$stage test metric exists in $log_file"
    return 0
  fi

  log "TEST stage=$stage checkpoint=$best"
  PORT="$port" CUDA_VISIBLE_DEVICES="$GPUS" bash "$DIST_TEST" \
    "$runtime" "$best" "$NUM_GPUS" --work-dir "$test_work" \
    2>&1 | tee "$log_file"
  grep -q 'coco/bbox_mAP:' "$log_file" || die "$stage test did not report coco/bbox_mAP"
}

EXPERIMENT_ID=""
ARCH=""
PRETRAINED=""
DFUI_ROOT=""
RUOD_ROOT="/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco"
OUTPUT_ROOT=""
DFUI_CONFIG=""
RUOD_CONFIG=""
DFUI_WORK_OVERRIDE=""
DFUI_BEST_OVERRIDE=""
GPUS="0,1"
DFUI_EPOCHS=48
RUOD_EPOCHS=24
DFUI_LR=""
RUOD_LR=""
DFUI_MILESTONES="32,44"
RUOD_MILESTONES="16,22"
BASE_PORT=32000
START_STAGE="all"
FORCE_STAGE=""
RESUME=1
ALLOW_INPUT_CHANGE=0
MIN_FREE_MIB=8192
MAX_KEEP_CKPTS=5
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
    --arch) ARCH="${2:-}"; shift 2 ;;
    --pretrained) PRETRAINED="${2:-}"; shift 2 ;;
    --dfui-root) DFUI_ROOT="${2:-}"; shift 2 ;;
    --ruod-root) RUOD_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --dfui-config) DFUI_CONFIG="${2:-}"; shift 2 ;;
    --ruod-config) RUOD_CONFIG="${2:-}"; shift 2 ;;
    --dfui-work) DFUI_WORK_OVERRIDE="${2:-}"; shift 2 ;;
    --dfui-best) DFUI_BEST_OVERRIDE="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --dfui-epochs) DFUI_EPOCHS="${2:-}"; shift 2 ;;
    --ruod-epochs) RUOD_EPOCHS="${2:-}"; shift 2 ;;
    --dfui-lr) DFUI_LR="${2:-}"; shift 2 ;;
    --ruod-lr) RUOD_LR="${2:-}"; shift 2 ;;
    --dfui-milestones) DFUI_MILESTONES="${2:-}"; shift 2 ;;
    --ruod-milestones) RUOD_MILESTONES="${2:-}"; shift 2 ;;
    --base-port) BASE_PORT="${2:-}"; shift 2 ;;
    --start-stage) START_STAGE="${2:-}"; shift 2 ;;
    --force-stage) FORCE_STAGE="${2:-}"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --allow-input-change) ALLOW_INPUT_CHANGE=1; shift ;;
    --min-free-mib) MIN_FREE_MIB="${2:-}"; shift 2 ;;
    --max-keep-ckpts) MAX_KEEP_CKPTS="${2:-}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$ARCH" in r50|vits) ;; *) die "--arch must be r50 or vits" ;; esac
case "$START_STAGE" in
  all|convert|dfui_train|dfui_test|export|ruod_train|ruod_test) ;;
  *) die "invalid --start-stage: $START_STAGE" ;;
esac
case "$FORCE_STAGE" in
  ""|all|convert|dfui_train|dfui_test|export|ruod_train|ruod_test) ;;
  *) die "invalid --force-stage: $FORCE_STAGE" ;;
esac
[ -n "$EXPERIMENT_ID" ] || die "--experiment-id is required"
[ -n "$PRETRAINED" ] || die "--pretrained is required"
[ -n "$DFUI_ROOT" ] || die "--dfui-root is required"
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid experiment id"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --gpus"
[ "$(tr ',' '\n' <<< "$GPUS" | sort -u | wc -l)" -eq "$(awk -F, '{print NF}' <<< "$GPUS")" ] ||
  die "GPU indices must be unique"
[ "$DFUI_EPOCHS" -gt 0 ] && [ "$RUOD_EPOCHS" -gt 0 ] || die "epochs must be positive"
if [ -n "$DFUI_WORK_OVERRIDE" ] || [ -n "$DFUI_BEST_OVERRIDE" ]; then
  case "$START_STAGE" in
    export|ruod_train|ruod_test) ;;
    *) die "--dfui-work/--dfui-best require --start-stage export, ruod_train, or ruod_test" ;;
  esac
fi
[ -z "$DFUI_WORK_OVERRIDE" ] || require_dir "$DFUI_WORK_OVERRIDE"
[ -z "$DFUI_BEST_OVERRIDE" ] || require_file "$DFUI_BEST_OVERRIDE"

require_file "$PRETRAINED"
require_dir "$DFUI_ROOT"
require_dir "$RUOD_ROOT"
require_file "$DFUI_ROOT/annotations/instances_train.json"
require_file "$DFUI_ROOT/annotations/instances_val.json"
require_file "$RUOD_ROOT/annotations/instances_train.json"
require_file "$RUOD_ROOT/annotations/instances_val.json"

DIST_TRAIN="$(pick_repo_file tools/dist_train.sh || true)"
DIST_TEST="$(pick_repo_file tools/dist_test.sh || true)"
CONVERT_TOOL="$(pick_repo_file tools/convert_ssl_backbone_to_mmdet.py || true)"
[ -n "$DIST_TRAIN" ] || die "tools/dist_train.sh not found"
[ -n "$DIST_TEST" ] || die "tools/dist_test.sh not found"
[ -n "$CONVERT_TOOL" ] || die "tools/convert_ssl_backbone_to_mmdet.py not found"

if [ -z "$DFUI_CONFIG" ]; then
  if [ "$ARCH" = "r50" ]; then
    DFUI_CONFIG="configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_easy_j10_scheme_c_s1.py"
  else
    DFUI_CONFIG="configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py"
  fi
fi
if [ -z "$RUOD_CONFIG" ]; then
  if [ "$ARCH" = "r50" ]; then
    RUOD_CONFIG="configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py"
  else
    RUOD_CONFIG="configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py"
  fi
fi
DFUI_CONFIG_PATH="$(pick_repo_file "$DFUI_CONFIG" || true)"
RUOD_CONFIG_PATH="$(pick_repo_file "$RUOD_CONFIG" || true)"
[ -n "$DFUI_CONFIG_PATH" ] || die "DFUI config not found: $DFUI_CONFIG"
[ -n "$RUOD_CONFIG_PATH" ] || die "RUOD config not found: $RUOD_CONFIG"

[ -n "$OUTPUT_ROOT" ] || OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_dfui_ruod/$EXPERIMENT_ID"
STATE_DIR="$OUTPUT_ROOT/.state"
NUM_GPUS="$(awk -F, '{print NF}' <<< "$GPUS")"
INPUT_FINGERPRINT="$(
  printf '%s\n' "$EXPERIMENT_ID|$ARCH|$PRETRAINED|$DFUI_ROOT|$RUOD_ROOT|$DFUI_CONFIG|$RUOD_CONFIG|$DFUI_WORK_OVERRIDE|$DFUI_BEST_OVERRIDE|$DFUI_EPOCHS|$RUOD_EPOCHS|$DFUI_LR|$RUOD_LR|$DFUI_MILESTONES|$RUOD_MILESTONES" |
    sha256sum | cut -d' ' -f1
)"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK_ONLY pipeline"
  echo "experiment_id=$EXPERIMENT_ID arch=$ARCH gpus=$GPUS"
  echo "pretrained=$PRETRAINED"
  echo "dfui_root=$DFUI_ROOT"
  echo "ruod_root=$RUOD_ROOT"
  echo "dfui_config=$DFUI_CONFIG"
  echo "ruod_config=$RUOD_CONFIG"
  echo "stages=convert,dfui_train,dfui_test,export,ruod_train,ruod_test"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/backbones" \
  "$OUTPUT_ROOT/work_dirs" "$OUTPUT_ROOT/test" "$STATE_DIR"

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
gpu_preflight "$GPUS" "$MIN_FREE_MIB"
export MAX_KEEP_CKPTS

CONVERTED="$OUTPUT_ROOT/converted/${EXPERIMENT_ID}_${ARCH}_teacher.pth"
if [ -n "$DFUI_WORK_OVERRIDE" ]; then
  DFUI_WORK="$DFUI_WORK_OVERRIDE"
else
  DFUI_WORK="$OUTPUT_ROOT/work_dirs/${EXPERIMENT_ID}_${ARCH}_dfui${DFUI_EPOCHS}e"
fi
DFUI_RUNTIME="$DFUI_WORK/runtime_config.py"
DFUI_TEST_LOG="$OUTPUT_ROOT/logs/dfui_test.log"
ADAPTED_BACKBONE="$OUTPUT_ROOT/backbones/${EXPERIMENT_ID}_${ARCH}_dfui${DFUI_EPOCHS}e_best_backbone.pth"
RUOD_WORK="$OUTPUT_ROOT/work_dirs/${EXPERIMENT_ID}_${ARCH}_ruod${RUOD_EPOCHS}e"
RUOD_RUNTIME="$RUOD_WORK/runtime_config.py"

stage_convert() {
  local prepend=""
  [ "$ARCH" = "vits" ] && prepend="backbone."
  if forced_stage convert; then
    retire_path "$CONVERTED"
  fi
  if [ -s "$CONVERTED" ]; then
    log "REUSE converted checkpoint=$CONVERTED"
    return 0
  fi
  "$PYTHON_BIN" "$CONVERT_TOOL" \
    --checkpoint "$PRETRAINED" --source teacher --prepend "$prepend" \
    --out "$CONVERTED"
}

stage_dfui_train() {
  if forced_stage dfui_train; then
    retire_path "$DFUI_WORK"
  fi
  prepare_config "dfui" "$DFUI_CONFIG_PATH" "$DFUI_ROOT" "$CONVERTED" \
    "$DFUI_WORK" "$DFUI_RUNTIME" "$DFUI_EPOCHS" "$DFUI_MILESTONES" "$DFUI_LR"
  run_train_stage "dfui_train" "$DFUI_RUNTIME" "$DFUI_WORK" "$DFUI_EPOCHS" "$BASE_PORT"
}

stage_dfui_test() {
  run_test_stage "dfui_test" "dfui" "$DFUI_RUNTIME" "$DFUI_WORK" "$((BASE_PORT + 1))"
}

stage_export() {
  local best
  best="$DFUI_BEST_OVERRIDE"
  [ -n "$best" ] || best="$(best_checkpoint "$DFUI_WORK" || true)"
  [ -n "$best" ] || die "cannot export backbone: no DFUI best checkpoint"
  require_file "$best"
  if forced_stage export; then
    retire_path "$ADAPTED_BACKBONE"
  fi
  if [ -s "$ADAPTED_BACKBONE" ]; then
    log "REUSE exported backbone=$ADAPTED_BACKBONE"
    return 0
  fi
  "$PYTHON_BIN" - "$best" "$ADAPTED_BACKBONE" <<'PY'
import sys
from pathlib import Path

import torch

source, output = map(Path, sys.argv[1:])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
state = checkpoint.get("state_dict", checkpoint)
backbone = {
    key[len("backbone."):]: value
    for key, value in state.items()
    if key.startswith("backbone.")
}
if not backbone:
    raise SystemExit(f"no backbone tensors in {source}")
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {"state_dict": backbone, "meta": {"source": str(source)}},
    output,
)
print(f"exported={output} tensors={len(backbone)}")
PY
}

stage_ruod_train() {
  require_file "$ADAPTED_BACKBONE"
  if forced_stage ruod_train; then
    retire_path "$RUOD_WORK"
  fi
  prepare_config "ruod" "$RUOD_CONFIG_PATH" "$RUOD_ROOT" "$ADAPTED_BACKBONE" \
    "$RUOD_WORK" "$RUOD_RUNTIME" "$RUOD_EPOCHS" "$RUOD_MILESTONES" "$RUOD_LR"
  run_train_stage "ruod_train" "$RUOD_RUNTIME" "$RUOD_WORK" "$RUOD_EPOCHS" "$((BASE_PORT + 10))"
}

stage_ruod_test() {
  run_test_stage "ruod_test" "ruod" "$RUOD_RUNTIME" "$RUOD_WORK" "$((BASE_PORT + 11))"
}

stage_rank() {
  case "$1" in
    all) echo 0 ;;
    convert) echo 1 ;;
    dfui_train) echo 2 ;;
    dfui_test) echo 3 ;;
    export) echo 4 ;;
    ruod_train) echo 5 ;;
    ruod_test) echo 6 ;;
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

log "experiment=$EXPERIMENT_ID arch=$ARCH gpus=$GPUS output=$OUTPUT_ROOT"
run_from convert
run_from dfui_train
run_from dfui_test
run_from export
run_from ruod_train
run_from ruod_test

DFUI_BEST="$DFUI_BEST_OVERRIDE"
[ -n "$DFUI_BEST" ] || DFUI_BEST="$(best_checkpoint "$DFUI_WORK" || true)"
RUOD_BEST="$(best_checkpoint "$RUOD_WORK" || true)"
{
  printf 'field\tvalue\n'
  printf 'experiment_id\t%s\n' "$EXPERIMENT_ID"
  printf 'arch\t%s\n' "$ARCH"
  printf 'pretrained\t%s\n' "$PRETRAINED"
  printf 'dfui_root\t%s\n' "$DFUI_ROOT"
  printf 'ruod_root\t%s\n' "$RUOD_ROOT"
  printf 'gpus\t%s\n' "$GPUS"
  printf 'dfui_best\t%s\n' "$DFUI_BEST"
  printf 'adapted_backbone\t%s\n' "$ADAPTED_BACKBONE"
  printf 'ruod_best\t%s\n' "$RUOD_BEST"
} > "$OUTPUT_ROOT/run_provenance.tsv"

log "PIPELINE COMPLETE dfui_best=$DFUI_BEST ruod_best=$RUOD_BEST"
echo "RESULT=PASS_DFUI_RUOD_PIPELINE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
