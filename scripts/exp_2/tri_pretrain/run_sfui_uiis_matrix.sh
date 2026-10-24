#!/usr/bin/env bash
set -Eeuo pipefail

# Manifest-driven SFUI -> UIIS10K Mask R-CNN matrix runner.
# Each manifest row may choose its own backbone, pretrained checkpoint,
# SFUI dataset, annotations, prefixes, and optional base configs.

die() {
  echo "ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:?set MANIFEST to a pipe-delimited experiment manifest}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new matrix output directory}"

R50_RUNNER="${R50_RUNNER:-$REPO_ROOT/scripts/exp_2/datasets/run_sfui_to_uiis_r50_mask.sh}"
VITS_RUNNER="${VITS_RUNNER:-$REPO_ROOT/scripts/exp_2/datasets/run_sfui_to_uiis_vits_mask.sh}"

R50_GPUS="${R50_GPUS:-6,7}"
VITS_GPUS="${VITS_GPUS:-6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-60}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
BASE_PORT="${BASE_PORT:-39800}"
SFUI_EPOCHS="${SFUI_EPOCHS:-48}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
RUN_TEST="${RUN_TEST:-1}"
CHECK_ONLY="${CHECK_ONLY:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MASTER_LOG="$OUTPUT_ROOT/logs/master_pipeline.log"
PROVENANCE="$OUTPUT_ROOT/run_provenance.tsv"

die_if_missing_file() {
  [ -s "$1" ] || die "missing or empty file: $1"
}

die_if_missing_dir() {
  [ -d "$1" ] || die "missing directory: $1"
}

gpu_count() {
  awk -F, '{print NF}' <<< "$1"
}

wait_for_gpus() {
  local group="$1" label="$2"
  [ "$WAIT_FOR_GPU" = 1 ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

  while true; do
    local ready=1 gpu free
    IFS=',' read -r -a ids <<< "$group"
    for gpu in "${ids[@]}"; do
      free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
      echo "GPU=$gpu free_MiB=$free required_MiB=$MIN_FREE_MIB stage=$label"
      if ! [[ "$free" =~ ^[0-9]+$ ]] || [ "$free" -lt "$MIN_FREE_MIB" ]; then
        ready=0
      fi
    done
    [ "$ready" = 1 ] && return 0
    echo "Waiting ${GPU_WAIT_INTERVAL}s for GPU group $group before $label"
    sleep "$GPU_WAIT_INTERVAL"
  done
}

check_port() {
  local port="$1"
  command -v ss >/dev/null 2>&1 || die "ss is required to check distributed ports"
  if ss -H -ltn | awk -v suffix=":$port" '$4 ~ (suffix "$") { found=1 } END { exit !found }'; then
    die "distributed port $port is already in use"
  fi
}

validate_sfui_data() {
  local root="$1" train_ann="$2" val_ann="$3" train_prefix="$4" val_prefix="$5"
  die_if_missing_dir "$root"
  die_if_missing_file "$train_ann"
  [ -z "$val_ann" ] || die_if_missing_file "$val_ann"

  "$PYTHON_BIN" - "$root" "$train_ann" "$val_ann" "$train_prefix" "$val_prefix" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_ann = Path(sys.argv[2])
val_ann = Path(sys.argv[3]) if sys.argv[3] else None
train_prefix = sys.argv[4]
val_prefix = sys.argv[5]

def load(path, prefix):
    data = json.loads(path.read_text(encoding='utf-8'))
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    if not categories:
        raise SystemExit(f'no categories: {path}')
    image_ids = [x['id'] for x in data.get('images', [])]
    if len(image_ids) != len(set(image_ids)):
        raise SystemExit(f'duplicate image ids: {path}')
    image_root = root / prefix
    missing = [x['file_name'] for x in data.get('images', [])
               if not (image_root / x['file_name']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} missing images under {image_root}: {missing[:3]}')
    return data, [(x['id'], x['name']) for x in categories]

train, train_categories = load(train_ann, train_prefix)
val = None
if val_ann:
    val, val_categories = load(val_ann, val_prefix)
    if train_categories != val_categories:
        raise SystemExit('SFUI train/val categories differ')
    train_names = {x['file_name'] for x in train['images']}
    val_names = {x['file_name'] for x in val['images']}
    if train_names & val_names:
        raise SystemExit('SFUI train/val filename overlap')

print(f'SFUI train images={len(train["images"])} annotations={len(train.get("annotations", []))}')
print(f'SFUI val images={len(val["images"]) if val else 0} annotations={len(val.get("annotations", [])) if val else 0}')
print(f'SFUI categories={len(train_categories)}')
print('RESULT=PASS_SFUI_MATRIX_DATA_PREFLIGHT')
PY
}

validate_uiis() {
  die_if_missing_dir "$UIIS_ROOT"
  die_if_missing_file "$UIIS_ROOT/annotations/instances_train.json"
  die_if_missing_file "$UIIS_ROOT/annotations/instances_val.json"
  [ -d "$UIIS_ROOT/train" ] || die "missing UIIS train directory: $UIIS_ROOT/train"
  [ -d "$UIIS_ROOT/val" ] || die "missing UIIS val directory: $UIIS_ROOT/val"
}

mkdir -p "$OUTPUT_ROOT/logs"
die_if_missing_file "$MANIFEST"
die_if_missing_file "$R50_RUNNER"
die_if_missing_file "$VITS_RUNNER"
validate_uiis

{
  printf 'field\tvalue\n'
  printf 'manifest\t%s\n' "$MANIFEST"
  printf 'uiis_root\t%s\n' "$UIIS_ROOT"
  printf 'r50_gpus\t%s\n' "$R50_GPUS"
  printf 'vits_gpus\t%s\n' "$VITS_GPUS"
  printf 'min_free_mib\t%s\n' "$MIN_FREE_MIB"
  printf 'sfui_epochs\t%s\n' "$SFUI_EPOCHS"
  printf 'uiis_epochs\t%s\n' "$UIIS_EPOCHS"
  printf 'check_only\t%s\n' "$CHECK_ONLY"
} > "$PROVENANCE"

exec > >(tee -a "$MASTER_LOG") 2>&1
echo "SFUI -> UIIS matrix pipeline"
echo "manifest=$MANIFEST"
echo "output_root=$OUTPUT_ROOT"

index=0
while IFS='|' read -r name backbone checkpoint sfui_root train_ann val_ann train_prefix val_prefix test_ann test_prefix sfui_config uiis_config expected_classes; do
  [ -z "$name" ] && continue
  [[ "$name" == \#* ]] && continue
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid experiment name: $name"
  [[ "$backbone" == r50 || "$backbone" == vits ]] || die "backbone must be r50 or vits: $backbone"

  checkpoint="${checkpoint:-}"
  sfui_root="${sfui_root:-}"
  train_ann="${train_ann:-$sfui_root/annotations/instances_train.json}"
  val_ann="${val_ann:-$sfui_root/annotations/instances_val.json}"
  train_prefix="${train_prefix:-images/}"
  val_prefix="${val_prefix:-images/}"
  test_ann="${test_ann:-$val_ann}"
  test_prefix="${test_prefix:-$val_prefix}"
  expected_classes="${expected_classes:-}"

  die_if_missing_file "$checkpoint"
  validate_sfui_data "$sfui_root" "$train_ann" "$val_ann" "$train_prefix" "$val_prefix"

  if [ -n "$expected_classes" ]; then
    actual_classes="$($PYTHON_BIN - "$train_ann" <<'PY'
import json, sys
print(len(json.loads(open(sys.argv[1]).read())['categories']))
PY
)"
    [ "$actual_classes" = "$expected_classes" ] || die "$name expected $expected_classes classes, got $actual_classes"
  fi

  if [ "$backbone" = r50 ]; then
    runner="$R50_RUNNER"
    gpu_group="$R50_GPUS"
    model_tag=r50
  else
    runner="$VITS_RUNNER"
    gpu_group="$VITS_GPUS"
    model_tag=vits
  fi

  output="$OUTPUT_ROOT/$name"
  log="$OUTPUT_ROOT/logs/${name}.log"
  port=$((BASE_PORT + index * 10))
  index=$((index + 1))

  if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$output/COMPLETE" ]; then
    echo "SKIP completed experiment: $name"
    continue
  fi

  check_port "$port"
  if [ "$CHECK_ONLY" != 1 ]; then
    wait_for_gpus "$gpu_group" "$name"
  fi
  mkdir -p "$output"

  echo "START name=$name backbone=$backbone gpu_group=$gpu_group"
  echo "checkpoint=$checkpoint"
  echo "sfui_root=$sfui_root"
  echo "output=$output"

  env_args=(
    "OUTPUT_ROOT=$output"
    "MODEL_TAG=$model_tag"
    "DINO_RAW=$checkpoint"
    "SFUI_DATA_ROOT=$sfui_root"
    "SFUI_TRAIN_ANN=$train_ann"
    "SFUI_VAL_ANN=$val_ann"
    "SFUI_TEST_ANN=$test_ann"
    "SFUI_TRAIN_PREFIX=$train_prefix"
    "SFUI_VAL_PREFIX=$val_prefix"
    "SFUI_TEST_PREFIX=$test_prefix"
    "UIIS_ROOT=$UIIS_ROOT"
    "SFUI_EPOCHS=$SFUI_EPOCHS"
    "UIIS_EPOCHS=$UIIS_EPOCHS"
    "SFUI_GPUS=$gpu_group"
    "UIIS_GPUS=$gpu_group"
    "MIN_FREE_MIB=$MIN_FREE_MIB"
    "WAIT_FOR_GPU=$WAIT_FOR_GPU"
    "BASE_PORT=$port"
    "SFUI_RUN_TEST=0"
    "UIIS_RUN_TEST=$RUN_TEST"
    "MAX_KEEP_CKPTS=$MAX_KEEP_CKPTS"
    "CHECK_ONLY=$CHECK_ONLY"
  )
  [ -z "$sfui_config" ] || env_args+=("SFUI_CONFIG=$sfui_config")
  [ -z "$uiis_config" ] || env_args+=("UIIS_CONFIG=$uiis_config")

  env "${env_args[@]}" bash "$runner" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || die "experiment failed: $name exit=$status"

  [ "$CHECK_ONLY" = 1 ] || touch "$output/COMPLETE"
  echo "COMPLETE name=$name"
done < "$MANIFEST"

echo "ALL_SELECTED_EXPERIMENTS_COMPLETE"
