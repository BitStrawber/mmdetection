#!/usr/bin/env bash
# Reusable SFUI -> UIIS Mask R-CNN matrix runner with resumable stages.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

usage() {
  cat <<'EOF'
Usage:
  run_sfui_uiis_pipeline.sh --manifest MANIFEST.tsv \
    --uiis-root UIIS_COCO --output-root RUN_DIR [options]

  run_sfui_uiis_pipeline.sh --name NAME --arch r50|vits \
    --pretrained CHECKPOINT --sfui-root SFUI_COCO \
    --uiis-root UIIS_COCO --output-root RUN_DIR [options]

Options:
  --r50-gpus LIST            GPUs used by ResNet-50 rows (default: 6,7)
  --vits-gpus LIST           GPUs used by ViT-S rows (default: 6,7)
  --sfui-epochs N            SFUI training epochs (default: 48)
  --uiis-epochs N            UIIS training epochs (default: 24)
  --min-free-mib N           Required free memory per selected GPU (default: 8192)
  --base-port N              First distributed training port (default: 39800)
  --run-test {0,1}           Run final UIIS test (default: 1)
  --wait-for-gpu {0,1}       Wait for GPU capacity (default: 1)
  --max-keep-ckpts N         Retained checkpoints per stage (default: 5)
  --no-skip-completed        Rerun rows even when COMPLETE exists
  --allow-input-change       Reuse an output root with changed core inputs
  --check-only               Validate inputs without training

Single-experiment row options:
  --train-ann PATH           Default: SFUI_ROOT/annotations/instances_train.json
  --val-ann PATH             Default: SFUI_ROOT/annotations/instances_val.json
  --test-ann PATH            Default: validation annotation
  --train-prefix PREFIX      Default: images/train/
  --val-prefix PREFIX        Default: images/val/
  --test-prefix PREFIX       Default: validation prefix
  --sfui-config PATH         Optional SFUI base config
  --uiis-config PATH         Optional UIIS base config
  --expected-classes N       Optional SFUI category-count assertion

Manifest columns:
  name|backbone|pretrain_checkpoint|sfui_root|train_ann|val_ann|
  train_prefix|val_prefix|test_ann|test_prefix|sfui_config|
  uiis_config|expected_sfui_classes

Interrupted runs resume from latest.pth. Rerun the exact same command and
OUTPUT_ROOT. Completed manifest rows are skipped when their COMPLETE marker
exists.
EOF
}

require_file() { [ -s "$1" ] || die "missing or empty file: $1"; }
require_dir() { [ -d "$1" ] || die "missing directory: $1"; }
absolute_path() { readlink -f -- "$1" || die "cannot resolve path: $1"; }

pick_repo_file() {
  local relative="$1"
  local path
  for path in "$REPO_ROOT/$relative" "$REPO_ROOT/.publish_repo/$relative"; do
    [ -s "$path" ] && { printf '%s\n' "$path"; return 0; }
  done
  return 1
}

MANIFEST=""
NAME=""
ARCH=""
PRETRAINED=""
SFUI_ROOT=""
UIIS_ROOT=""
OUTPUT_ROOT=""
TRAIN_ANN=""
VAL_ANN=""
TEST_ANN=""
TRAIN_PREFIX=""
VAL_PREFIX=""
TEST_PREFIX=""
SFUI_CONFIG=""
UIIS_CONFIG=""
EXPECTED_CLASSES=""
R50_GPUS="6,7"
VITS_GPUS="6,7"
SFUI_EPOCHS=48
UIIS_EPOCHS=24
MIN_FREE_MIB=8192
BASE_PORT=39800
RUN_TEST=1
WAIT_FOR_GPU=1
MAX_KEEP_CKPTS=5
SKIP_COMPLETED=1
ALLOW_INPUT_CHANGE=0
CHECK_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --arch) ARCH="${2:-}"; shift 2 ;;
    --pretrained) PRETRAINED="${2:-}"; shift 2 ;;
    --sfui-root) SFUI_ROOT="${2:-}"; shift 2 ;;
    --uiis-root) UIIS_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --train-ann) TRAIN_ANN="${2:-}"; shift 2 ;;
    --val-ann) VAL_ANN="${2:-}"; shift 2 ;;
    --test-ann) TEST_ANN="${2:-}"; shift 2 ;;
    --train-prefix) TRAIN_PREFIX="${2:-}"; shift 2 ;;
    --val-prefix) VAL_PREFIX="${2:-}"; shift 2 ;;
    --test-prefix) TEST_PREFIX="${2:-}"; shift 2 ;;
    --sfui-config) SFUI_CONFIG="${2:-}"; shift 2 ;;
    --uiis-config) UIIS_CONFIG="${2:-}"; shift 2 ;;
    --expected-classes) EXPECTED_CLASSES="${2:-}"; shift 2 ;;
    --r50-gpus) R50_GPUS="${2:-}"; shift 2 ;;
    --vits-gpus) VITS_GPUS="${2:-}"; shift 2 ;;
    --sfui-epochs) SFUI_EPOCHS="${2:-}"; shift 2 ;;
    --uiis-epochs) UIIS_EPOCHS="${2:-}"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="${2:-}"; shift 2 ;;
    --base-port) BASE_PORT="${2:-}"; shift 2 ;;
    --run-test) RUN_TEST="${2:-}"; shift 2 ;;
    --wait-for-gpu) WAIT_FOR_GPU="${2:-}"; shift 2 ;;
    --max-keep-ckpts) MAX_KEEP_CKPTS="${2:-}"; shift 2 ;;
    --no-skip-completed) SKIP_COMPLETED=0; shift ;;
    --allow-input-change) ALLOW_INPUT_CHANGE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [ -n "$MANIFEST" ]; then
  [ -z "$NAME$ARCH$PRETRAINED$SFUI_ROOT" ] ||
    die "use either --manifest or single-experiment options, not both"
  require_file "$MANIFEST"
  MANIFEST="$(absolute_path "$MANIFEST")"
else
  [ -n "$NAME" ] || die "--manifest or --name is required"
  [ -n "$ARCH" ] || die "--arch is required for a single experiment"
  [ -n "$PRETRAINED" ] || die "--pretrained is required for a single experiment"
  [ -n "$SFUI_ROOT" ] || die "--sfui-root is required for a single experiment"
  [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid experiment name"
  [[ "$ARCH" == r50 || "$ARCH" == vits ]] || die "--arch must be r50 or vits"
  require_file "$PRETRAINED"
  require_dir "$SFUI_ROOT"
  PRETRAINED="$(absolute_path "$PRETRAINED")"
  SFUI_ROOT="$(absolute_path "$SFUI_ROOT")"
  [ -z "$TRAIN_ANN" ] || TRAIN_ANN="$(absolute_path "$TRAIN_ANN")"
  [ -z "$VAL_ANN" ] || VAL_ANN="$(absolute_path "$VAL_ANN")"
  [ -z "$TEST_ANN" ] || TEST_ANN="$(absolute_path "$TEST_ANN")"
  [ -z "$SFUI_CONFIG" ] || SFUI_CONFIG="$(absolute_path "$SFUI_CONFIG")"
  [ -z "$UIIS_CONFIG" ] || UIIS_CONFIG="$(absolute_path "$UIIS_CONFIG")"
fi

[ -n "$UIIS_ROOT" ] || die "--uiis-root is required"
require_dir "$UIIS_ROOT"
UIIS_ROOT="$(absolute_path "$UIIS_ROOT")"
require_file "$UIIS_ROOT/annotations/instances_train.json"
require_file "$UIIS_ROOT/annotations/instances_val.json"
[ -d "$UIIS_ROOT/train" ] || die "missing UIIS train image directory"
[ -d "$UIIS_ROOT/val" ] || die "missing UIIS val image directory"
[[ "$R50_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --r50-gpus"
[[ "$VITS_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid --vits-gpus"
[[ "$RUN_TEST" =~ ^[01]$ ]] || die "--run-test must be 0 or 1"
[[ "$WAIT_FOR_GPU" =~ ^[01]$ ]] || die "--wait-for-gpu must be 0 or 1"
[ "$SFUI_EPOCHS" -gt 0 ] || die "--sfui-epochs must be positive"
[ "$UIIS_EPOCHS" -gt 0 ] || die "--uiis-epochs must be positive"

MATRIX_RUNNER="$(pick_repo_file scripts/exp_2/tri_pretrain/run_sfui_uiis_matrix.sh || true)"
[ -n "$MATRIX_RUNNER" ] || die "SFUI/UIIS matrix runner not found"

if [ -z "$OUTPUT_ROOT" ]; then
  if [ -n "$NAME" ]; then
    OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_sfui_uiis/$NAME"
  else
    OUTPUT_ROOT="$REPO_ROOT/work_dirs/auto_sfui_uiis/matrix_$(date +%Y%m%d_%H%M%S)"
  fi
fi

STATE_DIR="$OUTPUT_ROOT/.state"
mkdir -p "$OUTPUT_ROOT" "$STATE_DIR"
OUTPUT_ROOT="$(absolute_path "$OUTPUT_ROOT")"
STATE_DIR="$OUTPUT_ROOT/.state"

if [ -z "$MANIFEST" ]; then
  MANIFEST="$OUTPUT_ROOT/manifest.tsv"
  {
    printf '%s\n' '# name|backbone|pretrain_checkpoint|sfui_root|train_ann|val_ann|train_prefix|val_prefix|test_ann|test_prefix|sfui_config|uiis_config|expected_sfui_classes'
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
      "$NAME" "$ARCH" "$PRETRAINED" "$SFUI_ROOT" "$TRAIN_ANN" "$VAL_ANN" \
      "$TRAIN_PREFIX" "$VAL_PREFIX" "$TEST_ANN" "$TEST_PREFIX" \
      "$SFUI_CONFIG" "$UIIS_CONFIG" "$EXPECTED_CLASSES"
  } > "$MANIFEST"
fi

INPUT_FINGERPRINT="$(
  {
    printf '%s\n' "$UIIS_ROOT|$R50_GPUS|$VITS_GPUS|$SFUI_EPOCHS|$UIIS_EPOCHS|$RUN_TEST"
    sha256sum "$MANIFEST"
  } | sha256sum | cut -d' ' -f1
)"

if [ -s "$STATE_DIR/input_fingerprint" ]; then
  existing="$(cat "$STATE_DIR/input_fingerprint")"
  if [ "$existing" != "$INPUT_FINGERPRINT" ] && [ "$ALLOW_INPUT_CHANGE" != "1" ]; then
    die "core inputs changed for existing output root; use a new directory or --allow-input-change"
  fi
else
  printf '%s\n' "$INPUT_FINGERPRINT" > "$STATE_DIR/input_fingerprint"
fi

log "SFUI -> UIIS matrix manifest=$MANIFEST output=$OUTPUT_ROOT"
env \
  MANIFEST="$MANIFEST" \
  UIIS_ROOT="$UIIS_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  R50_GPUS="$R50_GPUS" \
  VITS_GPUS="$VITS_GPUS" \
  MIN_FREE_MIB="$MIN_FREE_MIB" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" \
  BASE_PORT="$BASE_PORT" \
  SFUI_EPOCHS="$SFUI_EPOCHS" \
  UIIS_EPOCHS="$UIIS_EPOCHS" \
  RUN_TEST="$RUN_TEST" \
  CHECK_ONLY="$CHECK_ONLY" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$MATRIX_RUNNER"

echo "RESULT=PASS_SFUI_UIIS_PIPELINE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
