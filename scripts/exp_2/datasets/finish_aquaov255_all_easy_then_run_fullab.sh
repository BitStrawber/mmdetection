#!/usr/bin/env bash
# Stop the prior pipeline after its all_easy Cascade stage, then launch a fresh
# full A/B cross-filter run once the requested GPUs have enough free memory.
set -Eeuo pipefail

CURRENT_ROOT="${CURRENT_ROOT:?set CURRENT_ROOT to the currently running run directory}"
CURRENT_PID="${CURRENT_PID:?set CURRENT_PID to the current pipeline bash PID}"
NEW_REPO="${NEW_REPO:?set NEW_REPO to a checkout containing the updated pipeline}"
AQUA_ROOT="${AQUA_ROOT:?set AQUA_ROOT to AquaOV255 root}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh output directory}"
GPU_GROUP="${GPU_GROUP:-2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OLD_CASCADE_EPOCHS="${OLD_CASCADE_EPOCHS:-24}"
MASK_EPOCHS="${MASK_EPOCHS:-24}"
MASK_LR="${MASK_LR:-0.005}"
CASCADE_EPOCHS="${CASCADE_EPOCHS:-24}"
CASCADE_LR="${CASCADE_LR:-0.005}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.65}"
IMAGENET_INIT="${IMAGENET_INIT:-torchvision://resnet50}"
BASE_PORT="${BASE_PORT:-38700}"

die() { echo "ERROR: $*" >&2; exit 1; }
proc_state() { ps -p "$1" -o stat= 2>/dev/null | tr -d '[:space:]'; }
proc_running() {
  local state
  state="$(proc_state "$1")"
  [[ -n "$state" && "$state" != Z* ]]
}

MARKER="$CURRENT_ROOT/work_dirs/cascade_mask_rcnn_aquaov255_easy_all_easy_${OLD_CASCADE_EPOCHS}e/.complete"
[[ -d "$CURRENT_ROOT" ]] || die "current run directory not found: $CURRENT_ROOT"
[[ -f "$NEW_REPO/scripts/exp_2/datasets/run_aquaov255_ab_maskrcnn_easy_cascade.sh" ]] ||
  die "updated pipeline script not found under NEW_REPO=$NEW_REPO"
[[ -d "$AQUA_ROOT/images" && -d "$AQUA_ROOT/masks" && -s "$AQUA_ROOT/category.txt" ]] ||
  die "AquaOV255 images, masks, or category.txt are missing"
[[ "$CURRENT_PID" =~ ^[0-9]+$ && "$MIN_FREE_MIB" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[0-9]+$ ]] ||
  die "CURRENT_PID, MIN_FREE_MIB, and POLL_SECONDS must be integers"

if proc_running "$CURRENT_PID"; then
  CURRENT_CMD="$(ps -p "$CURRENT_PID" -o args= 2>/dev/null || true)"
  [[ "$CURRENT_CMD" == *run_aquaov255_ab_maskrcnn_easy_cascade.sh* ]] ||
    die "CURRENT_PID=$CURRENT_PID is not the expected AquaOV255 pipeline: $CURRENT_CMD"
fi

echo "Waiting for current all_easy Cascade completion: $MARKER"
while [[ ! -f "$MARKER" ]]; do
  proc_running "$CURRENT_PID" || die "current pipeline exited before all_easy Cascade completed"
  sleep 0.1
done

echo "Current all_easy Cascade is complete; stopping the old pipeline before split10."
if proc_running "$CURRENT_PID"; then
  CURRENT_CMD="$(ps -p "$CURRENT_PID" -o args= 2>/dev/null || true)"
  [[ "$CURRENT_CMD" == *run_aquaov255_ab_maskrcnn_easy_cascade.sh* ]] ||
    die "refusing to stop PID whose command changed: $CURRENT_CMD"
  kill -TERM "$CURRENT_PID"
fi
while proc_running "$CURRENT_PID"; do sleep 1; done

SPLIT10_LOG="$CURRENT_ROOT/logs/cascade_mask_rcnn_split10_train.log"
if [[ -s "$SPLIT10_LOG" ]] && grep -q 'Epoch(train)' "$SPLIT10_LOG"; then
  echo "WARNING: split10 training emitted log records before the old launcher stopped."
fi

IFS=, read -r -a GPUS <<< "$GPU_GROUP"
[[ "${#GPUS[@]}" -gt 0 ]] || die "GPU_GROUP is empty"
echo "Waiting for each GPU in $GPU_GROUP to have at least ${MIN_FREE_MIB}MiB free."
while true; do
  READY=1
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
      echo "$(date '+%F %T') GPU=$gpu free-memory query failed; retrying"
      READY=0
    elif (( free_mib < MIN_FREE_MIB )); then
      echo "$(date '+%F %T') GPU=$gpu free=${free_mib}MiB; waiting"
      READY=0
    else
      echo "$(date '+%F %T') GPU=$gpu free=${free_mib}MiB; threshold met"
    fi
  done
  [[ "$READY" -eq 1 ]] && break
  sleep "$POLL_SECONDS"
done

mkdir -p "$OUTPUT_ROOT/logs"
echo "GPU_MEMORY_GATE=PASS group=$GPU_GROUP minimum_free_mib=$MIN_FREE_MIB"
echo "Starting new threshold=$SCORE_THRESHOLD run with Cascade variant=all_easy only."
cd "$NEW_REPO"
exec env \
  AQUA_ROOT="$AQUA_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  GPU_GROUP="$GPU_GROUP" \
  BASE_PORT="$BASE_PORT" \
  MASK_EPOCHS="$MASK_EPOCHS" \
  MASK_LR="$MASK_LR" \
  CASCADE_EPOCHS="$CASCADE_EPOCHS" \
  CASCADE_LR="$CASCADE_LR" \
  SCORE_THRESHOLD="$SCORE_THRESHOLD" \
  CASCADE_VARIANTS=all_easy \
  IMAGENET_INIT="$IMAGENET_INIT" \
  SKIP_COMPLETED=0 \
  RUN_TEST=1 \
  CHECK_ONLY=0 \
  bash scripts/exp_2/datasets/run_aquaov255_ab_maskrcnn_easy_cascade.sh
