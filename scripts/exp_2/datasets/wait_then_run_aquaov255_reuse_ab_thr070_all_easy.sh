#!/usr/bin/env bash
# Wait for the current full-data 0.65 all_easy run, then re-filter at 0.70
# using the same completed 0.60 A/B Mask R-CNN teachers.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CURRENT_ROOT="${CURRENT_ROOT:?Set CURRENT_ROOT to the active 0.65 run directory}"
CURRENT_PID="${CURRENT_PID:?Set CURRENT_PID to the active 0.65 pipeline PID}"
AQUA_ROOT="${AQUA_ROOT:-/media/HDD2/XCX/exp_2/AquaOV255/raw/AquaOV255}"
OLD_ROOT="${OLD_ROOT:-/media/HDD2/XCX/exp_2/aquaov255_maskrcnn_runs/aquaov255_fullab_maskrcnn_lr005_cascade_all_then_split10_20261009_002325}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh 0.70 run directory}"
GPU_GROUP="${GPU_GROUP:-2,3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CURRENT_CASCADE_EPOCHS="${CURRENT_CASCADE_EPOCHS:-24}"
CASCADE_EPOCHS="${CASCADE_EPOCHS:-24}"
CASCADE_LR="${CASCADE_LR:-0.005}"
RUN_TEST="${RUN_TEST:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }
proc_running() {
  local state
  state="$(ps -p "$1" -o stat= 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$state" && "$state" != Z* ]]
}

[[ "$CURRENT_PID" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[0-9]+$ ]] ||
  die "CURRENT_PID and POLL_SECONDS must be integers"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists: $OUTPUT_ROOT"
[[ -d "$CURRENT_ROOT" ]] || die "0.65 run directory not found: $CURRENT_ROOT"

COMPLETE_MARKER="$CURRENT_ROOT/work_dirs/cascade_mask_rcnn_aquaov255_easy_all_easy_${CURRENT_CASCADE_EPOCHS}e/.complete"
if proc_running "$CURRENT_PID"; then
  CURRENT_CMD="$(ps -p "$CURRENT_PID" -o args= 2>/dev/null || true)"
  [[ "$CURRENT_CMD" == *run_aquaov255_ab_maskrcnn_easy_cascade.sh* ||
     "$CURRENT_CMD" == *run_aquaov255_reuse_ab_thr065_all_easy.sh* ]] ||
    die "CURRENT_PID does not look like the AquaOV255 pipeline: $CURRENT_CMD"
fi

echo "Waiting for 0.65 all_easy completion marker: $COMPLETE_MARKER"
while [[ ! -f "$COMPLETE_MARKER" ]]; do
  proc_running "$CURRENT_PID" || die "0.65 pipeline exited before all_easy completed; 0.70 run was not started"
  sleep "$POLL_SECONDS"
done

while proc_running "$CURRENT_PID"; do
  echo "Completion marker exists; waiting for the 0.65 launcher to exit."
  sleep 2
done

[[ -s "$OLD_ROOT/work_dirs/maskrcnn_train_A_val_B/best_coco_bbox_mAP_epoch_23.pth" ]] ||
  die "Reusable A->B teacher checkpoint is missing"
[[ -s "$OLD_ROOT/work_dirs/maskrcnn_train_B_val_A/best_coco_bbox_mAP_epoch_23.pth" ]] ||
  die "Reusable B->A teacher checkpoint is missing"

echo "0.65 run completed; starting full-data 0.70 filtering and all_easy Cascade."
exec env \
  OLD_ROOT="$OLD_ROOT" \
  AQUA_ROOT="$AQUA_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  GPU_GROUP="$GPU_GROUP" \
  SCORE_THRESHOLD=0.70 \
  CASCADE_EPOCHS="$CASCADE_EPOCHS" \
  CASCADE_LR="$CASCADE_LR" \
  RUN_TEST="$RUN_TEST" \
  bash "$SCRIPT_DIR/run_aquaov255_reuse_ab_thr065_all_easy.sh"
