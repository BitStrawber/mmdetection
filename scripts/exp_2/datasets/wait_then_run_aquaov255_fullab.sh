#!/usr/bin/env bash
# Wait for a successful prior AquaOV255 run and sufficient free GPU memory,
# then start the full-source A/B cross-filter pipeline.
set -Eeuo pipefail

OLD_ROOT="${OLD_ROOT:?set OLD_ROOT to the completed/active old run directory}"
NEW_REPO="${NEW_REPO:?set NEW_REPO to a checkout containing the updated pipeline}"
AQUA_ROOT="${AQUA_ROOT:?set AQUA_ROOT to AquaOV255 root}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh output directory}"
OLD_LOG="${OLD_LOG:-$OLD_ROOT/logs/pipeline.log}"
GPU_GROUP="${GPU_GROUP:-2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OLD_CASCADE_EPOCHS="${OLD_CASCADE_EPOCHS:-24}"

MASK_EPOCHS="${MASK_EPOCHS:-24}"
MASK_LR="${MASK_LR:-0.005}"
CASCADE_EPOCHS="${CASCADE_EPOCHS:-24}"
CASCADE_LR="${CASCADE_LR:-0.005}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.60}"
EASY_VAL_RATIO="${EASY_VAL_RATIO:-0.10}"
IMAGENET_INIT="${IMAGENET_INIT:-torchvision://resnet50}"
BASE_PORT="${BASE_PORT:-38600}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$OLD_ROOT" ]] || die "old run directory not found: $OLD_ROOT"
[[ -f "$OLD_LOG" ]] || die "old pipeline log not found: $OLD_LOG"
[[ -f "$NEW_REPO/scripts/exp_2/datasets/run_aquaov255_ab_maskrcnn_easy_cascade.sh" ]] ||
  die "updated pipeline script not found under NEW_REPO=$NEW_REPO"
[[ -d "$AQUA_ROOT/images" && -d "$AQUA_ROOT/masks" && -s "$AQUA_ROOT/category.txt" ]] ||
  die "AquaOV255 images, masks, or category.txt are missing"
[[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || die "MIN_FREE_MIB must be an integer"
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || die "POLL_SECONDS must be an integer"

IFS=, read -r -a GPUS <<< "$GPU_GROUP"
[[ "${#GPUS[@]}" -gt 0 ]] || die "GPU_GROUP is empty"

echo "Waiting for old AquaOV255 pipeline: $OLD_ROOT"
while pgrep -af "[r]un_aquaov255_ab_maskrcnn_easy_cascade.sh" >/dev/null; do
  echo "$(date '+%F %T') old AquaOV255 pipeline process is still running"
  sleep "$POLL_SECONDS"
done

LAST_STATUS="$(grep -E '^(COMPLETE output_root=|ERROR:|FAILED:)' "$OLD_LOG" | tail -n 1 || true)"
echo "OLD_LAST_STATUS=$LAST_STATUS"
[[ "$LAST_STATUS" == "COMPLETE output_root=$OLD_ROOT" ]] ||
  die "old pipeline did not finish successfully; refusing to start the new run"

for variant in all_easy split10; do
  marker="$OLD_ROOT/work_dirs/cascade_mask_rcnn_aquaov255_easy_${variant}_${OLD_CASCADE_EPOCHS}e/.complete"
  [[ -f "$marker" ]] || die "old Cascade completion marker is missing: $marker"
done

echo "Old pipeline completed successfully. Waiting for free-memory threshold."
while true; do
  READY=1
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
      echo "$(date '+%F %T') GPU=$gpu free-memory query failed; will retry"
      READY=0
    elif (( free_mib < MIN_FREE_MIB )); then
      echo "$(date '+%F %T') GPU=$gpu free=${free_mib}MiB; need >=${MIN_FREE_MIB}MiB"
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
echo "Starting updated full-source AquaOV255 pipeline: $OUTPUT_ROOT"
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
  EASY_VAL_RATIO="$EASY_VAL_RATIO" \
  IMAGENET_INIT="$IMAGENET_INIT" \
  SKIP_COMPLETED=0 \
  RUN_TEST=1 \
  CHECK_ONLY=0 \
  bash scripts/exp_2/datasets/run_aquaov255_ab_maskrcnn_easy_cascade.sh
