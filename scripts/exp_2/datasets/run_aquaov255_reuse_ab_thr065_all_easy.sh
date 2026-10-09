#!/usr/bin/env bash
# Re-filter AquaOV255 at a new threshold using completed A/B Mask R-CNN teachers.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

OLD_ROOT="${OLD_ROOT:-/media/HDD2/XCX/exp_2/aquaov255_maskrcnn_runs/aquaov255_fullab_maskrcnn_lr005_cascade_all_then_split10_20261009_002325}"
AQUA_ROOT="${AQUA_ROOT:-/media/HDD2/XCX/exp_2/AquaOV255/raw/AquaOV255}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new, unused run directory}"
GPU_GROUP="${GPU_GROUP:-2,3}"
BASE_PORT="${BASE_PORT:-38800}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.65}"
MASK_EPOCHS=24
CASCADE_EPOCHS="${CASCADE_EPOCHS:-24}"
MASK_LR="${MASK_LR:-0.005}"
CASCADE_LR="${CASCADE_LR:-0.005}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_TEST="${RUN_TEST:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

awk -v threshold="$SCORE_THRESHOLD" 'BEGIN {
  if (threshold !~ /^[0-9]+([.][0-9]+)?$/ || threshold < 0 || threshold > 1) exit 1
}' || die "SCORE_THRESHOLD must be a number in [0, 1]"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists; choose a fresh directory: $OUTPUT_ROOT"
[[ -d "$AQUA_ROOT/images" && -d "$AQUA_ROOT/masks" && -s "$AQUA_ROOT/category.txt" ]] ||
  die "AquaOV255 images, masks, or category.txt are missing under $AQUA_ROOT"

COCO_SOURCE="$OLD_ROOT/aquaov255_instance_coco"
AB_SOURCE="$OLD_ROOT/ab_split"
COCO_ANN="$COCO_SOURCE/train/annotations/instances_train.json"
[[ -f "$COCO_SOURCE/.complete" && -s "$COCO_ANN" ]] || die "Completed source COCO build not found"
[[ -f "$AB_SOURCE/.complete" && -s "$AB_SOURCE/train_A.json" && -s "$AB_SOURCE/train_B.json" ]] ||
  die "Completed source A/B split not found"

for stage in maskrcnn_train_A_val_B maskrcnn_train_B_val_A; do
  work="$OLD_ROOT/work_dirs/$stage"
  [[ -s "$work/epoch_${MASK_EPOCHS}.pth" ]] || die "$stage did not reach epoch $MASK_EPOCHS"
  [[ -s "$work/best_coco_bbox_mAP_epoch_23.pth" ]] || die "$stage best epoch-23 bbox AP checkpoint is missing"
  [[ -s "$work/runtime_config.py" ]] || die "$stage runtime config is missing"
done

python - "$COCO_SOURCE/summary.json" "$COCO_ANN" "$AB_SOURCE" "$AQUA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

summary_path, coco_path, ab_root, aqua_root = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected = {"images": 20722, "train_images": 20722, "val_images": 0, "source_validation": False}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"ERROR: source COCO {key}={summary.get(key)!r}; expected {value!r}")
expected_sources = {
    "source_images": str((aqua_root / "images").resolve()),
    "source_masks": str((aqua_root / "masks").resolve()),
    "official_category_file": str((aqua_root / "category.txt").resolve()),
}
for key, value in expected_sources.items():
    if summary.get(key) != value:
        raise SystemExit(f"ERROR: source COCO {key}={summary.get(key)!r}; expected {value!r}")

source = json.loads(coco_path.read_text(encoding="utf-8"))
parts = [json.loads((ab_root / f"train_{fold}.json").read_text(encoding="utf-8")) for fold in ("A", "B")]
source_ids = {int(image["id"]) for image in source["images"]}
fold_ids = [{int(image["id"]) for image in part["images"]} for part in parts]
if len(source_ids) != 20722 or len(fold_ids[0]) != 10361 or len(fold_ids[1]) != 10361:
    raise SystemExit("ERROR: source/A/B image counts do not match the expected full 20,722-image split")
if fold_ids[0] & fold_ids[1] or fold_ids[0] | fold_ids[1] != source_ids:
    raise SystemExit("ERROR: A/B split is not disjoint and exhaustive for the source COCO images")
if parts[0].get("categories") != source.get("categories") or parts[1].get("categories") != source.get("categories"):
    raise SystemExit("ERROR: A/B category definitions do not match the source COCO")
print("Verified source COCO and exact completed A/B split: 20,722 = 10,361 + 10,361; overlap=0")
PY

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/work_dirs"
COCO_ROOT="$OUTPUT_ROOT/aquaov255_instance_coco"
AB_ROOT="$OUTPUT_ROOT/ab_split"
ln -s "$COCO_SOURCE" "$COCO_ROOT"
ln -s "$AB_SOURCE" "$AB_ROOT"

link_or_copy() {
  local source="$1" target="$2"
  mkdir -p "$(dirname "$target")"
  if ! ln "$source" "$target" 2>/dev/null; then
    cp -p "$source" "$target"
  fi
}

for stage in maskrcnn_train_A_val_B maskrcnn_train_B_val_A; do
  source_work="$OLD_ROOT/work_dirs/$stage"
  target_work="$OUTPUT_ROOT/work_dirs/$stage"
  mkdir -p "$target_work"
  link_or_copy "$source_work/epoch_${MASK_EPOCHS}.pth" "$target_work/epoch_${MASK_EPOCHS}.pth"
  link_or_copy "$source_work/best_coco_bbox_mAP_epoch_23.pth" "$target_work/best_coco_bbox_mAP_epoch_23.pth"
  link_or_copy "$source_work/runtime_config.py" "$target_work/runtime_config.py"
  printf '%s\n' "$target_work/best_coco_bbox_mAP_epoch_23.pth" > "$target_work/best_checkpoint.txt"
  printf 'Reused from %s\n' "$source_work" > "$target_work/.complete"
done

printf 'field\tvalue\n' > "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'source_run\t%s\n' "$OLD_ROOT" >> "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'A_to_B_checkpoint\t%s\n' "$OLD_ROOT/work_dirs/maskrcnn_train_A_val_B/best_coco_bbox_mAP_epoch_23.pth" >> "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'B_to_A_checkpoint\t%s\n' "$OLD_ROOT/work_dirs/maskrcnn_train_B_val_A/best_coco_bbox_mAP_epoch_23.pth" >> "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'mask_training\tSKIPPED_REUSED_COMPLETED_24E_CHECKPOINTS\n' >> "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'filter_threshold\t%s\n' "$SCORE_THRESHOLD" >> "$OUTPUT_ROOT/reuse_provenance.tsv"
printf 'cascade_variants\tall_easy_only\n' >> "$OUTPUT_ROOT/reuse_provenance.tsv"

echo "Reusing completed Mask R-CNN teachers from: $OLD_ROOT"
echo "Re-filtering threshold=$SCORE_THRESHOLD; Cascade variant=all_easy only"
echo "New output root: $OUTPUT_ROOT"

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
  SKIP_COMPLETED=1 \
  RUN_TEST="$RUN_TEST" \
  CHECK_ONLY=0 \
  bash scripts/exp_2/datasets/run_aquaov255_ab_maskrcnn_easy_cascade.sh
