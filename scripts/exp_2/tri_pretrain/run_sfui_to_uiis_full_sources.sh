#!/usr/bin/env bash
# Run full-pretrained DINO sources through an SFUI detection stage, then
# transfer each best backbone to the existing UIIS10K Mask R-CNN stage.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SFUI_NAME="${SFUI_NAME:?set SFUI_NAME, e.g. dfui_realuw}"
SFUI_ROOT="${SFUI_ROOT:?set SFUI_ROOT to the SFUI COCO dataset root}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SFUI_SPLIT_ROOT="${SFUI_SPLIT_ROOT:-}"
if [ -n "$SFUI_SPLIT_ROOT" ]; then
  SFUI_TRAIN_ANN="${SFUI_TRAIN_ANN:-$SFUI_SPLIT_ROOT/annotations/instances_train.json}"
  SFUI_VAL_ANN="${SFUI_VAL_ANN:-$SFUI_SPLIT_ROOT/annotations/instances_val.json}"
else
  SFUI_TRAIN_ANN="${SFUI_TRAIN_ANN:-$SFUI_ROOT/annotations/instances_train.json}"
  SFUI_VAL_ANN="${SFUI_VAL_ANN:-}"
fi
SFUI_TEST_ANN="${SFUI_TEST_ANN:-${SFUI_VAL_ANN:-$SFUI_TRAIN_ANN}}"
# Most reusable SFUI datasets store images under images/. AquaOV255's
# historical instance layout is train/images/; pass those prefixes explicitly.
SFUI_TRAIN_PREFIX="${SFUI_TRAIN_PREFIX:-images/}"
SFUI_VAL_PREFIX="${SFUI_VAL_PREFIX:-images/}"
SFUI_TEST_PREFIX="${SFUI_TEST_PREFIX:-images/}"

DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
UIIS_ROOT="${UIIS_ROOT:-$DATA_ROOT/UIIS10K/coco}"
RUOD_ROOT="${RUOD_ROOT:-$DATA_ROOT/RUOD/coco}"
SFUI_CLASS_NAMES_FILE="${SFUI_CLASS_NAMES_FILE:-}"
SFUI_EXPECTED_CLASSES="${SFUI_EXPECTED_CLASSES:-}"
IMAGENET_R50_RAW="${IMAGENET_R50_RAW:?set full ImageNet DINO ResNet-50 checkpoint}"
IMAGENET_VITS_RAW="${IMAGENET_VITS_RAW:?set full ImageNet DINO ViT-S checkpoint}"
REALUW_R50_RAW="${REALUW_R50_RAW:?set full RealUW DINO ResNet-50 checkpoint}"
REALUW_VITS_RAW="${REALUW_VITS_RAW:?set full RealUW DINO ViT-S checkpoint}"
SYNTHETIC5_R50_RAW="${SYNTHETIC5_R50_RAW:?set full Synthetic5 DINO ResNet-50 checkpoint}"
SYNTHETIC5_VITS_RAW="${SYNTHETIC5_VITS_RAW:?set full Synthetic5 DINO ViT-S checkpoint}"

R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-4,5}"
PARALLEL_ARCHITECTURES="${PARALLEL_ARCHITECTURES:-0}"
BASE_PORT="${BASE_PORT:-30600}"
DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECK_ONLY="${CHECK_ONLY:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-0}"

R50_RUNNER="$REPO_ROOT/scripts/exp_2/datasets/run_sfui_to_uiis_r50_mask.sh"
VITS_RUNNER="$REPO_ROOT/scripts/exp_2/datasets/run_sfui_to_uiis_vits_mask.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$SFUI_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "SFUI_NAME may contain only letters, digits, dot, underscore, and hyphen"
[[ "$CHECK_ONLY" =~ ^[01]$ ]] || die "CHECK_ONLY must be 0 or 1"
[[ "$RUN_TEST" =~ ^[01]$ ]] || die "RUN_TEST must be 0 or 1"
[[ "$PARALLEL_ARCHITECTURES" =~ ^[01]$ ]] || die "PARALLEL_ARCHITECTURES must be 0 or 1"
[[ "$DFUI_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "DFUI_EPOCHS must be a positive integer"
[[ "$UIIS_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "UIIS_EPOCHS must be a positive integer"
[[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || die "MIN_FREE_MIB must be a non-negative integer"
[[ -z "$SFUI_EXPECTED_CLASSES" || "$SFUI_EXPECTED_CLASSES" =~ ^[1-9][0-9]*$ ]] || die "SFUI_EXPECTED_CLASSES must be a positive integer"

for file in \
  "$R50_RUNNER" \
  "$VITS_RUNNER" \
  tools/exp_2/prepare_sfui_uiis_mask_configs.py \
  tools/convert_ssl_backbone_to_mmdet.py; do
  [ -s "$file" ] || die "required repository file is missing: $file"
done

for root in "$UIIS_ROOT"; do
  [ -s "$root/annotations/instances_train.json" ] &&
  [ -s "$root/annotations/instances_val.json" ] || die "missing UIIS annotations: $root"
done

python - "$SFUI_ROOT" "$SFUI_CLASS_NAMES_FILE" "$SFUI_EXPECTED_CLASSES" "$UIIS_ROOT" \
  "$SFUI_TRAIN_ANN" "$SFUI_VAL_ANN" \
  "$SFUI_TRAIN_PREFIX" "$SFUI_VAL_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

sfui_root = Path(sys.argv[1])
class_file = Path(sys.argv[2]) if sys.argv[2] else None
expected_classes = int(sys.argv[3]) if sys.argv[3] else None
uiis_root = Path(sys.argv[4])
sfui_train_ann = Path(sys.argv[5])
sfui_val_ann = Path(sys.argv[6]) if sys.argv[6] else None
sfui_train_prefix = sys.argv[7]
sfui_val_prefix = sys.argv[8]

def load_split(root, split, prefix, ann_path=None):
    ann_path = Path(ann_path) if ann_path else root / "annotations" / f"instances_{split}.json"
    if not ann_path.is_file():
        raise SystemExit(f"MISSING annotation: {ann_path}")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    if not categories:
        raise SystemExit(f"NO categories: {ann_path}")
    ids = [item["id"] for item in categories]
    names = [item["name"] for item in categories]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise SystemExit(f"duplicate category ids or names: {ann_path}")
    known = set(ids)
    unknown = sorted({ann.get("category_id") for ann in data.get("annotations", [])} - known)
    if unknown:
        raise SystemExit(f"annotations reference undefined categories {unknown[:10]}: {ann_path}")
    image_ids = [image["id"] for image in data.get("images", [])]
    if len(image_ids) != len(set(image_ids)):
        raise SystemExit(f"duplicate image ids: {ann_path}")
    image_root = root / prefix
    missing = [image["file_name"] for image in data.get("images", [])
               if not (image_root / image["file_name"]).is_file()]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} image files missing under {image_root}; e.g. {missing[:3]}")
    return data, [(item["id"], item["name"]) for item in categories]

train, train_categories = load_split(sfui_root, "train", sfui_train_prefix, sfui_train_ann)
val = None
val_categories = train_categories
if sfui_val_ann is not None:
    val, val_categories = load_split(sfui_root, "val", sfui_val_prefix, sfui_val_ann)
if val is not None and train_categories != val_categories:
    raise SystemExit("SFUI train/val category IDs and names differ")
train_names = {image["file_name"] for image in train["images"]}
if val is not None:
    val_names = {image["file_name"] for image in val["images"]}
    if train_names & val_names:
        raise SystemExit(f"SFUI train/val filename overlap: {len(train_names & val_names)}")

if class_file is not None:
    names_data = json.loads(class_file.read_text(encoding="utf-8"))
    if isinstance(names_data, dict):
        names_data = names_data.get("classes")
    if not isinstance(names_data, list) or not all(isinstance(name, str) and name for name in names_data):
        raise SystemExit("SFUI_CLASS_NAMES_FILE must contain a JSON string array or {\"classes\": [...]}")
    if len(names_data) != len(train_categories) or len(set(names_data)) != len(names_data):
        raise SystemExit("class-name override must be unique and match the COCO category count/order")
    classes = names_data
else:
    classes = [name for _, name in train_categories]
if expected_classes is not None and len(classes) != expected_classes:
    raise SystemExit(
        f"SFUI category count mismatch: expected {expected_classes}, got {len(classes)}"
    )

uiis_counts = {}
for split in ("train", "val"):
    data, _ = load_split(uiis_root, split, split)
    uiis_counts[split] = len(data["images"])

print(f"SFUI_ROOT={sfui_root}")
print(f"SFUI_train_images={len(train['images'])} SFUI_val_images={len(val['images']) if val is not None else 0}")
print(f"SFUI_train_annotations={len(train.get('annotations', []))} SFUI_val_annotations={len(val.get('annotations', [])) if val is not None else 0}")
print(f"SFUI_categories={len(classes)} class_names={classes}")
print(f"UIIS_train_images={uiis_counts['train']} UIIS_val_images={uiis_counts['val']}")
print("RESULT=PASS_SFUI_AND_UIIS_DATA_PREFLIGHT")
PY

declare -A RAW_R50=(
  [imagenet]="$IMAGENET_R50_RAW"
  [realuw]="$REALUW_R50_RAW"
  [synthetic5]="$SYNTHETIC5_R50_RAW"
)
declare -A RAW_VITS=(
  [imagenet]="$IMAGENET_VITS_RAW"
  [realuw]="$REALUW_VITS_RAW"
  [synthetic5]="$SYNTHETIC5_VITS_RAW"
)

for source in imagenet realuw synthetic5; do
  for spec in "${RAW_R50[$source]}:resnet50" "${RAW_VITS[$source]}:vit_small"; do
    checkpoint="${spec%:*}"
    arch="${spec##*:}"
    [ -s "$checkpoint" ] || die "missing $source $arch full DINO checkpoint: $checkpoint"
    python - "$checkpoint" "$arch" <<'PY'
import sys
import torch

path, expected_arch = sys.argv[1:]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
args = checkpoint.get("args")
teacher = checkpoint.get("teacher")
actual_arch = getattr(args, "arch", None)
print(f"checkpoint={path} epoch={checkpoint.get('epoch')} arch={actual_arch}")
if checkpoint.get("epoch") != 100 or actual_arch != expected_arch or not isinstance(teacher, dict) or not teacher:
    raise SystemExit("checkpoint is not a complete matching 100-epoch DINO checkpoint")
PY
  done
done

if [ -n "$SFUI_CLASS_NAMES_FILE" ]; then
  [ -s "$SFUI_CLASS_NAMES_FILE" ] || die "class names file not found: $SFUI_CLASS_NAMES_FILE"
fi

if [ "$MIN_FREE_MIB" -gt 0 ]; then
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
  awk -F', *' -v r50="$R50_GPUS" -v vits="$VITS_GPUS" -v min="$MIN_FREE_MIB" '
    BEGIN { n=split(r50,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1; n=split(vits,b,","); for(i=1;i<=n;i++) wanted[b[i]]=1 }
    ($1 in wanted) { print "GPU=" $1 " free_MiB=" $2; if ($2 < min) failed=1; seen[$1]=1 }
    END { for (gpu in wanted) if (!seen[gpu]) failed=1; exit failed }
  ' || die "selected GPUs do not meet MIN_FREE_MIB=$MIN_FREE_MIB"
fi

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs"
PROVENANCE="$OUTPUT_ROOT/run_provenance.tsv"
{
  printf 'field\tvalue\n'
  printf 'sfui_name\t%s\n' "$SFUI_NAME"
  printf 'sfui_root\t%s\n' "$SFUI_ROOT"
  printf 'sfui_train_ann\t%s\n' "$SFUI_TRAIN_ANN"
  printf 'sfui_val_ann\t%s\n' "${SFUI_VAL_ANN:-DISABLED}"
  printf 'sfui_test_ann\t%s\n' "${SFUI_TEST_ANN:-DISABLED}"
  printf 'sfui_split_root\t%s\n' "${SFUI_SPLIT_ROOT:-ANNOTATION_PATHS}"
  printf 'sfui_train_prefix\t%s\n' "$SFUI_TRAIN_PREFIX"
  printf 'sfui_val_prefix\t%s\n' "$SFUI_VAL_PREFIX"
  printf 'sfui_test_prefix\t%s\n' "$SFUI_TEST_PREFIX"
  printf 'sfui_expected_classes\t%s\n' "${SFUI_EXPECTED_CLASSES:-AUTO}"
  printf 'sfui_class_names_file\t%s\n' "${SFUI_CLASS_NAMES_FILE:-ANNOTATION_CATEGORIES}"
  printf 'uiis_root\t%s\n' "$UIIS_ROOT"
  printf 'r50_gpus\t%s\n' "$R50_GPUS"
  printf 'vits_gpus\t%s\n' "$VITS_GPUS"
  printf 'parallel_architectures\t%s\n' "$PARALLEL_ARCHITECTURES"
  printf 'dfui_epochs\t%s\n' "$DFUI_EPOCHS"
  printf 'uiis_epochs\t%s\n' "$UIIS_EPOCHS"
  printf 'check_only\t%s\n' "$CHECK_ONLY"
  for source in imagenet realuw synthetic5; do
    printf '%s_r50_raw\t%s\n' "$source" "${RAW_R50[$source]}"
    printf '%s_r50_sha256\t%s\n' "$source" "$(sha256sum "${RAW_R50[$source]}" | awk '{print $1}')"
    printf '%s_vits_raw\t%s\n' "$source" "${RAW_VITS[$source]}"
    printf '%s_vits_sha256\t%s\n' "$source" "$(sha256sum "${RAW_VITS[$source]}" | awk '{print $1}')"
  done
} > "$PROVENANCE"

echo "Original SFUI -> UIIS Mask R-CNN full-pretraining comparison"
echo "sfui_name=$SFUI_NAME sfui_root=$SFUI_ROOT"
echo "three sources x two backbones; SFUI Mask R-CNN=${DFUI_EPOCHS}e -> UIIS Mask R-CNN=${UIIS_EPOCHS}e"
echo "R50_GPUS=$R50_GPUS VITS_GPUS=$VITS_GPUS parallel_architectures=$PARALLEL_ARCHITECTURES check_only=$CHECK_ONLY"
echo "output_root=$OUTPUT_ROOT"

source_index=0
for source in imagenet realuw synthetic5; do
  for backbone in r50 vits; do
    if [[ "$backbone" == r50 ]]; then
      raw="${RAW_R50[$source]}"
      runner="$R50_RUNNER"
      gpu_group="$R50_GPUS"
      model_tag=r50
    else
      raw="${RAW_VITS[$source]}"
      runner="$VITS_RUNNER"
      gpu_group="$VITS_GPUS"
      model_tag=vits
    fi

    model_prefix="${SFUI_NAME}_${source}_${model_tag}_maskrcnn"
    stage_output="$OUTPUT_ROOT/$model_prefix"
    port="$((BASE_PORT + source_index * 10))"
    log="$OUTPUT_ROOT/logs/${model_prefix}.log"
    source_index=$((source_index + 1))

    echo "START name=$model_prefix runner=$runner gpu_group=$gpu_group port=$port"
    env \
      OUTPUT_ROOT="$stage_output" \
      MODEL_TAG="$model_tag" \
      DINO_RAW="$raw" \
      SFUI_DATA_ROOT="$SFUI_ROOT" \
      SFUI_TRAIN_ANN="$SFUI_TRAIN_ANN" \
      SFUI_VAL_ANN="$SFUI_VAL_ANN" \
      SFUI_TEST_ANN="$SFUI_TEST_ANN" \
      SFUI_TRAIN_PREFIX="$SFUI_TRAIN_PREFIX" \
      SFUI_VAL_PREFIX="$SFUI_VAL_PREFIX" \
      SFUI_TEST_PREFIX="$SFUI_TEST_PREFIX" \
      UIIS_ROOT="$UIIS_ROOT" \
      SFUI_EPOCHS="$DFUI_EPOCHS" \
      UIIS_EPOCHS="$UIIS_EPOCHS" \
      SFUI_GPUS="$gpu_group" \
      UIIS_GPUS="$gpu_group" \
      MIN_FREE_MIB="$MIN_FREE_MIB" \
      BASE_PORT="$port" \
      SFUI_RUN_TEST=0 \
      UIIS_RUN_TEST="$RUN_TEST" \
      MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
      CHECK_ONLY="$CHECK_ONLY" \
      bash "$runner" 2>&1 | tee "$log"

    echo "COMPLETE name=$model_prefix"
  done
done

if [ "$CHECK_ONLY" = 1 ]; then
  echo "CHECK_ONLY=1 passed; no training or evaluation was started"
else
  echo "COMPLETE SFUI=$SFUI_NAME output_root=$OUTPUT_ROOT"
fi
