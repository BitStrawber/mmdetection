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

DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
UIIS_ROOT="${UIIS_ROOT:-$DATA_ROOT/UIIS10K/coco}"
RUOD_ROOT="${RUOD_ROOT:-$DATA_ROOT/RUOD/coco}"
SFUI_CLASS_NAMES_FILE="${SFUI_CLASS_NAMES_FILE:-}"
IMAGENET_R50_RAW="${IMAGENET_R50_RAW:?set full ImageNet DINO ResNet-50 checkpoint}"
IMAGENET_VITS_RAW="${IMAGENET_VITS_RAW:?set full ImageNet DINO ViT-S checkpoint}"
REALUW_R50_RAW="${REALUW_R50_RAW:?set full RealUW DINO ResNet-50 checkpoint}"
REALUW_VITS_RAW="${REALUW_VITS_RAW:?set full RealUW DINO ViT-S checkpoint}"
SYNTHETIC5_R50_RAW="${SYNTHETIC5_R50_RAW:?set full Synthetic5 DINO ResNet-50 checkpoint}"
SYNTHETIC5_VITS_RAW="${SYNTHETIC5_VITS_RAW:?set full Synthetic5 DINO ViT-S checkpoint}"

R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-30600}"
DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
UIIS_EPOCHS="${UIIS_EPOCHS:-24}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECK_ONLY="${CHECK_ONLY:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-0}"

PIPELINE="$SCRIPT_DIR/run_pretrained_dfui_then_uiis_mask.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$SFUI_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "SFUI_NAME may contain only letters, digits, dot, underscore, and hyphen"
[[ "$CHECK_ONLY" =~ ^[01]$ ]] || die "CHECK_ONLY must be 0 or 1"
[[ "$RUN_TEST" =~ ^[01]$ ]] || die "RUN_TEST must be 0 or 1"
[[ "$DFUI_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "DFUI_EPOCHS must be a positive integer"
[[ "$UIIS_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "UIIS_EPOCHS must be a positive integer"
[[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || die "MIN_FREE_MIB must be a non-negative integer"

for file in \
  "$PIPELINE" \
  scripts/exp_2/tri_pretrain/run_nested_scale_downstream.sh \
  tools/dist_train.sh tools/dist_test.sh tools/convert_ssl_backbone_to_mmdet.py \
  configs/exp_2/dfui_imagenet1k_stage_configs/r50_dfui_ruod/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_easy_j10_scheme_c_s1.py \
  configs/exp_2/dfui_imagenet1k_stage_configs/r50_dfui_ruod_uiis/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py \
  configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py \
  configs/exp_2/mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py \
  configs/exp_2/tri_pretrain/mask-rcnn_vit-small_dino_fpn_24e_uiis10k_control100k.py; do
  [ -s "$file" ] || die "required repository file is missing: $file"
done

for root in "$RUOD_ROOT" "$UIIS_ROOT"; do
  [ -s "$root/annotations/instances_train.json" ] &&
  [ -s "$root/annotations/instances_val.json" ] || die "missing downstream dataset annotations: $root"
done

python - "$SFUI_ROOT" "$SFUI_CLASS_NAMES_FILE" "$UIIS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

sfui_root = Path(sys.argv[1])
class_file = Path(sys.argv[2]) if sys.argv[2] else None
uiis_root = Path(sys.argv[3])

def load_split(root, split, prefix):
    ann_path = root / "annotations" / f"instances_{split}.json"
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

train, train_categories = load_split(sfui_root, "train", "images")
val, val_categories = load_split(sfui_root, "val", "images")
if train_categories != val_categories:
    raise SystemExit("SFUI train/val category IDs and names differ")
train_names = {image["file_name"] for image in train["images"]}
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

uiis_counts = {}
for split in ("train", "val"):
    data, _ = load_split(uiis_root, split, split)
    uiis_counts[split] = len(data["images"])

print(f"SFUI_ROOT={sfui_root}")
print(f"SFUI_train_images={len(train['images'])} SFUI_val_images={len(val['images'])}")
print(f"SFUI_train_annotations={len(train.get('annotations', []))} SFUI_val_annotations={len(val.get('annotations', []))}")
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
    wanted[$1] { print "GPU=" $1 " free_MiB=" $2; if ($2 < min) failed=1; seen[$1]=1 }
    END { for (gpu in wanted) if (!seen[gpu]) failed=1; exit failed }
  ' || die "selected GPUs do not meet MIN_FREE_MIB=$MIN_FREE_MIB"
fi

mkdir -p "$OUTPUT_ROOT"
PROVENANCE="$OUTPUT_ROOT/run_provenance.tsv"
{
  printf 'field\tvalue\n'
  printf 'sfui_name\t%s\n' "$SFUI_NAME"
  printf 'sfui_root\t%s\n' "$SFUI_ROOT"
  printf 'sfui_class_names_file\t%s\n' "${SFUI_CLASS_NAMES_FILE:-ANNOTATION_CATEGORIES}"
  printf 'uiis_root\t%s\n' "$UIIS_ROOT"
  printf 'r50_gpus\t%s\n' "$R50_GPUS"
  printf 'vits_gpus\t%s\n' "$VITS_GPUS"
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

echo "SFUI -> UIIS full-pretraining comparison"
echo "sfui_name=$SFUI_NAME sfui_root=$SFUI_ROOT"
echo "three sources x two backbones; SFUI detector=${DFUI_EPOCHS}e -> UIIS Mask R-CNN=${UIIS_EPOCHS}e"
echo "R50_GPUS=$R50_GPUS VITS_GPUS=$VITS_GPUS check_only=$CHECK_ONLY"
echo "output_root=$OUTPUT_ROOT"

source_index=0
for source in imagenet realuw synthetic5; do
  model_prefix="${SFUI_NAME}_${source}_full"
  source_output="$OUTPUT_ROOT/$source"
  port="$((BASE_PORT + source_index * 1000))"
  echo "START_SOURCE=$source MODEL_PREFIX=$model_prefix PORT=$port"

  MODEL_PREFIX="$model_prefix" \
  SOURCE_LABEL="$source" \
  SCALE_LABEL=full \
  R50_RAW="${RAW_R50[$source]}" \
  VITS_RAW="${RAW_VITS[$source]}" \
  DATA_ROOT="$DATA_ROOT" \
  RUOD_ROOT="$RUOD_ROOT" \
  UIIS_ROOT="$UIIS_ROOT" \
  DFUI_RUOD_ROOT="$SFUI_ROOT" \
  DFUI_RUOD_UIIS_ROOT="$SFUI_ROOT" \
  DFUI_DYNAMIC_CLASSES=1 \
  DFUI_CLASS_NAMES_FILE="$SFUI_CLASS_NAMES_FILE" \
  VARIANTS=dfui_ruod \
  DFUI_FOLLOWUPS=mask \
  ALLOW_TARGET_DOMAIN_UIIS=0 \
  DFUI_EPOCHS="$DFUI_EPOCHS" \
  R50_GPUS="$R50_GPUS" \
  VITS_GPUS="$VITS_GPUS" \
  BASE_PORT="$port" \
  OUTPUT_ROOT="$source_output" \
  WORK_ROOT="$source_output/work_dirs" \
  BACKBONE_ROOT="$source_output/backbones" \
  CONVERTED_ROOT="$source_output/converted" \
  LOG_ROOT="$source_output/logs" \
  RUN_TEST="$RUN_TEST" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
  UIIS_MASK_EPOCHS="$UIIS_EPOCHS" \
  CHECK_ONLY="$CHECK_ONLY" \
  RUN_DIRECT=0 \
  RUN_DFUI=1 \
  bash "$PIPELINE"

  source_index=$((source_index + 1))
done

if [ "$CHECK_ONLY" = 1 ]; then
  echo "CHECK_ONLY=1 passed; no training or evaluation was started"
else
  echo "COMPLETE SFUI=$SFUI_NAME output_root=$OUTPUT_ROOT"
fi
