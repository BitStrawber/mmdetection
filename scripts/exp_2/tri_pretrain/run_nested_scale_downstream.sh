#!/usr/bin/env bash
# One manually transferred nested-scale DINO pair -> direct and DFUI-adapted
# downstream evaluation.  R50 and ViT-S pipelines run on independent 2-GPU groups.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SOURCE="${SOURCE:?set SOURCE to imagenet, realuw, or synthetic5}"
SCALE="${SCALE:?set SCALE to 100k, 300k, 500k, 800k, or 1m}"
RAW_ROOT="${RAW_ROOT:-/media/SSD1/XCX/exp_2/manual_dino_nested_transfer}"
R50_RAW="${R50_RAW:-$RAW_ROOT/scale${SCALE}_${SOURCE}_dino_resnet50_100e/checkpoint.pth}"
VITS_RAW="${VITS_RAW:-$RAW_ROOT/scale${SCALE}_${SOURCE}_dino_vits_100e/checkpoint.pth}"
R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-30300}"
WORK_ROOT="${WORK_ROOT:-work_dirs/nested_scale_downstream}"
BACKBONE_ROOT="${BACKBONE_ROOT:-work_dirs/nested_scale_downstream_backbones}"
CONVERTED_ROOT="${CONVERTED_ROOT:-work_dirs/nested_scale_downstream_converted}"
LOG_ROOT="${LOG_ROOT:-logs/nested_scale_downstream}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2/UIIS10K/coco}"
DFUI_RUOD_ROOT="${DFUI_RUOD_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_EASY}"
DFUI_RUOD_UIIS_ROOT="${DFUI_RUOD_UIIS_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY}"
R50_DET_CONFIG="${R50_DET_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py}"
R50_MASK_CONFIG="${R50_MASK_CONFIG:-configs/exp_2/mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py}"
VITS_DET_CONFIG="${VITS_DET_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
VITS_MASK_CONFIG="${VITS_MASK_CONFIG:-configs/exp_2/tri_pretrain/mask-rcnn_vit-small_dino_fpn_24e_uiis10k_control100k.py}"
R50_DFUI_RUOD_CONFIG="${R50_DFUI_RUOD_CONFIG:-configs/exp_2/dfui_imagenet1k_stage_configs/r50_dfui_ruod/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_easy_j10_scheme_c_s1.py}"
R50_DFUI_RUOD_UIIS_CONFIG="${R50_DFUI_RUOD_UIIS_CONFIG:-configs/exp_2/dfui_imagenet1k_stage_configs/r50_dfui_ruod_uiis/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py}"
VITS_DFUI_CONFIG="${VITS_DFUI_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
RUN_DIRECT="${RUN_DIRECT:-1}"
RUN_DFUI="${RUN_DFUI:-1}"
DFUI_FOLLOWUPS="${DFUI_FOLLOWUPS:-ruod,mask}"
DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

mkdir -p "$WORK_ROOT" "$BACKBONE_ROOT" "$CONVERTED_ROOT" "$LOG_ROOT"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/${SOURCE}_${SCALE}_pipeline_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
die() { echo "ERROR: $*" >&2; exit 1; }
gpu_count() { awk -F, '{print NF}' <<< "$1"; }

validate_raw() {
  local raw="$1" arch="$2"
  python - "$raw" "$arch" <<'PY'
import sys, torch
p, expected = sys.argv[1:]
c = torch.load(p, map_location='cpu', weights_only=False)
a, t = c.get('args'), c.get('teacher')
print(f'raw={p} epoch={c.get("epoch")} arch={getattr(a, "arch", None)} tensors={len(t) if isinstance(t,dict) else 0}')
if c.get('epoch') != 100 or getattr(a, 'arch', None) != expected or not isinstance(t, dict): raise SystemExit('invalid raw DINO checkpoint')
PY
}

convert() {
  local raw="$1" output="$2" arch="$3" prefix="$4"
  validate_raw "$raw" "$arch"
  [ -s "$output" ] && return
  local partial="${output}.partial.$$"
  rm -f -- "$partial"
  python tools/convert_ssl_backbone_to_mmdet.py --checkpoint "$raw" --source teacher --prepend "$prefix" --out "$partial"
  [ -s "$partial" ] || die "conversion failed: $output"
  mv -- "$partial" "$output"
}

best_checkpoint() { find "$1" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-; }
export_backbone() {
  local source="$1" output="$2"
  [ -s "$output" ] && return
  python - "$source" "$output" <<'PY'
import sys, torch
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
c = torch.load(src, map_location='cpu', weights_only=False); state=c.get('state_dict',c)
b={k[9:]:v for k,v in state.items() if k.startswith('backbone.')}
if not b: raise SystemExit('no backbone tensors')
dst.parent.mkdir(parents=True, exist_ok=True); torch.save({'state_dict':b,'meta':{'source':str(src)}},dst)
PY
}

run_train() {
  local name="$1" config="$2" init="$3" root="$4" kind="$5" group="$6" port="$7" epochs="$8"
  local work="$WORK_ROOT/$name" marker="$work/.complete" best save_best train_images val_images
  best="$(best_checkpoint "$work" || true)"
  if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$marker" ] && [ -n "$best" ]; then echo "REUSE $name: $best"; return; fi
  mkdir -p "$work"
  if [ "$kind" = "det" ]; then save_best='coco/bbox_mAP'; else save_best='coco/segm_mAP'; fi
  if [[ "$name" == *dfui_* ]]; then
    # Both DFUI mixtures store all train and validation images under images/.
    train_images="$root/images/"; val_images="$root/images/"
  else
    train_images="$root/train/"; val_images="$root/val/"
  fi
  echo "START $name  config=$config  init=$init  data=$root  gpus=$group"
  local opts=(load_from=None model.backbone.init_cfg.type=Pretrained model.backbone.init_cfg.checkpoint="$init"
    train_cfg.max_epochs="$epochs" default_hooks.checkpoint.save_best="$save_best" default_hooks.checkpoint.max_keep_ckpts="$MAX_KEEP_CKPTS"
    train_dataloader.dataset.data_root="$root/" train_dataloader.dataset.ann_file="$root/annotations/instances_train.json" train_dataloader.dataset.data_prefix.img="$train_images"
    val_dataloader.dataset.data_root="$root/" val_dataloader.dataset.ann_file="$root/annotations/instances_val.json" val_dataloader.dataset.data_prefix.img="$val_images"
    test_dataloader.dataset.data_root="$root/" test_dataloader.dataset.ann_file="$root/annotations/instances_val.json" test_dataloader.dataset.data_prefix.img="$val_images"
    val_evaluator.ann_file="$root/annotations/instances_val.json" test_evaluator.ann_file="$root/annotations/instances_val.json")
  if [[ "$name" == *dfui_* ]]; then
    opts+=(model.roi_head.bbox_head.0.num_classes=11 model.roi_head.bbox_head.1.num_classes=11 model.roi_head.bbox_head.2.num_classes=11)
    opts+=(train_dataloader.dataset.metainfo.classes="('holothurian','echinus','scallop','starfish','fish','corals','diver','cuttlefish','turtle','jellyfish','waterweeds')" val_dataloader.dataset.metainfo.classes="('holothurian','echinus','scallop','starfish','fish','corals','diver','cuttlefish','turtle','jellyfish','waterweeds')" test_dataloader.dataset.metainfo.classes="('holothurian','echinus','scallop','starfish','fish','corals','diver','cuttlefish','turtle','jellyfish','waterweeds')")
    [[ "$name" == *vits* ]] && opts+=(param_scheduler.1.milestones='[32,44]')
  fi
  CUDA_VISIBLE_DEVICES="$group" PORT="$port" bash tools/dist_train.sh "$config" "$(gpu_count "$group")" --work-dir "$work" --cfg-options "${opts[@]}" 2>&1 | tee "$LOG_ROOT/$name.log"
  best="$(best_checkpoint "$work" || true)"; [ -n "$best" ] || die "no best checkpoint: $name"; touch "$marker"
  if [ "$RUN_TEST" = 1 ]; then CUDA_VISIBLE_DEVICES="$group" PORT="$((port+100))" bash tools/dist_test.sh "$config" "$best" "$(gpu_count "$group")" --cfg-options "${opts[@]}" 2>&1 | tee "$LOG_ROOT/${name}_test.log"; fi
}

run_arch() {
  local arch="$1" group="$2" port="$3" raw init det mask dfui_base
  if [ "$arch" = resnet50 ]; then raw="$R50_RAW"; init="$CONVERTED_ROOT/scale${SCALE}_${SOURCE}_r50_teacher.pth"; det="$R50_DET_CONFIG"; mask="$R50_MASK_CONFIG"; dfui_base="$R50_DFUI_RUOD_CONFIG"; convert "$raw" "$init" resnet50 ''; else raw="$VITS_RAW"; init="$CONVERTED_ROOT/scale${SCALE}_${SOURCE}_vits_teacher.pth"; det="$VITS_DET_CONFIG"; mask="$VITS_MASK_CONFIG"; dfui_base="$VITS_DFUI_CONFIG"; convert "$raw" "$init" vit_small backbone.; fi
  local prefix="scale${SCALE}_${SOURCE}_${arch}"
  if [ "$RUN_DIRECT" = 1 ]; then run_train "${prefix}_direct_ruod24e_det" "$det" "$init" "$RUOD_ROOT" det "$group" "$port" 24; run_train "${prefix}_direct_uiis24e_mask" "$mask" "$init" "$UIIS_ROOT" mask "$group" "$((port+1))" 24; fi
  [ "$RUN_DFUI" = 1 ] || return
  for branch in dfui_ruod dfui_ruod_uiis; do
    local root config dfui_name dfui_work dfui_best adapted
    if [ "$branch" = dfui_ruod ]; then root="$DFUI_RUOD_ROOT"; config="$dfui_base"; else root="$DFUI_RUOD_UIIS_ROOT"; config="${R50_DFUI_RUOD_UIIS_CONFIG}"; [ "$arch" = vits ] && config="$VITS_DFUI_CONFIG"; fi
    dfui_name="${prefix}_${branch}_cascade48e"; run_train "$dfui_name" "$config" "$init" "$root" det "$group" "$((port+10))" "$DFUI_EPOCHS"
    dfui_work="$WORK_ROOT/$dfui_name"; dfui_best="$(best_checkpoint "$dfui_work")"; adapted="$BACKBONE_ROOT/${dfui_name}_best_backbone.pth"; export_backbone "$dfui_best" "$adapted"
    [[ ",$DFUI_FOLLOWUPS," == *,ruod,* ]] && run_train "${prefix}_${branch}_backbone_ruod24e_det" "$det" "$adapted" "$RUOD_ROOT" det "$group" "$((port+11))" 24
    [[ ",$DFUI_FOLLOWUPS," == *,mask,* ]] && run_train "${prefix}_${branch}_backbone_uiis24e_mask" "$mask" "$adapted" "$UIIS_ROOT" mask "$group" "$((port+12))" 24
    port=$((port+20))
  done
}

for file in tools/dist_train.sh tools/dist_test.sh tools/convert_ssl_backbone_to_mmdet.py "$R50_RAW" "$VITS_RAW" "$R50_DET_CONFIG" "$R50_MASK_CONFIG" "$VITS_DET_CONFIG" "$VITS_MASK_CONFIG" "$R50_DFUI_RUOD_CONFIG" "$R50_DFUI_RUOD_UIIS_CONFIG" "$VITS_DFUI_CONFIG"; do [ -s "$file" ] || die "missing required file: $file"; done
for root in "$RUOD_ROOT" "$UIIS_ROOT" "$DFUI_RUOD_ROOT" "$DFUI_RUOD_UIIS_ROOT"; do [ -f "$root/annotations/instances_train.json" ] && [ -f "$root/annotations/instances_val.json" ] || die "invalid dataset: $root"; done
validate_raw "$R50_RAW" resnet50; validate_raw "$VITS_RAW" vit_small
echo "source=$SOURCE scale=$SCALE raw_root=$RAW_ROOT followups=$DFUI_FOLLOWUPS log=$PIPELINE_LOG"
if [ "$CHECK_ONLY" = 1 ]; then echo "CHECK_ONLY=1 passed"; exit 0; fi
run_arch resnet50 "$R50_GPUS" "$BASE_PORT" & p1=$!
run_arch vits "$VITS_GPUS" "$((BASE_PORT+200))" & p2=$!
wait "$p1"; wait "$p2"
echo "COMPLETE source=$SOURCE scale=$SCALE"
