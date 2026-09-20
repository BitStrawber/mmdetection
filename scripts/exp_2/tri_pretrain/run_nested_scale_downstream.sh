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
ARCHITECTURES="${ARCHITECTURES:-resnet50,vits}"
PARALLEL_ARCHITECTURES="${PARALLEL_ARCHITECTURES:-1}"
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
DIRECT_TASKS="${DIRECT_TASKS:-ruod,uiis}"
RUN_DFUI="${RUN_DFUI:-1}"
# Unless explicitly requested as an ablation, DFUI means the complete
# DFUI + RUOD Easy + UIIS Easy mixture.
VARIANTS="${VARIANTS:-dfui_ruod_uiis}"
# The standard DFUI experiment is a 48e intermediate detector adaptation
# followed by a 24e RUOD detector transfer.  Mask transfer is opt-in.
DFUI_FOLLOWUPS="${DFUI_FOLLOWUPS:-ruod}"
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

best_checkpoint() { find "$1" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-; }

prepare_dfui_detector_config() {
  local source_config="$1" output_config="$2" use_vits_schedule="$3"
  python - "$source_config" "$output_config" "$use_vits_schedule" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

source, output = map(Path, sys.argv[1:3])
use_vits_schedule = sys.argv[3] == '1'
cfg = Config.fromfile(source)
bbox_heads = cfg.model.roi_head.bbox_head
if not isinstance(bbox_heads, (list, tuple)) or len(bbox_heads) != 3:
    raise SystemExit(
        'Expected a three-stage Cascade R-CNN bbox_head list, got '
        f'{type(bbox_heads).__name__}: {bbox_heads!r}')
for stage in bbox_heads:
    stage['num_classes'] = 11
if use_vits_schedule:
    schedulers = cfg.param_scheduler
    if not isinstance(schedulers, (list, tuple)) or len(schedulers) < 2:
        raise SystemExit('Expected at least two ViT-S parameter schedulers')
    schedulers[1]['milestones'] = [32, 44]
cfg.dump(output)
print(f'Wrote DFUI 11-class Cascade config: {output}')
PY
}

prepare_runtime_config() {
  local source_config="$1" output_config="$2" init="$3" root="$4" kind="$5" epochs="$6" save_best="$7" max_keep_ckpts="$8" is_dfui_detector="$9"
  python - "$source_config" "$output_config" "$init" "$root" "$kind" "$epochs" "$save_best" "$max_keep_ckpts" "$is_dfui_detector" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

source, output, init, root, kind, epochs, save_best, max_keep_ckpts, is_dfui_detector = sys.argv[1:]
source = Path(source)
output = Path(output)
root = Path(root)
is_dfui_detector = is_dfui_detector == '1'
cfg = Config.fromfile(source)

if is_dfui_detector:
    train_images = val_images = 'images/'
else:
    train_images, val_images = 'train/', 'val/'

classes = (
    'holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
    'diver', 'cuttlefish', 'turtle', 'jellyfish', 'waterweeds')

def configure_dataset(dataset, ann_file, image_prefix):
    """Reach the leaf COCO dataset even when it is wrapped by a sampler."""
    if 'dataset' in dataset:
        configure_dataset(dataset['dataset'], ann_file, image_prefix)
        return
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            configure_dataset(child, ann_file, image_prefix)
        return
    dataset['data_root'] = f'{root}/'
    dataset['ann_file'] = str(root / 'annotations' / ann_file)
    dataset['data_prefix'] = dict(img=image_prefix)
    if is_dfui_detector:
        dataset['metainfo'] = dict(classes=classes)

configure_dataset(cfg.train_dataloader['dataset'], 'instances_train.json', train_images)
configure_dataset(cfg.val_dataloader['dataset'], 'instances_val.json', val_images)
configure_dataset(cfg.test_dataloader['dataset'], 'instances_val.json', val_images)

for evaluator_name in ('val_evaluator', 'test_evaluator'):
    evaluator = cfg.get(evaluator_name)
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            item['ann_file'] = str(root / 'annotations' / 'instances_val.json')
    elif evaluator is not None:
        evaluator['ann_file'] = str(root / 'annotations' / 'instances_val.json')

cfg.model.backbone['init_cfg'] = dict(type='Pretrained', checkpoint=init)
cfg['load_from'] = None
cfg.train_cfg['max_epochs'] = int(epochs)
cfg.default_hooks.checkpoint['save_best'] = save_best
cfg.default_hooks.checkpoint['max_keep_ckpts'] = int(max_keep_ckpts)

output.parent.mkdir(parents=True, exist_ok=True)
cfg.dump(output)
print(
    f'Wrote runtime config: {output}; train_prefix={train_images}; '
    f'val_prefix={val_images}; init={init}')
PY
}

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
  local work marker best save_best train_config runtime_config test_log test_status use_vits_schedule=0 is_dfui_detector=0
  work="$WORK_ROOT/$name"
  marker="$work/.complete"
  best="$(best_checkpoint "$work" || true)"
  if [ "$kind" = "det" ]; then save_best='coco/bbox_mAP'; else save_best='coco/segm_mAP'; fi
  test_log="$LOG_ROOT/${name}_test.log"
  if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$marker" ] && [ -n "$best" ]; then echo "REUSE $name: $best"; return; fi

  # A prior test can emit complete COCO metrics yet return nonzero from its launcher.
  # Recover only a fully saved run whose corresponding test log contains the target metric.
  if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$work/epoch_${epochs}.pth" ] && [ -n "$best" ] && \
     { [ "$RUN_TEST" != 1 ] || grep -qF "${save_best}:" "$test_log" 2>/dev/null; }; then
    touch "$marker"
    echo "RECOVER completed $name: $best"
    return
  fi

  mkdir -p "$work"
  if [ "$kind" = "det" ] && [[ "$name" == *dfui_* ]]; then
    is_dfui_detector=1
  fi
  train_config="$config"
  if [ "$is_dfui_detector" = 1 ]; then
    train_config="$work/dfui_11class_config.py"
    [[ "$name" == *vits* ]] && use_vits_schedule=1
    prepare_dfui_detector_config "$config" "$train_config" "$use_vits_schedule"
  fi
  runtime_config="$work/runtime_config.py"
  prepare_runtime_config "$train_config" "$runtime_config" "$init" "$root" "$kind" "$epochs" "$save_best" "$MAX_KEEP_CKPTS" "$is_dfui_detector"
  train_config="$runtime_config"
  echo "START $name  config=$train_config  init=$init  data=$root  gpus=$group"
  CUDA_VISIBLE_DEVICES="$group" PORT="$port" bash tools/dist_train.sh "$train_config" "$(gpu_count "$group")" --work-dir "$work" 2>&1 | tee "$LOG_ROOT/$name.log"
  best="$(best_checkpoint "$work" || true)"; [ -n "$best" ] || die "no best checkpoint: $name"
  if [ "$RUN_TEST" = 1 ]; then
    set +e
    CUDA_VISIBLE_DEVICES="$group" PORT="$((port+100))" bash tools/dist_test.sh "$train_config" "$best" "$(gpu_count "$group")" 2>&1 | tee "$test_log"
    test_status="${PIPESTATUS[0]}"
    set -e
    if [ "$test_status" -ne 0 ]; then
      if grep -qF "${save_best}:" "$test_log"; then
        echo "WARNING: $name test launcher exited $test_status after reporting ${save_best}; accepting completed evaluation."
      else
        die "$name test failed with exit code $test_status and did not report ${save_best}"
      fi
    fi
  fi
  touch "$marker"
}

run_arch() {
  local arch="$1" group="$2" port="$3" raw init det mask dfui_base
  if [ "$arch" = resnet50 ]; then raw="$R50_RAW"; init="$CONVERTED_ROOT/scale${SCALE}_${SOURCE}_r50_teacher.pth"; det="$R50_DET_CONFIG"; mask="$R50_MASK_CONFIG"; dfui_base="$R50_DFUI_RUOD_CONFIG"; convert "$raw" "$init" resnet50 ''; else raw="$VITS_RAW"; init="$CONVERTED_ROOT/scale${SCALE}_${SOURCE}_vits_teacher.pth"; det="$VITS_DET_CONFIG"; mask="$VITS_MASK_CONFIG"; dfui_base="$VITS_DFUI_CONFIG"; convert "$raw" "$init" vit_small backbone.; fi
  local prefix="scale${SCALE}_${SOURCE}_${arch}"
  if [ "$RUN_DIRECT" = 1 ]; then
    [[ ",$DIRECT_TASKS," == *,ruod,* ]] && run_train "${prefix}_direct_ruod24e_det" "$det" "$init" "$RUOD_ROOT" det "$group" "$port" 24
    [[ ",$DIRECT_TASKS," == *,uiis,* ]] && run_train "${prefix}_direct_uiis24e_mask" "$mask" "$init" "$UIIS_ROOT" mask "$group" "$((port+1))" 24
  fi
  [ "$RUN_DFUI" = 1 ] || return
  local -a branches=()
  IFS=',' read -r -a branches <<< "$VARIANTS"
  for branch in "${branches[@]}"; do
    branch="${branch//[[:space:]]/}"
    case "$branch" in
      dfui_ruod|dfui_ruod_uiis) ;;
      *) die "unsupported DFUI variant: $branch (expected dfui_ruod and/or dfui_ruod_uiis)" ;;
    esac
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
IFS=',' read -r -a ARCHITECTURE_LIST <<< "$ARCHITECTURES"
[ "${#ARCHITECTURE_LIST[@]}" -gt 0 ] || die "ARCHITECTURES must contain resnet50 and/or vits"
for arch in "${ARCHITECTURE_LIST[@]}"; do
  arch="${arch//[[:space:]]/}"
  case "$arch" in resnet50|vits) ;; *) die "unsupported architecture: $arch" ;; esac
done
case "$PARALLEL_ARCHITECTURES" in 0|1) ;; *) die "PARALLEL_ARCHITECTURES must be 0 or 1" ;; esac
echo "source=$SOURCE scale=$SCALE raw_root=$RAW_ROOT architectures=$ARCHITECTURES parallel_architectures=$PARALLEL_ARCHITECTURES direct_tasks=$DIRECT_TASKS variants=$VARIANTS followups=$DFUI_FOLLOWUPS log=$PIPELINE_LOG"
if [ "$CHECK_ONLY" = 1 ]; then echo "CHECK_ONLY=1 passed"; exit 0; fi
if [ "$PARALLEL_ARCHITECTURES" = 1 ]; then
  pids=()
  for arch in "${ARCHITECTURE_LIST[@]}"; do
    arch="${arch//[[:space:]]/}"
    if [ "$arch" = resnet50 ]; then
      run_arch resnet50 "$R50_GPUS" "$BASE_PORT" & pids+=("$!")
    else
      run_arch vits "$VITS_GPUS" "$((BASE_PORT+200))" & pids+=("$!")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
else
  for arch in "${ARCHITECTURE_LIST[@]}"; do
    arch="${arch//[[:space:]]/}"
    if [ "$arch" = resnet50 ]; then
      run_arch resnet50 "$R50_GPUS" "$BASE_PORT"
    else
      run_arch vits "$VITS_GPUS" "$((BASE_PORT+200))"
    fi
  done
fi
echo "COMPLETE source=$SOURCE scale=$SCALE"
