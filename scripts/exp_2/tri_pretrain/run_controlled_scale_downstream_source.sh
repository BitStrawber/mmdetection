#!/usr/bin/env bash
# Execute one controlled-scale source on one two-GPU group. The source stages
# are deliberately serial: direct RUOD, DFUI-to-RUOD transfer, then direct
# UIIS10K instance segmentation; ResNet-50 precedes ViT-S in each task.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SOURCE="${SOURCE:?set SOURCE to imagenet, realuw, or synthetic5}"
SCALE="${SCALE:-300k}"
R50_RAW="${R50_RAW:?set R50_RAW to the source ResNet-50 DINO checkpoint}"
VITS_RAW="${VITS_RAW:?set VITS_RAW to the source ViT-S DINO checkpoint}"
GPU_GROUP="${GPU_GROUP:?set GPU_GROUP to one two-GPU group, for example 2,3}"
BASE_PORT="${BASE_PORT:?set BASE_PORT to a source-specific port range}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the source output directory}"

DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
RUOD_ROOT="${RUOD_ROOT:-$DATA_ROOT/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-$DATA_ROOT/UIIS10K/coco}"
DFUI_RUOD_UIIS_ROOT="${DFUI_RUOD_UIIS_ROOT:-$DATA_ROOT/DFUI_RUOD_UIIS_EASY}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECK_ONLY="${CHECK_ONLY:-0}"

WORK_ROOT="$OUTPUT_ROOT/work_dirs"
BACKBONE_ROOT="$OUTPUT_ROOT/backbones"
CONVERTED_ROOT="$OUTPUT_ROOT/converted"
LOG_ROOT="$OUTPUT_ROOT/logs"
STATUS_ROOT="$OUTPUT_ROOT/status"
mkdir -p "$WORK_ROOT" "$BACKBONE_ROOT" "$CONVERTED_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

case "$SOURCE" in imagenet|realuw|synthetic5) ;; *) die "unsupported SOURCE: $SOURCE" ;; esac
case "$SCALE" in 100k|300k|500k|800k|1m) ;; *) die "unsupported SCALE: $SCALE" ;; esac
case "$RUN_TEST" in 0|1) ;; *) die "RUN_TEST must be 0 or 1" ;; esac
case "$SKIP_COMPLETED" in 0|1) ;; *) die "SKIP_COMPLETED must be 0 or 1" ;; esac

for root in "$RUOD_ROOT" "$UIIS_ROOT" "$DFUI_RUOD_UIIS_ROOT"; do
  [ -f "$root/annotations/instances_train.json" ] || die "missing train annotations: $root"
  [ -f "$root/annotations/instances_val.json" ] || die "missing validation annotations: $root"
done

{
  printf 'field\tvalue\n'
  printf 'source\t%s\n' "$SOURCE"
  printf 'scale\t%s\n' "$SCALE"
  printf 'gpu_group\t%s\n' "$GPU_GROUP"
  printf 'r50_raw\t%s\n' "$R50_RAW"
  printf 'vits_raw\t%s\n' "$VITS_RAW"
  printf 'dfui_dataset\t%s\n' "$DFUI_RUOD_UIIS_ROOT"
  printf 'dfui_variant\tdfui_ruod_uiis\n'
  printf 'dfui_epochs\t48\n'
  printf 'dfui_followup\truod\n'
  printf 'ruod_dataset\t%s\n' "$RUOD_ROOT"
  printf 'ruod_epochs\t24\n'
  printf 'uiis_dataset\t%s\n' "$UIIS_ROOT"
  printf 'uiis_mask_epochs\t24\n'
  printf 'run_test\t%s\n' "$RUN_TEST"
  printf 'skip_completed\t%s\n' "$SKIP_COMPLETED"
} > "$OUTPUT_ROOT/run_provenance.tsv"

{
  printf 'order\tfinal_result\tarchitecture\tstage\tinit_source\tdataset\tepochs\n'
  printf '1\tdirect_ruod\tresnet50\truod_detector\traw_dino_teacher\tRUOD\t24\n'
  printf '2\tdirect_ruod\tvits\truod_detector\traw_dino_teacher\tRUOD\t24\n'
  printf '3\tdfui_to_ruod\tresnet50\tdfui_detector\traw_dino_teacher\tDFUI_RUOD_UIIS_EASY\t48\n'
  printf '4\tdfui_to_ruod\tresnet50\truod_detector\tdfui_best_backbone\tRUOD\t24\n'
  printf '5\tdfui_to_ruod\tvits\tdfui_detector\traw_dino_teacher\tDFUI_RUOD_UIIS_EASY\t48\n'
  printf '6\tdfui_to_ruod\tvits\truod_detector\tdfui_best_backbone\tRUOD\t24\n'
  printf '7\tdirect_uiis\tresnet50\tuiis_mask\traw_dino_teacher\tUIIS10K\t24\n'
  printf '8\tdirect_uiis\tvits\tuiis_mask\traw_dino_teacher\tUIIS10K\t24\n'
} > "$STATUS_ROOT/task_plan.tsv"

run_phase() {
  local phase="$1" arch="$2" run_direct="$3" direct_tasks="$4" run_dfui="$5" dfui_followups="$6" port_offset="$7"
  local phase_log="$LOG_ROOT/${phase}.pipeline.log"
  echo "============================================================"
  echo "START phase=$phase source=$SOURCE arch=$arch gpus=$GPU_GROUP"
  echo "log=$phase_log"
  echo "============================================================"
  env \
    SOURCE="$SOURCE" \
    SCALE="$SCALE" \
    R50_RAW="$R50_RAW" \
    VITS_RAW="$VITS_RAW" \
    R50_GPUS="$GPU_GROUP" \
    VITS_GPUS="$GPU_GROUP" \
    BASE_PORT="$((BASE_PORT + port_offset))" \
    ARCHITECTURES="$arch" \
    PARALLEL_ARCHITECTURES=0 \
    RUN_DIRECT="$run_direct" \
    DIRECT_TASKS="$direct_tasks" \
    RUN_DFUI="$run_dfui" \
    VARIANTS=dfui_ruod_uiis \
    DFUI_FOLLOWUPS="$dfui_followups" \
    DFUI_EPOCHS=48 \
    WORK_ROOT="$WORK_ROOT" \
    BACKBONE_ROOT="$BACKBONE_ROOT" \
    CONVERTED_ROOT="$CONVERTED_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    PIPELINE_LOG="$phase_log" \
    RUOD_ROOT="$RUOD_ROOT" \
    UIIS_ROOT="$UIIS_ROOT" \
    DFUI_RUOD_UIIS_ROOT="$DFUI_RUOD_UIIS_ROOT" \
    RUN_TEST="$RUN_TEST" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
    CHECK_ONLY="$CHECK_ONLY" \
    bash "$SCRIPT_DIR/run_nested_scale_downstream.sh"
  [ "$CHECK_ONLY" = 1 ] || touch "$STATUS_ROOT/${phase}.complete"
}

echo "============================================================"
echo "Controlled-scale downstream source worker"
echo "source=$SOURCE scale=$SCALE gpu_group=$GPU_GROUP"
echo "output_root=$OUTPUT_ROOT"
echo "dfui_dataset=$DFUI_RUOD_UIIS_ROOT"
echo "============================================================"

# Prioritize direct RUOD baselines, then run the two-stage DFUI transfer.
run_phase direct_ruod_resnet50 resnet50 1 ruod 0 '' 0
run_phase direct_ruod_vits vits 1 ruod 0 '' 20
run_phase dfui_to_ruod_resnet50 resnet50 0 '' 1 ruod 40
run_phase dfui_to_ruod_vits vits 0 '' 1 ruod 70
run_phase direct_uiis_resnet50 resnet50 1 uiis 0 '' 100
run_phase direct_uiis_vits vits 1 uiis 0 '' 120

[ "$CHECK_ONLY" = 1 ] || touch "$STATUS_ROOT/source.complete"
echo "COMPLETE source=$SOURCE scale=$SCALE output_root=$OUTPUT_ROOT"
