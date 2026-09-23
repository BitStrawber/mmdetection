#!/usr/bin/env bash
# Execute one controlled-scale source on one two-GPU group. The four direct 24e
# tasks are deliberately serial; ResNet-50 precedes ViT-S throughout.
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

for root in "$RUOD_ROOT" "$UIIS_ROOT"; do
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
  printf '2\tdirect_uiis\tresnet50\tuiis_mask\traw_dino_teacher\tUIIS10K\t24\n'
  printf '3\tdirect_ruod\tvits\truod_detector\traw_dino_teacher\tRUOD\t24\n'
  printf '4\tdirect_uiis\tvits\tuiis_mask\traw_dino_teacher\tUIIS10K\t24\n'
} > "$STATUS_ROOT/task_plan.tsv"

run_phase() {
  local phase="$1" arch="$2" direct_tasks="$3" port_offset="$4"
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
    RUN_DIRECT=1 \
    DIRECT_TASKS="$direct_tasks" \
    RUN_DFUI=0 \
    WORK_ROOT="$WORK_ROOT" \
    BACKBONE_ROOT="$BACKBONE_ROOT" \
    CONVERTED_ROOT="$CONVERTED_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    PIPELINE_LOG="$phase_log" \
    RUOD_ROOT="$RUOD_ROOT" \
    UIIS_ROOT="$UIIS_ROOT" \
    RUN_TEST="$RUN_TEST" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
    CHECK_ONLY="$CHECK_ONLY" \
    bash "$SCRIPT_DIR/run_nested_scale_downstream.sh"
  if [ "$CHECK_ONLY" != 1 ]; then
    touch "$STATUS_ROOT/${phase}.complete"
  fi
  return 0
}

echo "============================================================"
echo "Controlled-scale downstream source worker"
echo "source=$SOURCE scale=$SCALE gpu_group=$GPU_GROUP"
echo "output_root=$OUTPUT_ROOT"
echo "============================================================"

# Run the four direct baselines in the requested architecture and task order.
run_phase direct_ruod_resnet50 resnet50 ruod 0
run_phase direct_uiis_resnet50 resnet50 uiis 20
run_phase direct_ruod_vits vits ruod 40
run_phase direct_uiis_vits vits uiis 60

[ "$CHECK_ONLY" = 1 ] || touch "$STATUS_ROOT/source.complete"
echo "COMPLETE source=$SOURCE scale=$SCALE output_root=$OUTPUT_ROOT"
