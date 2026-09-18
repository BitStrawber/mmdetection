#!/usr/bin/env bash
# Launch the complete 300k downstream matrix.  Sources run independently on
# separate two-GPU groups while every source's eight training stages are serial.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN/Controlled300K}"
DATA_ROOT="${DATA_ROOT:-/media/HDD0/XCX/exp_2}"
RUN_NAME="${RUN_NAME:-controlled300k_dfui_ruod_uiis_direct_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/downstream_runs/$RUN_NAME}"
RUN_TEST="${RUN_TEST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
CHECK_ONLY="${CHECK_ONLY:-0}"

IMAGENET_GPUS="${IMAGENET_GPUS:-2,3}"
REALUW_GPUS="${REALUW_GPUS:-4,5}"
SYNTHETIC5_GPUS="${SYNTHETIC5_GPUS:-6,7}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
MASTER_LOG="${MASTER_LOG:-$OUTPUT_ROOT/logs/matrix.log}"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() { echo "ERROR: $*" >&2; exit 1; }

for file in \
  "$PRETRAIN_ROOT/ImageNet/DINO_ResNet50_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/ImageNet/DINO_ViTS_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/RealUW/DINO_ResNet50_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/RealUW/DINO_ViTS_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/Synthetic5/DINO_ResNet50_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/Synthetic5/DINO_ViTS_100e/checkpoint.pth" \
  "$PRETRAIN_ROOT/metadata/CHECKSUMS.sha256"; do
  [ -s "$file" ] || die "missing required Controlled300K file: $file"
done

{
  printf 'source\tgpu_group\tr50_checkpoint\tvits_checkpoint\tfinal_results\n'
  printf 'imagenet\t%s\t%s\t%s\tdfui_to_ruod,direct_ruod,direct_uiis\n' "$IMAGENET_GPUS" "$PRETRAIN_ROOT/ImageNet/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/ImageNet/DINO_ViTS_100e/checkpoint.pth"
  printf 'realuw\t%s\t%s\t%s\tdfui_to_ruod,direct_ruod,direct_uiis\n' "$REALUW_GPUS" "$PRETRAIN_ROOT/RealUW/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/RealUW/DINO_ViTS_100e/checkpoint.pth"
  printf 'synthetic5\t%s\t%s\t%s\tdfui_to_ruod,direct_ruod,direct_uiis\n' "$SYNTHETIC5_GPUS" "$PRETRAIN_ROOT/Synthetic5/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/Synthetic5/DINO_ViTS_100e/checkpoint.pth"
} > "$OUTPUT_ROOT/run_matrix.tsv"

launch_source() {
  local source="$1" gpu_group="$2" base_port="$3" r50="$4" vits="$5"
  local source_root="$OUTPUT_ROOT/$source"
  mkdir -p "$source_root/logs"
  env \
    SOURCE="$source" \
    SCALE=300k \
    R50_RAW="$r50" \
    VITS_RAW="$vits" \
    GPU_GROUP="$gpu_group" \
    BASE_PORT="$base_port" \
    OUTPUT_ROOT="$source_root" \
    DATA_ROOT="$DATA_ROOT" \
    RUN_TEST="$RUN_TEST" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
    CHECK_ONLY="$CHECK_ONLY" \
    bash "$SCRIPT_DIR/run_controlled_scale_downstream_source.sh" \
    > "$source_root/logs/source_launcher.log" 2>&1 &
  LAUNCHED_PID="$!"
}

echo "============================================================"
echo "Controlled300K downstream matrix"
echo "output_root=$OUTPUT_ROOT"
echo "ImageNet GPUs=$IMAGENET_GPUS"
echo "RealUW GPUs=$REALUW_GPUS"
echo "Synthetic5 GPUs=$SYNTHETIC5_GPUS"
echo "Each source runs eight training stages serially."
echo "============================================================"

launch_source imagenet "$IMAGENET_GPUS" 31000 "$PRETRAIN_ROOT/ImageNet/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/ImageNet/DINO_ViTS_100e/checkpoint.pth"
pid_imagenet="$LAUNCHED_PID"
launch_source realuw "$REALUW_GPUS" 32000 "$PRETRAIN_ROOT/RealUW/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/RealUW/DINO_ViTS_100e/checkpoint.pth"
pid_realuw="$LAUNCHED_PID"
launch_source synthetic5 "$SYNTHETIC5_GPUS" 33000 "$PRETRAIN_ROOT/Synthetic5/DINO_ResNet50_100e/checkpoint.pth" "$PRETRAIN_ROOT/Synthetic5/DINO_ViTS_100e/checkpoint.pth"
pid_synthetic5="$LAUNCHED_PID"

printf 'source\tpid\nimagenet\t%s\nrealuw\t%s\nsynthetic5\t%s\n' \
  "$pid_imagenet" "$pid_realuw" "$pid_synthetic5" \
  > "$OUTPUT_ROOT/status/source_workers.tsv"

status=0
wait "$pid_imagenet" || status=1
wait "$pid_realuw" || status=1
wait "$pid_synthetic5" || status=1

if [ "$status" -ne 0 ]; then
  echo "FAILED: one or more source workers failed; inspect $OUTPUT_ROOT/*/logs/source_launcher.log" >&2
  exit "$status"
fi

[ "$CHECK_ONLY" = 1 ] || touch "$OUTPUT_ROOT/status/matrix.complete"
echo "COMPLETE controlled300k matrix output_root=$OUTPUT_ROOT"
