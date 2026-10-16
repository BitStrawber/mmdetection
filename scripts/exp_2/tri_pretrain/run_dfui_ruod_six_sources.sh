#!/usr/bin/env bash
set -Eeuo pipefail

# Generic serial launcher for six pretrained-source -> DFUI -> RUOD runs.
# The underlying R50/ViT runners must honor at least one checkpoint variable
# passed below; the launcher supplies common aliases for older runners.

die() {
  echo "ERROR: $*" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DFUI_ROOT="${DFUI_ROOT:?set DFUI_ROOT}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/RUOD/coco}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/media/SSD1/XCX/exp_2/BitStrawber_Output/PRETRAIN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/dfui_ruod_runs/dfui_ruod_six_sources_$(date +%Y%m%d_%H%M%S)}"

R50_SCRIPT="${R50_SCRIPT:-$repo_root/scripts/exp_2/tri_pretrain/run_dfui_to_ruod_r50.sh}"
VITS_SCRIPT="${VITS_SCRIPT:-$repo_root/scripts/exp_2/tri_pretrain/run_dfui_to_ruod_vits.sh}"

R50_GPUS="${R50_GPUS:-4,5}"
VITS_GPUS="${VITS_GPUS:-4,5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8192}"
BASE_PORT="${BASE_PORT:-30600}"
DFUI_EPOCHS="${DFUI_EPOCHS:-48}"
RUOD_EPOCHS="${RUOD_EPOCHS:-24}"
DFUI_LR="${DFUI_LR:-0.005}"
RUOD_LR="${RUOD_LR:-0.005}"
RUN_TEST="${RUN_TEST:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
SOURCES="${SOURCES:-all}"

R50_ARGS="${R50_ARGS:-}"
VITS_ARGS="${VITS_ARGS:-}"

MANIFEST="${MANIFEST:-$OUTPUT_ROOT/pretrained_sources.tsv}"
MASTER_LOG="$OUTPUT_ROOT/logs/master_pipeline.log"
RESULTS_TSV="$OUTPUT_ROOT/results.tsv"
PROVENANCE_TSV="$OUTPUT_ROOT/run_provenance.tsv"

mkdir -p "$OUTPUT_ROOT/logs"

require_file() {
  [ -s "$1" ] || die "missing or empty file: $1"
}

require_dir() {
  [ -d "$1" ] || die "missing directory: $1"
}

check_dataset() {
  local root="$1"
  require_file "$root/annotations/instances_train.json"
  require_file "$root/annotations/instances_val.json"
}

gpu_is_ready() {
  local group="$1"
  local failed=0
  local seen=0

  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

  IFS=',' read -ra requested <<< "$group"
  while IFS=',' read -r index free; do
    index="${index//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    for wanted in "${requested[@]}"; do
      if [ "$index" = "$wanted" ]; then
        seen=$((seen + 1))
        echo "GPU=$index free_MiB=$free"
        [ "$free" -ge "$MIN_FREE_MIB" ] || failed=1
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  [ "$seen" -eq "${#requested[@]}" ] || failed=1
  return "$failed"
}

port_is_free() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn | grep -q ":${port} "
  else
    ! (command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1)
  fi
}

write_default_manifest() {
  mkdir -p "$(dirname "$MANIFEST")"
  cat > "$MANIFEST" <<EOF
# name|backbone|checkpoint
imagenet_r50|r50|$PRETRAIN_ROOT/ImageNet/DINO_ResNet50_100e/checkpoint.pth
imagenet_vits|vits|$PRETRAIN_ROOT/ImageNet/DINO_ViTS_100e/checkpoint.pth
realuw_r50|r50|$PRETRAIN_ROOT/RealUW/DINO_ResNet50_100e/checkpoint.pth
realuw_vits|vits|$PRETRAIN_ROOT/RealUW/DINO_ViTS_100e/checkpoint.pth
synthetic5_r50|r50|$PRETRAIN_ROOT/Controlled1M/Synthetic5/DINO_ResNet50_100e/checkpoint.pth
synthetic5_vits|vits|$PRETRAIN_ROOT/Controlled1M/Synthetic5/DINO_ViTS_100e/checkpoint.pth
EOF
}

if [ ! -e "$MANIFEST" ]; then
  write_default_manifest
fi

require_dir "$DFUI_ROOT"
require_dir "$RUOD_ROOT"
check_dataset "$DFUI_ROOT"
check_dataset "$RUOD_ROOT"
require_file "$R50_SCRIPT"
require_file "$VITS_SCRIPT"

if ! grep -qE 'PRETRAIN_CKPT|S1_CKPT|DINO_RAW|SOURCE_CKPT|CHECKPOINT' "$R50_SCRIPT"; then
  die "R50 runner does not expose a recognizable pretrained-checkpoint variable: $R50_SCRIPT"
fi
if ! grep -qE 'PRETRAIN_CKPT|S1_CKPT|DINO_RAW|SOURCE_CKPT|CHECKPOINT' "$VITS_SCRIPT"; then
  die "ViT-S runner does not expose a recognizable pretrained-checkpoint variable: $VITS_SCRIPT"
fi

if [ "$CHECK_ONLY" != "1" ]; then
  echo "===== GPU $R50_GPUS / $VITS_GPUS preflight ====="
  gpu_is_ready "$R50_GPUS" || die "R50 GPUs do not meet MIN_FREE_MIB=$MIN_FREE_MIB"
  gpu_is_ready "$VITS_GPUS" || die "ViT-S GPUs do not meet MIN_FREE_MIB=$MIN_FREE_MIB"
fi

mkdir -p "$OUTPUT_ROOT"
printf 'field\tvalue\n' > "$PROVENANCE_TSV"
printf 'source\tbackbone\tstatus\tbest_bbox_mAP\tbest_bbox_mAP_50\tbest_bbox_mAP_75\toutput_root\n' > "$RESULTS_TSV"

printf 'dfui_root\t%s\nruod_root\t%s\npretrain_root\t%s\n' \
  "$DFUI_ROOT" "$RUOD_ROOT" "$PRETRAIN_ROOT" >> "$PROVENANCE_TSV"
printf 'dfui_epochs\t%s\nruod_epochs\t%s\ndfui_lr\t%s\nruod_lr\t%s\n' \
  "$DFUI_EPOCHS" "$RUOD_EPOCHS" "$DFUI_LR" "$RUOD_LR" >> "$PROVENANCE_TSV"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "===== CHECK_ONLY manifest ====="
  cat "$MANIFEST"
  while IFS='|' read -r name backbone checkpoint; do
    [ -z "$name" ] && continue
    [[ "$name" == \#* ]] && continue
    require_file "$checkpoint"
    case "$backbone" in
      r50) require_file "$R50_SCRIPT" ;;
      vits) require_file "$VITS_SCRIPT" ;;
      *) die "unknown backbone '$backbone' for $name" ;;
    esac
    echo "OK $name backbone=$backbone checkpoint=$checkpoint"
  done < "$MANIFEST"
  echo "CHECK_ONLY=1 passed; no training was started"
  exit 0
fi

run_one() {
  local name="$1"
  local backbone="$2"
  local checkpoint="$3"
  local output="$OUTPUT_ROOT/$name"
  local log="$OUTPUT_ROOT/logs/${name}.log"
  local gpu_group
  local runner
  local runner_args

  require_file "$checkpoint"
  mkdir -p "$output/logs"

  if [ "$backbone" = "r50" ]; then
    gpu_group="$R50_GPUS"
    runner="$R50_SCRIPT"
    runner_args="$R50_ARGS"
  else
    gpu_group="$VITS_GPUS"
    runner="$VITS_SCRIPT"
    runner_args="$VITS_ARGS"
  fi

  local port="$BASE_PORT"
  if [ "$backbone" = "vits" ]; then
    port=$((BASE_PORT + 100))
  fi
  port_is_free "$port" || die "port $port is already occupied for $name"

  {
    echo "START name=$name backbone=$backbone"
    echo "checkpoint=$checkpoint"
    echo "dfui_root=$DFUI_ROOT"
    echo "ruod_root=$RUOD_ROOT"
    echo "gpu_group=$gpu_group"
    echo "output_root=$output"
    echo "dfui_epochs=$DFUI_EPOCHS ruod_epochs=$RUOD_EPOCHS"
    echo "dfui_lr=$DFUI_LR ruod_lr=$RUOD_LR"
    echo "runner=$runner"
  } | tee "$log"

  printf 'source\t%s\nbackbone\t%s\ncheckpoint\t%s\ngpu_group\t%s\noutput_root\t%s\n' \
    "$name" "$backbone" "$checkpoint" "$gpu_group" "$output" >> "$PROVENANCE_TSV"

  # These aliases cover the existing runner variants. The runner itself must
  # use one of them when converting/loading the pretrained backbone.
  env \
    DFUI_ROOT="$DFUI_ROOT" \
    DFUI_RUOD_ROOT="$DFUI_ROOT" \
    RUOD_ROOT="$RUOD_ROOT" \
    PRETRAIN_CKPT="$checkpoint" \
    S1_CKPT="$checkpoint" \
    DINO_RAW="$checkpoint" \
    SOURCE_CKPT="$checkpoint" \
    CHECKPOINT="$checkpoint" \
    OUTPUT_ROOT="$output" \
    RUN_ROOT="$output" \
    WORK_ROOT="$output/work_dirs" \
    BACKBONE_ROOT="$output/backbones" \
    CONVERTED_ROOT="$output/converted" \
    LOG_ROOT="$output/logs" \
    R50_GPUS="$gpu_group" \
    VITS_GPUS="$gpu_group" \
    GPU_GROUP="$gpu_group" \
    GPUS="$gpu_group" \
    BASE_PORT="$port" \
    DFUI_EPOCHS="$DFUI_EPOCHS" \
    RUOD_EPOCHS="$RUOD_EPOCHS" \
    DFUI_LR="$DFUI_LR" \
    RUOD_LR="$RUOD_LR" \
    RUN_TEST="$RUN_TEST" \
    CHECK_ONLY=0 \
    bash -c "bash \"$runner\" $runner_args" \
    2>&1 | tee -a "$log"

  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "FAILED name=$name exit=$status" | tee -a "$log"
    printf '%s\t%s\tFAILED\t-\t-\t-\t%s\n' \
      "$name" "$backbone" "$output" >> "$RESULTS_TSV"
    return "$status"
  fi

  local result
  result=$(grep -hE 'Epoch\(test\).*coco/bbox_mAP:|bbox_mAP_copypaste:' "$log" 2>/dev/null | tail -n 1 || true)
  local map ap50 ap75
  map=$(sed -nE 's/.*coco\/bbox_mAP: ([0-9.]+).*/\1/p' <<< "$result")
  ap50=$(sed -nE 's/.*coco\/bbox_mAP_50: ([0-9.]+).*/\1/p' <<< "$result")
  ap75=$(sed -nE 's/.*coco\/bbox_mAP_75: ([0-9.]+).*/\1/p' <<< "$result")

  printf '%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
    "$name" "$backbone" "${map:--}" "${ap50:--}" "${ap75:--}" "$output" >> "$RESULTS_TSV"
  echo "COMPLETE name=$name mAP=${map:--}" | tee -a "$log"
}

echo "DFUI->RUOD six-source serial pipeline" | tee "$MASTER_LOG"
echo "manifest=$MANIFEST" | tee -a "$MASTER_LOG"
echo "output_root=$OUTPUT_ROOT" | tee -a "$MASTER_LOG"

while IFS='|' read -r name backbone checkpoint; do
  [ -z "$name" ] && continue
  [[ "$name" == \#* ]] && continue

  case ",$SOURCES," in
    *,all,*) ;;
    *,"$name",*) ;;
    *) continue ;;
  esac

  run_one "$name" "$backbone" "$checkpoint" >> "$MASTER_LOG" 2>&1 || exit $?
done < "$MANIFEST"

echo "ALL_SELECTED_EXPERIMENTS_COMPLETE" | tee -a "$MASTER_LOG"
echo "RESULTS_TSV=$RESULTS_TSV" | tee -a "$MASTER_LOG"
