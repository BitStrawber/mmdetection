#!/usr/bin/env bash
# Run nested-scale DINO-100e pretraining.  The outer loop is scale, therefore
# all 100K jobs finish before 300K begins, and so on.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SUBSET_ROOT="${SUBSET_ROOT:-/media/SSD1/XCX/exp_2/dino_nested_control}"
RUNNER="${RUNNER:-scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh}"
R50_CONFIG="${R50_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_resnet50_100e.sh}"
VITS_CONFIG="${VITS_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_vits_100e.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-30100}"
WORK_ROOT="${WORK_ROOT:-work_dirs/nested_dino_scales}"
LOG_ROOT="${LOG_ROOT:-logs/nested_dino_scales}"
SCALES="${SCALES:-100k,300k,500k,800k,1m}"
SOURCES="${SOURCES:-imagenet,realuw,synthetic5}"
DINO_EPOCHS="${DINO_EPOCHS:-100}"
DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-50}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# Allow an optional low-utilization memory guard; all thresholds remain
# overridable for fully idle GPU runs.
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-10500}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-100}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-1}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-2}"
# The parent pipeline owns the idle-period reservation controller.  Child
# launchers create this request before their GPU-idle check and remove it after
# their DINO process exits, giving training priority over the reservation.
GPU_YIELD_REQUEST_FILE="${GPU_YIELD_REQUEST_FILE:-$LOG_ROOT/gpu_yield_request}"
IDLE_GPU_OCCUPIER_ENABLED="${IDLE_GPU_OCCUPIER_ENABLED:-1}"
IDLE_GPU_OCCUPY_MB="${IDLE_GPU_OCCUPY_MB:-13312}"
IDLE_GPU_RESERVE_MB="${IDLE_GPU_RESERVE_MB:-2048}"
IDLE_GPU_CHECK_INTERVAL="${IDLE_GPU_CHECK_INTERVAL:-2}"
IDLE_GPU_IDLE_CHECKS="${IDLE_GPU_IDLE_CHECKS:-1}"
IDLE_GPU_OCCUPIER="${IDLE_GPU_OCCUPIER:-scripts/exp_2/tri_pretrain/run_gpu_yielding_occupier.sh}"
IDLE_GPU_OCCUPIER_PID=""

mkdir -p "$WORK_ROOT" "$LOG_ROOT"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

die() { echo "ERROR: $*" >&2; exit 1; }

stop_idle_gpu_occupier() {
  if [[ -n "$IDLE_GPU_OCCUPIER_PID" ]] && kill -0 "$IDLE_GPU_OCCUPIER_PID" 2>/dev/null; then
    echo "Stopping idle GPU occupier (PID $IDLE_GPU_OCCUPIER_PID)."
    kill -TERM "$IDLE_GPU_OCCUPIER_PID" 2>/dev/null || true
    wait "$IDLE_GPU_OCCUPIER_PID" 2>/dev/null || true
  fi
  IDLE_GPU_OCCUPIER_PID=""
}

cleanup_pipeline_resources() {
  rm -f "$GPU_YIELD_REQUEST_FILE"
  stop_idle_gpu_occupier
}

start_idle_gpu_occupier() {
  [[ "$IDLE_GPU_OCCUPIER_ENABLED" == "1" ]] || {
    echo "Idle GPU occupier disabled."
    return
  }

  [[ -f "$IDLE_GPU_OCCUPIER" ]] || die "missing idle GPU occupier: $IDLE_GPU_OCCUPIER"
  rm -f "$GPU_YIELD_REQUEST_FILE"
  local occupier_log_dir="$LOG_ROOT/idle_gpu_occupier"
  mkdir -p "$occupier_log_dir"

  echo "Starting pipeline-managed idle GPU occupier: GPUs=$GPU_IDS occupy=${IDLE_GPU_OCCUPY_MB}MiB reserve=${IDLE_GPU_RESERVE_MB}MiB."
  env \
    OCCUPY_MB="$IDLE_GPU_OCCUPY_MB" \
    RESERVE_MB="$IDLE_GPU_RESERVE_MB" \
    CHECK_INTERVAL="$IDLE_GPU_CHECK_INTERVAL" \
    IDLE_CHECKS="$IDLE_GPU_IDLE_CHECKS" \
    TRAIN_USERS="$(id -un)" \
    YIELD_ON_ANY_USER_COMPUTE=1 \
    TRAIN_MATCH='run_dino_with_index.py|main_dino.py|torchrun' \
    GPU_YIELD_REQUEST_FILE="$GPU_YIELD_REQUEST_FILE" \
    LOG_DIR="$occupier_log_dir" \
    bash "$IDLE_GPU_OCCUPIER" "$GPU_IDS" &
  IDLE_GPU_OCCUPIER_PID=$!
  echo "$IDLE_GPU_OCCUPIER_PID" > "$occupier_log_dir/controller.pid"
  echo "Idle GPU occupier PID=$IDLE_GPU_OCCUPIER_PID log=$occupier_log_dir/controller.log"
}

trap cleanup_pipeline_resources EXIT
trap 'cleanup_pipeline_resources; exit 130' INT
trap 'cleanup_pipeline_resources; exit 143' TERM

expected_images() {
  case "$1" in 100k) echo 100000;; 300k) echo 300000;; 500k) echo 500000;; 800k) echo 800000;; 1m) echo 1000000;; *) die "unknown scale: $1";; esac
}

validate_subset() {
  local source="$1" scale="$2" expected="$3" root="$SUBSET_ROOT/$source/$scale"
  python - "$root/subset_manifest.json" "$source" "$scale" "$expected" <<'PY'
import json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
source, scale, expected = sys.argv[2:]
payload = json.loads(manifest_path.read_text())
if (payload.get('schema_version') != 2 or payload.get('storage_mode') != 'index_only' or
        payload['source'] != source or payload['label'] != scale or
        payload['image_count'] != int(expected)):
    raise SystemExit(f'manifest mismatch: {manifest_path}')
base = Path(payload['base_index_file'])
remaining = Path(payload['remaining_index_file'])
base_count = sum(1 for line in base.open(encoding='utf-8') if line.strip())
remaining_count = sum(1 for line in remaining.open(encoding='utf-8') if line.strip())
if base_count != payload['base_count'] or remaining_count < payload['additional_count']:
    raise SystemExit(f'index length mismatch: {manifest_path}')
if not Path(payload['source_root']).is_dir():
    raise SystemExit(f"source root missing: {payload['source_root']}")
print(f'validated {source}/{scale}: {payload["image_count"]} indexed images; selection={payload["selection_sha256"]}')
PY
}

manifest_source_root() {
  python - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['source_root'])
PY
}

validate_checkpoint() {
  local checkpoint="$1" arch="$2"
  python - "$checkpoint" "$arch" <<'PY'
import sys, torch
path, expected_arch = sys.argv[1:]
ckpt = torch.load(path, map_location='cpu', weights_only=False)
args, teacher = ckpt.get('args'), ckpt.get('teacher')
print(f'checkpoint={path} epoch={ckpt.get("epoch")} arch={getattr(args, "arch", None)} teacher={len(teacher) if isinstance(teacher, dict) else 0}')
if ckpt.get('epoch') != 100 or getattr(args, 'arch', None) != expected_arch or not isinstance(teacher, dict):
    raise SystemExit('incomplete or incompatible DINO checkpoint')
PY
}

run_one() {
  local scale="$1" source="$2" backbone="$3" exp_id="$4" config="$5" arch="$6" port="$7"
  local count name root checkpoint source_root
  count="$(expected_images "$scale")"
  root="$SUBSET_ROOT/$source/$scale"
  source_root="$(manifest_source_root "$root/subset_manifest.json")"
  name="scale${scale}_${source}_dino_${backbone}_100e"
  checkpoint="$WORK_ROOT/$name/checkpoint.pth"
  validate_subset "$source" "$scale" "$count"
  if [ "$SKIP_COMPLETED" = "1" ] && [ -s "$checkpoint" ]; then
    validate_checkpoint "$checkpoint" "$arch"
    echo "REUSE completed: $name"
    return
  fi
  echo "================================================================"
  echo "START $name"
  echo "data_path=$source_root (indexed)  images=$count  gpus=$GPU_IDS"
  echo "gpu_start_rule=memory.used<=${GPU_MAX_MEM_MB}MB util<=${GPU_MAX_UTIL}% checks=${GPU_IDLE_CHECKS} interval=${GPU_WAIT_INTERVAL}s"
  env EXP_ID="$exp_id" TASK_CONFIG="$config" DINO_NAME="$name" \
    DINO_EPOCHS="$DINO_EPOCHS" DINO_BATCH_SIZE_PER_GPU="$DINO_BATCH_SIZE_PER_GPU" \
    DINO_NUM_WORKERS="$DINO_NUM_WORKERS" DINO_SAVECKP_FREQ="$DINO_SAVECKP_FREQ" \
    DINO_INIT_CHECKPOINT= DINO_DATA_PATH="$source_root" \
    DINO_INDEX_MANIFEST="$root/subset_manifest.json" BUILD_REALUW_SSL=0 \
    GPU_IDS="$GPU_IDS" PORT="$port" WORK_ROOT="$WORK_ROOT" LOG_DIR="$LOG_ROOT" \
    WAIT_FOR_GPUS="$WAIT_FOR_GPUS" GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
    GPU_MAX_UTIL="$GPU_MAX_UTIL" GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
    GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
    GPU_YIELD_REQUEST_FILE="$GPU_YIELD_REQUEST_FILE" bash "$RUNNER"
  validate_checkpoint "$checkpoint" "$arch"
}

[ -f "$RUNNER" ] || die "missing runner: $RUNNER"
[ -f "$R50_CONFIG" ] || die "missing R50 config: $R50_CONFIG"
[ -f "$VITS_CONFIG" ] || die "missing ViT-S config: $VITS_CONFIG"

IFS=',' read -r -a scale_list <<< "$SCALES"
IFS=',' read -r -a source_list <<< "$SOURCES"
port="$BASE_PORT"
for scale in "${scale_list[@]}"; do
  for source in "${source_list[@]}"; do
    validate_subset "$source" "$scale" "$(expected_images "$scale")"
  done
done
if [ "$CHECK_ONLY" = "1" ]; then echo "CHECK_ONLY=1 passed. log=$PIPELINE_LOG"; exit 0; fi

start_idle_gpu_occupier

for scale in "${scale_list[@]}"; do
  echo "################ SCALE $scale: all sources, then next scale ################"
  for source in "${source_list[@]}"; do
    run_one "$scale" "$source" resnet50 j7 "$R50_CONFIG" resnet50 "$port"; port=$((port+1))
    run_one "$scale" "$source" vits j14 "$VITS_CONFIG" vit_small "$port"; port=$((port+1))
  done
done
echo "COMPLETE: $PIPELINE_LOG"
