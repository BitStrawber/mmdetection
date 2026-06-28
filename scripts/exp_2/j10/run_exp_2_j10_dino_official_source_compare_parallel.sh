#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# J10 source-comparison with the official ImageNet DINO ResNet-50 backbone.
#
# Three parallel pipelines:
#   1) S1 DFUI_ALL                         -> S2 RUOD
#   2) S1 DFUI_RUOD_EASY                   -> S2 RUOD
#   3) S1 DFUI_RUOD_UIIS_EASY              -> S2 RUOD
#
# Each pipeline reuses run_exp_2_j10_scheme_c.sh:
#   S1 Cascade R-CNN supervised source adaptation
#   extract backbone_only.pth
#   S2 unchanged RUOD config loading the S1 backbone only

PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-work_dirs/j10_dino_official_source_compare}"
LOG_DIR="${LOG_DIR:-logs/j10_dino_official_source_compare}"

PRETRAIN_DIR="${PRETRAIN_DIR:-../pretrained_weights}"
DINO_URL="${DINO_URL:-https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_checkpoint.pth}"
DINO_RAW_CKPT="${DINO_RAW_CKPT:-$PRETRAIN_DIR/dino_rn50_checkpoint.pth}"
DINO_BACKBONE_CKPT="${DINO_BACKBONE_CKPT:-$PRETRAIN_DIR/dino_rn50_official_100e_backbone.pth}"

BUILD_S1_DATASETS="${BUILD_S1_DATASETS:-1}"
OVERWRITE_S1_DATASETS="${OVERWRITE_S1_DATASETS:-0}"
RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"

DFUI_ALL_ROOT="${DFUI_ALL_ROOT:-/media/HDD0/XCX/exp_2/DFUI_ALL}"
DFUI_RUOD_EASY_ROOT="${DFUI_RUOD_EASY_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_EASY}"
DFUI_RUOD_UIIS_EASY_ROOT="${DFUI_RUOD_UIIS_EASY_ROOT:-/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_EASY}"

DFUI_IMG_DIR="${DFUI_IMG_DIR:-/media/HDD0/XCX/exp_2/dfui/images}"
DFUI_ANN="${DFUI_ANN:-/media/HDD0/XCX/exp_2/dfui/annotations/instances_train2017.json /media/HDD0/XCX/exp_2/dfui/annotations/instances_val2017.json /media/HDD0/XCX/exp_2/dfui/annotations/instances_test2017.json}"
RUOD_EASY_IMG_DIR="${RUOD_EASY_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
RUOD_EASY_ANN="${RUOD_EASY_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json}"
UIIS_EASY_IMG_DIR="${UIIS_EASY_IMG_DIR:-/media/HDD0/XCX/exp_2/UIIS10K/img}"
UIIS_EASY_ANN="${UIIS_EASY_ANN:-/media/HDD0/XCX/exp_2/UIIS10K/coco/annotations/cross_split_det/easy_merged.json}"

S2_CONFIG="${S2_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py}"

TASK_NAMES=(${TASK_NAMES:-dfui_all dfui_ruod_easy dfui_ruod_uiis_easy})
S1_CONFIGS=(${S1_CONFIGS:-configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_dfui_all_j10_scheme_c_s1.py configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_easy_j10_scheme_c_s1.py configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py})
GPU_GROUPS=(${GPU_GROUPS:-2,3 4,5 6,7})
PORTS=(${PORTS:-29731 29732 29733})

FROZEN_STAGES="${FROZEN_STAGES:-2}"
S1_LR="${S1_LR:-0.001875}"
S1_EPOCHS="${S1_EPOCHS:-48}"
S1_MILESTONES="${S1_MILESTONES:-[32,44]}"
S1_WEIGHT_DECAY="${S1_WEIGHT_DECAY:-0.0001}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
RUN_S2="${RUN_S2:-1}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

mkdir -p "$PRETRAIN_DIR" "$WORK_DIR" "$LOG_DIR"

if [ "${#TASK_NAMES[@]}" -ne "${#S1_CONFIGS[@]}" ] || \
   [ "${#TASK_NAMES[@]}" -ne "${#GPU_GROUPS[@]}" ] || \
   [ "${#TASK_NAMES[@]}" -ne "${#PORTS[@]}" ]; then
    echo "Error: TASK_NAMES, S1_CONFIGS, GPU_GROUPS and PORTS must have the same length."
    echo "TASK_NAMES=${TASK_NAMES[*]}"
    echo "S1_CONFIGS=${S1_CONFIGS[*]}"
    echo "GPU_GROUPS=${GPU_GROUPS[*]}"
    echo "PORTS=${PORTS[*]}"
    exit 1
fi

download_dino() {
    if [ "$RUN_DOWNLOAD" != "1" ]; then
        echo "RUN_DOWNLOAD=$RUN_DOWNLOAD, skip DINO checkpoint download."
        return
    fi
    if [ -f "$DINO_RAW_CKPT" ]; then
        echo "DINO raw checkpoint exists: $DINO_RAW_CKPT"
        return
    fi
    echo "Download DINO official ResNet-50 checkpoint:"
    echo "  url: $DINO_URL"
    echo "  out: $DINO_RAW_CKPT"
    if command -v wget >/dev/null 2>&1; then
        wget -c --show-progress -O "$DINO_RAW_CKPT" "$DINO_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$DINO_RAW_CKPT" "$DINO_URL"
    else
        "$PYTHON" - <<PY
from urllib.request import urlretrieve
urlretrieve("$DINO_URL", "$DINO_RAW_CKPT")
PY
    fi
}

convert_dino() {
    if [ "$RUN_CONVERT" != "1" ]; then
        echo "RUN_CONVERT=$RUN_CONVERT, skip DINO checkpoint conversion."
        return
    fi
    if [ ! -f "$DINO_RAW_CKPT" ]; then
        echo "Error: missing DINO raw checkpoint: $DINO_RAW_CKPT" >&2
        exit 1
    fi
    if [ "$FORCE_CONVERT" != "1" ] && [ -f "$DINO_BACKBONE_CKPT" ]; then
        echo "Converted DINO backbone exists: $DINO_BACKBONE_CKPT"
        return
    fi
    echo "Convert DINO official checkpoint to MMDet backbone init checkpoint:"
    echo "  raw     : $DINO_RAW_CKPT"
    echo "  backbone: $DINO_BACKBONE_CKPT"
    "$PYTHON" tools/convert_ssl_backbone_to_mmdet.py \
        --checkpoint "$DINO_RAW_CKPT" \
        --source teacher \
        --prepend "" \
        --out "$DINO_BACKBONE_CKPT" \
        2>&1 | tee "$LOG_DIR/dino_rn50_official_convert_backbone.log"
}

dataset_exists() {
    local root="$1"
    [ -f "$root/annotations/instances_train.json" ] && \
    [ -f "$root/annotations/instances_val.json" ] && \
    [ -d "$root/images" ]
}

build_source_datasets() {
    if [ "$BUILD_S1_DATASETS" != "1" ]; then
        echo "BUILD_S1_DATASETS=$BUILD_S1_DATASETS, skip S1 dataset building."
        return
    fi

    local overwrite_arg=()
    if [ "$OVERWRITE_S1_DATASETS" = "1" ]; then
        overwrite_arg=(--overwrite)
    fi

    read -r -a dfui_ann_args <<< "$DFUI_ANN"

    if dataset_exists "$DFUI_ALL_ROOT" && [ "$OVERWRITE_S1_DATASETS" != "1" ]; then
        echo "DFUI_ALL exists, skip build: $DFUI_ALL_ROOT"
    else
        "$PYTHON" tools/merge_dfui_sources_easy.py \
            --dfui-img-dir "$DFUI_IMG_DIR" \
            --dfui-ann "${dfui_ann_args[@]}" \
            --out-root "$DFUI_ALL_ROOT" \
            "${overwrite_arg[@]}" \
            2>&1 | tee "$LOG_DIR/build_dfui_all.log"
    fi

    if dataset_exists "$DFUI_RUOD_EASY_ROOT" && [ "$OVERWRITE_S1_DATASETS" != "1" ]; then
        echo "DFUI_RUOD_EASY exists, skip build: $DFUI_RUOD_EASY_ROOT"
    else
        "$PYTHON" tools/merge_dfui_sources_easy.py \
            --dfui-img-dir "$DFUI_IMG_DIR" \
            --dfui-ann "${dfui_ann_args[@]}" \
            --include-ruod-easy \
            --ruod-easy-img-dir "$RUOD_EASY_IMG_DIR" \
            --ruod-easy-ann "$RUOD_EASY_ANN" \
            --out-root "$DFUI_RUOD_EASY_ROOT" \
            "${overwrite_arg[@]}" \
            2>&1 | tee "$LOG_DIR/build_dfui_ruod_easy.log"
    fi

    if dataset_exists "$DFUI_RUOD_UIIS_EASY_ROOT" && [ "$OVERWRITE_S1_DATASETS" != "1" ]; then
        echo "DFUI_RUOD_UIIS_EASY exists, skip build: $DFUI_RUOD_UIIS_EASY_ROOT"
    else
        "$PYTHON" tools/merge_dfui_sources_easy.py \
            --dfui-img-dir "$DFUI_IMG_DIR" \
            --dfui-ann "${dfui_ann_args[@]}" \
            --include-ruod-easy \
            --ruod-easy-img-dir "$RUOD_EASY_IMG_DIR" \
            --ruod-easy-ann "$RUOD_EASY_ANN" \
            --include-uiis-easy \
            --uiis-easy-img-dir "$UIIS_EASY_IMG_DIR" \
            --uiis-easy-ann "$UIIS_EASY_ANN" \
            --out-root "$DFUI_RUOD_UIIS_EASY_ROOT" \
            "${overwrite_arg[@]}" \
            2>&1 | tee "$LOG_DIR/build_dfui_ruod_uiis_easy.log"
    fi
}

echo "========================================="
echo "J10 DINO official source comparison"
echo "========================================="
echo "WORK_DIR: $WORK_DIR"
echo "LOG_DIR: $LOG_DIR"
echo "DINO_RAW_CKPT: $DINO_RAW_CKPT"
echo "DINO_BACKBONE_CKPT: $DINO_BACKBONE_CKPT"
echo "TASK_NAMES: ${TASK_NAMES[*]}"
echo "GPU_GROUPS: ${GPU_GROUPS[*]}"
echo "PORTS: ${PORTS[*]}"
echo "FROZEN_STAGES: $FROZEN_STAGES"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "S1_WEIGHT_DECAY: $S1_WEIGHT_DECAY"
echo "RUN_S2: $RUN_S2"
echo "WAIT_FOR_GPUS: $WAIT_FOR_GPUS"
echo "========================================="

download_dino
convert_dino
build_source_datasets

pids=()
names=()

for i in "${!TASK_NAMES[@]}"; do
    task="${TASK_NAMES[$i]}"
    s1_config="${S1_CONFIGS[$i]}"
    gpus="${GPU_GROUPS[$i]}"
    port="${PORTS[$i]}"
    exp_name="j10_dino_official_${task}_f${FROZEN_STAGES}_lr${S1_LR}_e${S1_EPOCHS}"
    exp_name="${exp_name//./}"
    launcher_log="$LOG_DIR/${exp_name}_launcher.log"

    echo "Launching $exp_name on GPUs $gpus, port $port"
    (
        WORK_DIR="$WORK_DIR" \
        LOG_DIR="$LOG_DIR" \
        EXP_NAME="$exp_name" \
        GPU_IDS="$gpus" \
        PORT="$port" \
        S1_CONFIG="$s1_config" \
        S2_CONFIG="$S2_CONFIG" \
        FROZEN_STAGES="$FROZEN_STAGES" \
        S1_LR="$S1_LR" \
        S1_EPOCHS="$S1_EPOCHS" \
        S1_MILESTONES="$S1_MILESTONES" \
        S1_WEIGHT_DECAY="$S1_WEIGHT_DECAY" \
        MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
        RUN_S2="$RUN_S2" \
        WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
        GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        bash "$SCRIPT_DIR/run_exp_2_j10_scheme_c.sh" \
            2>&1 | sed "s/^/[$exp_name] /" | tee "$launcher_log"
    ) &

    pids+=($!)
    names+=("$exp_name")
done

failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    name="${names[$i]}"
    if wait "$pid"; then
        echo "[$name] finished successfully"
    else
        echo "[$name] failed"
        failed=1
    fi
done

echo "========================================="
echo "J10 DINO official source-comparison summary"
echo "========================================="
for name in "${names[@]}"; do
    echo "$name"
    echo "  launcher: $LOG_DIR/${name}_launcher.log"
    echo "  s1 log  : $LOG_DIR/${name}_s1.log"
    echo "  s2 log  : $LOG_DIR/${name}_s2.log"
    echo "  s1 dir  : $WORK_DIR/${name}_s1"
    echo "  s2 dir  : $WORK_DIR/${name}_s2"
done

exit "$failed"
