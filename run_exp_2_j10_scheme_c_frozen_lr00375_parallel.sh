#!/bin/bash
set -euo pipefail

# J10 scheme C frozen-stage sweep.
#
# Backbone route:
#   S1: DFUI+RUOD-easy+UIIS-easy supervised backbone adaptation.
#   S2: unchanged RUOD config, loading only the S1 backbone checkpoint.
#
# This sweep fixes S1 LR at 0.00375 and compares frozen_stages=1/2/3.
#
# Default GPU allocation:
#   frozen=1 -> GPU 2,3
#   frozen=2 -> GPU 4,5
#   frozen=3 -> GPU 6,7
#
# Common overrides:
#   S1_EPOCHS=24 S1_MILESTONES='[16,22]' bash run_exp_2_j10_scheme_c_frozen_lr00375_parallel.sh
#   RUN_S2=0 bash run_exp_2_j10_scheme_c_frozen_lr00375_parallel.sh

WORK_DIR=${WORK_DIR:-work_dirs}
LOG_DIR=${LOG_DIR:-logs}
S1_LR=${S1_LR:-0.00375}
S1_EPOCHS=${S1_EPOCHS:-48}
S1_MILESTONES=${S1_MILESTONES:-[32,44]}
S1_WEIGHT_DECAY=${S1_WEIGHT_DECAY:-0.0001}
MAX_KEEP_CKPTS=${MAX_KEEP_CKPTS:-5}
RUN_S2=${RUN_S2:-1}

FROZEN_STAGES_LIST=(${FROZEN_STAGES_LIST:-1 2 3})
GPU_GROUPS=(${GPU_GROUPS:-2,3 4,5 6,7})
PORTS=(${PORTS:-29621 29622 29623})

mkdir -p "$LOG_DIR"

if [ "${#FROZEN_STAGES_LIST[@]}" -ne "${#GPU_GROUPS[@]}" ] || \
   [ "${#FROZEN_STAGES_LIST[@]}" -ne "${#PORTS[@]}" ]; then
    echo "Error: FROZEN_STAGES_LIST, GPU_GROUPS, and PORTS must have the same length."
    echo "FROZEN_STAGES_LIST=${FROZEN_STAGES_LIST[*]}"
    echo "GPU_GROUPS=${GPU_GROUPS[*]}"
    echo "PORTS=${PORTS[*]}"
    exit 1
fi

echo "========================================="
echo "J10 scheme C frozen-stage sweep"
echo "========================================="
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "S1_WEIGHT_DECAY: $S1_WEIGHT_DECAY"
echo "RUN_S2: $RUN_S2"
echo "FROZEN_STAGES_LIST: ${FROZEN_STAGES_LIST[*]}"
echo "GPU_GROUPS: ${GPU_GROUPS[*]}"
echo "PORTS: ${PORTS[*]}"
echo "========================================="

pids=()
names=()

for i in "${!FROZEN_STAGES_LIST[@]}"; do
    frozen="${FROZEN_STAGES_LIST[$i]}"
    gpus="${GPU_GROUPS[$i]}"
    port="${PORTS[$i]}"
    exp_name="j10_scheme_c_f${frozen}_lr00375_e${S1_EPOCHS}"
    launcher_log="$LOG_DIR/${exp_name}_launcher.log"

    echo "Launching $exp_name on GPUs $gpus, port $port"
    (
        EXP_NAME="$exp_name" \
        GPU_IDS="$gpus" \
        PORT="$port" \
        FROZEN_STAGES="$frozen" \
        S1_LR="$S1_LR" \
        S1_EPOCHS="$S1_EPOCHS" \
        S1_MILESTONES="$S1_MILESTONES" \
        S1_WEIGHT_DECAY="$S1_WEIGHT_DECAY" \
        MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
        RUN_S2="$RUN_S2" \
        bash run_exp_2_j10_scheme_c.sh \
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
echo "Frozen-stage sweep summary"
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
