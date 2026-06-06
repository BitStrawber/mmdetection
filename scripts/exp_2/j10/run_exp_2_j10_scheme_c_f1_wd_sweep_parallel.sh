#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# J10 scheme C weight-decay sweep.
#
# Fixed from the current best recipe:
#   frozen_stages=1
#   lr=0.00375
#   epochs=48
#   milestones=[32,44]
#   S2 config unchanged
#
# Compared recipes:
#   wd=0.00005
#   wd=0.0001
#   wd=0.0005
#
# Outputs:
#   logs/j10_scheme_c_tuning/
#   work_dirs/j10_scheme_c_tuning/

WORK_DIR=${WORK_DIR:-work_dirs/j10_scheme_c_tuning}
LOG_DIR=${LOG_DIR:-logs/j10_scheme_c_tuning}
FROZEN_STAGES=${FROZEN_STAGES:-1}
S1_LR=${S1_LR:-0.00375}
S1_EPOCHS=${S1_EPOCHS:-48}
S1_MILESTONES=${S1_MILESTONES:-[32,44]}
MAX_KEEP_CKPTS=${MAX_KEEP_CKPTS:-5}
RUN_S2=${RUN_S2:-1}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_MAX_MEM_MB=${GPU_MAX_MEM_MB:-3000}
GPU_MAX_UTIL=${GPU_MAX_UTIL:-10}
GPU_IDLE_CHECKS=${GPU_IDLE_CHECKS:-2}
GPU_WAIT_INTERVAL=${GPU_WAIT_INTERVAL:-30}

if [ -n "${EXP_NAMES:-}" ]; then
    read -r -a EXP_NAMES <<< "$EXP_NAMES"
else
    EXP_NAMES=(j10_scheme_c_f1_lr00375_e48_wd00005 j10_scheme_c_f1_lr00375_e48_wd0001 j10_scheme_c_f1_lr00375_e48_wd0005)
fi
if [ -n "${S1_WEIGHT_DECAYS:-}" ]; then
    read -r -a S1_WEIGHT_DECAYS <<< "$S1_WEIGHT_DECAYS"
else
    S1_WEIGHT_DECAYS=(0.00005 0.0001 0.0005)
fi
if [ -n "${GPU_GROUPS:-}" ]; then
    read -r -a GPU_GROUPS <<< "$GPU_GROUPS"
else
    GPU_GROUPS=(2,3 4,5 6,7)
fi
if [ -n "${PORTS:-}" ]; then
    read -r -a PORTS <<< "$PORTS"
else
    PORTS=(29641 29642 29643)
fi

mkdir -p "$LOG_DIR"

if [ "${#EXP_NAMES[@]}" -ne "${#S1_WEIGHT_DECAYS[@]}" ] || \
   [ "${#EXP_NAMES[@]}" -ne "${#GPU_GROUPS[@]}" ] || \
   [ "${#EXP_NAMES[@]}" -ne "${#PORTS[@]}" ]; then
    echo "Error: EXP_NAMES, S1_WEIGHT_DECAYS, GPU_GROUPS, and PORTS must have the same length."
    echo "EXP_NAMES=${EXP_NAMES[*]}"
    echo "S1_WEIGHT_DECAYS=${S1_WEIGHT_DECAYS[*]}"
    echo "GPU_GROUPS=${GPU_GROUPS[*]}"
    echo "PORTS=${PORTS[*]}"
    exit 1
fi

echo "========================================="
echo "J10 scheme C f1 weight-decay sweep"
echo "========================================="
echo "FROZEN_STAGES: $FROZEN_STAGES"
echo "S1_LR: $S1_LR"
echo "S1_EPOCHS: $S1_EPOCHS"
echo "S1_MILESTONES: $S1_MILESTONES"
echo "RUN_S2: $RUN_S2"
echo "WAIT_FOR_GPUS: $WAIT_FOR_GPUS"
echo "GPU_MAX_MEM_MB: $GPU_MAX_MEM_MB"
echo "GPU_MAX_UTIL: $GPU_MAX_UTIL"
echo "GPU_IDLE_CHECKS: $GPU_IDLE_CHECKS"
echo "GPU_WAIT_INTERVAL: $GPU_WAIT_INTERVAL"
echo "EXP_NAMES: ${EXP_NAMES[*]}"
echo "S1_WEIGHT_DECAYS: ${S1_WEIGHT_DECAYS[*]}"
echo "GPU_GROUPS: ${GPU_GROUPS[*]}"
echo "PORTS: ${PORTS[*]}"
echo "========================================="

pids=()
names=()

for i in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$i]}"
    wd="${S1_WEIGHT_DECAYS[$i]}"
    gpus="${GPU_GROUPS[$i]}"
    port="${PORTS[$i]}"
    launcher_log="$LOG_DIR/${exp_name}_launcher.log"

    echo "Launching $exp_name on GPUs $gpus, port $port"
    echo "  S1_LR=$S1_LR S1_EPOCHS=$S1_EPOCHS S1_MILESTONES=$S1_MILESTONES S1_WEIGHT_DECAY=$wd"
    (
        EXP_NAME="$exp_name" \
        GPU_IDS="$gpus" \
        PORT="$port" \
        FROZEN_STAGES="$FROZEN_STAGES" \
        S1_LR="$S1_LR" \
        S1_EPOCHS="$S1_EPOCHS" \
        S1_MILESTONES="$S1_MILESTONES" \
        S1_WEIGHT_DECAY="$wd" \
        MAX_KEEP_CKPTS="$MAX_KEEP_CKPTS" \
        RUN_S2="$RUN_S2" \
        WAIT_FOR_GPUS="$WAIT_FOR_GPUS" \
        GPU_MAX_MEM_MB="$GPU_MAX_MEM_MB" \
        GPU_MAX_UTIL="$GPU_MAX_UTIL" \
        GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" \
        GPU_WAIT_INTERVAL="$GPU_WAIT_INTERVAL" \
        WORK_DIR="$WORK_DIR" \
        LOG_DIR="$LOG_DIR" \
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
echo "Weight-decay sweep summary"
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
