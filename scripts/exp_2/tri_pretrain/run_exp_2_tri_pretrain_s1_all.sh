#!/bin/bash
set -euo pipefail

# Sequentially run all Tri-pretrain S1 tasks on the same GPU set.
# Override TASKS to run a subset, for example:
#   TASKS="j11 j13" GPU_IDS=0,1,2,3,4,5,6,7 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="${TASKS:-j6 j7 j11 j12 j13}"
BUILD_FIRST="${BUILD_FIRST:-1}"

first=1
for task in $TASKS; do
    if [ "$first" = "1" ]; then
        export BUILD_REALUW_SSL="$BUILD_FIRST"
        first=0
    else
        export BUILD_REALUW_SSL=0
    fi
    export EXP_ID="$task"
    unset CONFIG
    case "$task" in
        j11)
            [ -n "${J11_CONFIG:-}" ] && export CONFIG="$J11_CONFIG"
            ;;
        j12)
            [ -n "${J12_CONFIG:-}" ] && export CONFIG="$J12_CONFIG"
            ;;
        j13)
            [ -n "${J13_CONFIG:-}" ] && export CONFIG="$J13_CONFIG"
            ;;
    esac
    bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
done
