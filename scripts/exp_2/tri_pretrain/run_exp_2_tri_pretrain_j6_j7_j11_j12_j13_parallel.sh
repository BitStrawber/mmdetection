#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
mkdir -p "$LOG_DIR"

# Default GPU allocation uses five 2-GPU experiments. Override any group by
# passing J6_GPU_IDS, J7_GPU_IDS, J11_GPU_IDS, J12_GPU_IDS, or J13_GPU_IDS.
J6_GPU_IDS="${J6_GPU_IDS:-0,1}"
J7_GPU_IDS="${J7_GPU_IDS:-2,3}"
J11_GPU_IDS="${J11_GPU_IDS:-4,5}"
J12_GPU_IDS="${J12_GPU_IDS:-6,7}"
J13_GPU_IDS="${J13_GPU_IDS:-0,1}"

launch() {
    local exp_id="$1"
    local script="$2"
    local gpu_ids="$3"
    local port="$4"
    local log_file="$LOG_DIR/${exp_id}_launcher.log"

    echo "Launching $exp_id on GPU $gpu_ids, port $port"
    (
        GPU_IDS="$gpu_ids" PORT="$port" bash "$script" \
            2>&1 | sed "s/^/[$exp_id] /" | tee "$log_file"
    ) &
    pids+=("$!")
    names+=("$exp_id")
}

pids=()
names=()

launch j6 "$SCRIPT_DIR/run_exp_2_j6.sh" "$J6_GPU_IDS" 29686
launch j7 "$SCRIPT_DIR/run_exp_2_j7.sh" "$J7_GPU_IDS" 29687
launch j11 "$SCRIPT_DIR/run_exp_2_j11.sh" "$J11_GPU_IDS" 29691
launch j12 "$SCRIPT_DIR/run_exp_2_j12.sh" "$J12_GPU_IDS" 29692
launch j13 "$SCRIPT_DIR/run_exp_2_j13.sh" "$J13_GPU_IDS" 29693

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
echo "Tri-pretrain J6/J7/J11/J12/J13 summary"
echo "========================================="
for name in "${names[@]}"; do
    echo "$name"
    echo "  launcher: $LOG_DIR/${name}_launcher.log"
    echo "  det log : $LOG_DIR/${name}_det.log"
    echo "  mask log: $LOG_DIR/${name}_mask.log"
done

exit "$failed"
