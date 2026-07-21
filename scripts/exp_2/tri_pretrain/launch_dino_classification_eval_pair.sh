#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_dino_classification_eval_pair.sh"

PRESET="${PRESET:-imagenet}"
EXP_PREFIX="${EXP_PREFIX:-$PRESET}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/tri_pretrain/classification}"
RESET_LAUNCH_LOG="${RESET_LAUNCH_LOG:-0}"

mkdir -p "$LOG_ROOT"

LAUNCH_LOG="$LOG_ROOT/${EXP_PREFIX}_classification_launcher.log"
PID_FILE="$LOG_ROOT/${EXP_PREFIX}_classification_launcher.pid"

if [ ! -f "$RUNNER" ]; then
    echo "Error: runner is missing: $RUNNER" >&2
    exit 1
fi

if [ -s "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Error: $EXP_PREFIX classification launcher is already running" >&2
        echo "PID: $old_pid" >&2
        echo "Log: $LAUNCH_LOG" >&2
        exit 1
    fi
fi

if [ "$RESET_LAUNCH_LOG" = "1" ] && [ -f "$LAUNCH_LOG" ]; then
    mv "$LAUNCH_LOG" "${LAUNCH_LOG}.before_restart_$(date +%Y%m%d_%H%M%S)"
fi

nohup env \
    PRESET="$PRESET" \
    EXP_PREFIX="$EXP_PREFIX" \
    LOG_ROOT="$LOG_ROOT" \
    bash "$RUNNER" \
    >> "$LAUNCH_LOG" 2>&1 &

launcher_pid=$!
echo "$launcher_pid" > "$PID_FILE"

echo "DINO classification evaluation launched"
echo "PRESET:     $PRESET"
echo "EXP_PREFIX: $EXP_PREFIX"
echo "PID:        $launcher_pid"
echo "LOG:        $LAUNCH_LOG"
echo "PID_FILE:   $PID_FILE"
