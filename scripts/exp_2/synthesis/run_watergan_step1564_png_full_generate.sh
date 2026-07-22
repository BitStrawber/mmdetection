#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTH_INPUT_MODE=png \
RUN_SMOKE="${RUN_SMOKE:-1}" \
  exec bash "${SCRIPT_DIR}/run_watergan_step1564_official_mat_full_generate.sh" "$@"
