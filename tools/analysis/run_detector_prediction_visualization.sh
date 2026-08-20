#!/usr/bin/env bash
# Render detector predictions for a shared RUOD manifest in two visual styles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SAMPLE_ROOT="${SAMPLE_ROOT:?SAMPLE_ROOT must contain manifest.jsonl}"
MODELS_CONFIG="${MODELS_CONFIG:-${SCRIPT_DIR}/models.cam.example.json}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
MODELS="${MODELS:-}"
DEVICES="${DEVICES:-cuda:0}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.30}"
MAX_DETECTIONS="${MAX_DETECTIONS:-30}"
MINIMUM_BOX_AREA="${MINIMUM_BOX_AREA:-4.0}"
LINE_WIDTH="${LINE_WIDTH:-0}"
FONT_SCALE="${FONT_SCALE:-0.032}"
FONT_MIN_SIZE="${FONT_MIN_SIZE:-12}"
FONT_MAX_SIZE="${FONT_MAX_SIZE:-24}"
INCLUDE_SCORE="${INCLUDE_SCORE:-1}"
PANEL_TILE_WIDTH="${PANEL_TILE_WIDTH:-640}"
PANEL_TILE_HEIGHT="${PANEL_TILE_HEIGHT:-480}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-3}"
OVERWRITE="${OVERWRITE:-0}"

[[ -s "${SAMPLE_ROOT}/manifest.jsonl" ]] || { echo "Error: missing ${SAMPLE_ROOT}/manifest.jsonl" >&2; exit 1; }
[[ -s "${MODELS_CONFIG}" ]] || { echo "Error: missing ${MODELS_CONFIG}" >&2; exit 1; }
[[ "${PARALLEL_MODELS}" =~ ^[1-9][0-9]*$ ]] || { echo 'Error: PARALLEL_MODELS must be positive' >&2; exit 1; }

mapfile -t SELECTED_MODELS < <(
  python - "${MODELS_CONFIG}" "${MODELS}" <<'PY'
import json
import sys

config_path, requested = sys.argv[1:]
selected = {item.strip() for item in requested.split(',') if item.strip()}
with open(config_path, encoding='utf-8') as handle:
    specs = json.load(handle).get('models', [])
models = [str(spec['id']) for spec in specs if not selected or str(spec['id']) in selected]
if not models:
    raise SystemExit('No selected models in configuration')
print('\n'.join(models))
PY
)

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
[[ ${#DEVICE_LIST[@]} -gt 0 ]] || { echo 'Error: DEVICES is empty' >&2; exit 1; }

mkdir -p "${OUT_ROOT}/logs"

echo '============================================================'
echo 'RUOD detector prediction visualization'
echo '============================================================'
echo "Sample:          ${SAMPLE_ROOT}/manifest.jsonl"
echo "Models config:   ${MODELS_CONFIG}"
echo "Models:          ${SELECTED_MODELS[*]}"
echo "Devices:         ${DEVICES}"
echo "Parallel models: ${PARALLEL_MODELS}"
echo "Score threshold: ${SCORE_THRESHOLD}"
echo "Max detections:  ${MAX_DETECTIONS}"
echo "Output:          ${OUT_ROOT}"
echo 'Styles:          uniform yellow; detector-specific colors'
echo '============================================================'

run_worker() {
  local model_id="$1"
  local device="$2"
  local worker_log="${OUT_ROOT}/logs/${model_id}.log"
  local score_flag=(--include-score)
  [[ "${INCLUDE_SCORE}" == '0' ]] && score_flag=(--no-include-score)
  local overwrite_flag=()
  [[ "${OVERWRITE}" == '1' ]] && overwrite_flag=(--overwrite)
  echo "START model=${model_id} device=${device} log=${worker_log}"
  python -m tools.analysis.visualize_detector_predictions \
    --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
    --models-config "${MODELS_CONFIG}" \
    --models "${model_id}" \
    --out-dir "${OUT_ROOT}" \
    --device "${device}" \
    --color-modes uniform,model \
    --score-threshold "${SCORE_THRESHOLD}" \
    --max-detections "${MAX_DETECTIONS}" \
    --minimum-box-area "${MINIMUM_BOX_AREA}" \
    --line-width "${LINE_WIDTH}" \
    --font-scale "${FONT_SCALE}" \
    --font-min-size "${FONT_MIN_SIZE}" \
    --font-max-size "${FONT_MAX_SIZE}" \
    --panel-tile-width "${PANEL_TILE_WIDTH}" \
    --panel-tile-height "${PANEL_TILE_HEIGHT}" \
    --png-compress-level "${PNG_COMPRESS_LEVEL}" \
    "${score_flag[@]}" \
    "${overwrite_flag[@]}" \
    > "${worker_log}" 2>&1
}

pids=()
for index in "${!SELECTED_MODELS[@]}"; do
  while (( ${#pids[@]} >= PARALLEL_MODELS )); do
    pid="${pids[0]}"
    if ! wait "${pid}"; then
      echo "FAILED worker pid=${pid}" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
  done
  model_id="${SELECTED_MODELS[$index]}"
  device="${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}"
  run_worker "${model_id}" "${device}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

overwrite_flag=()
[[ "${OVERWRITE}" == '1' ]] && overwrite_flag=(--overwrite)
python -m tools.analysis.visualize_detector_predictions \
  --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
  --models-config "${MODELS_CONFIG}" \
  --models "${MODELS}" \
  --out-dir "${OUT_ROOT}" \
  --color-modes uniform,model \
  --panel-tile-width "${PANEL_TILE_WIDTH}" \
  --panel-tile-height "${PANEL_TILE_HEIGHT}" \
  --png-compress-level "${PNG_COMPRESS_LEVEL}" \
  --compose-panels \
  "${overwrite_flag[@]}"

python - "${OUT_ROOT}" "${SAMPLE_ROOT}" "${MODELS_CONFIG}" "${SCORE_THRESHOLD}" "${MAX_DETECTIONS}" "${DEVICES}" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
payload = {
    'status': 'complete',
    'sample_root': sys.argv[2],
    'models_config': sys.argv[3],
    'score_threshold': float(sys.argv[4]),
    'max_detections': int(sys.argv[5]),
    'devices': sys.argv[6],
    'visual_styles': {
        'uniform': 'all detector boxes use the same yellow color',
        'model': 'each detector uses its stable model-specific color',
    },
    'outputs': {
        'individual': 'one annotated PNG for each model and image',
        'panels_2x2': 'one four-detector comparison panel for each image and style',
    },
}
with (out_root / 'COMPLETE.json').open('w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY

echo '============================================================'
echo "Detector visualization complete: ${OUT_ROOT}"
echo "Uniform panels: ${OUT_ROOT}/uniform/panels_2x2"
echo "Model-color panels: ${OUT_ROOT}/model/panels_2x2"
PY
