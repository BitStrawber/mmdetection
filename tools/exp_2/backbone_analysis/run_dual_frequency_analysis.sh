#!/usr/bin/env bash
set -euo pipefail

# Run identical analysis products for fixed-scale and dataset-energy bands.
# Required dataset/model variables are forwarded to run_pipeline.sh.

BASE_OUT_ROOT="${BASE_OUT_ROOT:?Set BASE_OUT_ROOT}"
PIPELINE="${PIPELINE:-tools/exp_2/backbone_analysis/run_pipeline.sh}"

common_env=(
  RUOD_ANN="${RUOD_ANN:?Set RUOD_ANN}"
  RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:?Set RUOD_IMAGE_ROOT}"
  CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-}"
  SAMPLES="${SAMPLES:-50}"
  SEED="${SEED:-2026}"
  DEVICE="${DEVICE:-cuda:0}"
  ANALYSIS_MODELS="${ANALYSIS_MODELS:-imagenet_backbone,cascade_ruod_backbone}"
  CKA_Y_MODEL="${CKA_Y_MODEL:-${CKA_MODEL_A:-imagenet_backbone}}"
  CKA_Y_MODELS="${CKA_Y_MODELS:-${CKA_Y_MODEL:-${CKA_MODEL_A:-imagenet_backbone}}}"
  CKA_X_MODEL="${CKA_X_MODEL:-${CKA_MODEL_B:-cascade_ruod_backbone}}"
  VARIANTS="clean,low,mid,high,remove_low,remove_mid,remove_high"
  FREQUENCY_RESPONSE_VARIANTS="low,mid,high,remove_low,remove_mid,remove_high"
  FREQUENCY_MODEL_INPUT_MODE="${FREQUENCY_MODEL_INPUT_MODE:-natural-energy}"
  RUN_FREQUENCY_ACTIVATION="${RUN_FREQUENCY_ACTIVATION:-1}"
  RUN_FREQUENCY_INPUT_VISUALS="${RUN_FREQUENCY_INPUT_VISUALS:-1}"
  RUN_FREQUENCY_FIGURES="${RUN_FREQUENCY_FIGURES:-1}"
  RUN_DETECTION_FREQUENCY_EVAL="${RUN_DETECTION_FREQUENCY_EVAL:-0}"
  RUN_FOURIER_SENSITIVITY="${RUN_FOURIER_SENSITIVITY:-0}"
  OVERWRITE="${OVERWRITE:-0}"
)

if [[ -n "${MODELS_CONFIG_INPUT:-}" ]]; then
  common_env+=(MODELS_CONFIG_INPUT="${MODELS_CONFIG_INPUT}")
fi

echo "===== fixed-scale frequency analysis ====="
env "${common_env[@]}" \
  OUT_ROOT="${BASE_OUT_ROOT}/fixed" \
  FREQUENCY_BAND_POLICY=fixed \
  BANDS="${FIXED_BANDS:-low:0:1/32,mid:1/32:1/8,high:1/8:max}" \
  bash "${PIPELINE}"

adaptive_env=(
  OUT_ROOT="${BASE_OUT_ROOT}/dataset_energy"
  FREQUENCY_BAND_POLICY=dataset-energy
  FREQUENCY_ENERGY_QUANTILES="${FREQUENCY_ENERGY_QUANTILES:-1/3,2/3}"
  FREQUENCY_ENERGY_BINS="${FREQUENCY_ENERGY_BINS:-1024}"
  FREQUENCY_ENERGY_COLOR_SPACE="${FREQUENCY_ENERGY_COLOR_SPACE:-rgb}"
)
if [[ -n "${FREQUENCY_CALIBRATION_MANIFEST:-}" ]]; then
  adaptive_env+=(
    FREQUENCY_CALIBRATION_MANIFEST="${FREQUENCY_CALIBRATION_MANIFEST}")
fi

echo "===== dataset-energy frequency analysis ====="
env "${common_env[@]}" "${adaptive_env[@]}" bash "${PIPELINE}"

echo "Dual frequency analysis completed: ${BASE_OUT_ROOT}"
