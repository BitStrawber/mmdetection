#!/usr/bin/env bash
set -euo pipefail

# Modular ImageNet-backbone vs RUOD-Cascade-backbone analysis pipeline.
# Every stage can be disabled and run independently with RUN_* variables.

RUOD_ANN="${RUOD_ANN:?Set RUOD_ANN to a COCO annotation JSON}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:?Set RUOD_IMAGE_ROOT to the image directory}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-}"
CASCADE_CONFIG="${CASCADE_CONFIG:-configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
MODELS_CONFIG_INPUT="${MODELS_CONFIG_INPUT:-}"

SAMPLES="${SAMPLES:-50}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda:0}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
BANDS="${BANDS:-low:0:1/32,mid:1/32:1/8,high:1/8:max}"
VARIANTS="${VARIANTS:-clean,low,mid,high,remove_low,remove_mid,remove_high}"
FREQUENCY_RESPONSE_VARIANTS="${FREQUENCY_RESPONSE_VARIANTS:-low,mid,high,remove_low,remove_mid,remove_high}"
FREQUENCY_METHOD="${FREQUENCY_METHOD:-soft-cpp}"
FREQUENCY_BAND_POLICY="${FREQUENCY_BAND_POLICY:-fixed}"
FREQUENCY_ENERGY_QUANTILES="${FREQUENCY_ENERGY_QUANTILES:-1/3,2/3}"
FREQUENCY_ENERGY_BINS="${FREQUENCY_ENERGY_BINS:-1024}"
FREQUENCY_ENERGY_COLOR_SPACE="${FREQUENCY_ENERGY_COLOR_SPACE:-rgb}"
FREQUENCY_CALIBRATION_MANIFEST="${FREQUENCY_CALIBRATION_MANIFEST:-}"
FREQUENCY_TRANSITION_RATIO="${FREQUENCY_TRANSITION_RATIO:-0.25}"
FREQUENCY_RESIZE="${FREQUENCY_RESIZE:-1333x800}"
FREQUENCY_PAD_FRACTION="${FREQUENCY_PAD_FRACTION:-0.05}"
FREQUENCY_MODEL_INPUT_MODE="${FREQUENCY_MODEL_INPUT_MODE:-natural-energy}"
BACKBONE_STATE_KEY="${BACKBONE_STATE_KEY:-}"
BACKBONE_PREFIX="${BACKBONE_PREFIX:-}"
BACKBONE_MINIMUM_MATCH="${BACKBONE_MINIMUM_MATCH:-0.5}"
MATERIALIZE="${MATERIALIZE:-copy}"
SAVE_SPATIAL="${SAVE_SPATIAL:-1}"
SPATIAL_SAMPLES="${SPATIAL_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
ANALYSIS_MODELS="${ANALYSIS_MODELS:-imagenet_backbone,cascade_ruod_backbone}"
# CKA matrix rows are the y-axis; columns are the x-axis. Keep the RUOD
# detector as the horizontal reference. CKA_MODEL_A/B remain compatibility
# aliases for older launch commands.
CKA_Y_MODEL="${CKA_Y_MODEL:-${CKA_MODEL_A:-imagenet_backbone}}"
CKA_Y_MODELS="${CKA_Y_MODELS:-${CKA_Y_MODEL}}"
CKA_X_MODEL="${CKA_X_MODEL:-${CKA_MODEL_B:-cascade_ruod_backbone}}"

RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_FREQUENCY_IMAGES="${RUN_FREQUENCY_IMAGES:-1}"
RUN_FEATURES="${RUN_FEATURES:-1}"
RUN_CKA="${RUN_CKA:-1}"
RUN_FREQUENCY_RESPONSE="${RUN_FREQUENCY_RESPONSE:-1}"
RUN_ACTIVATION="${RUN_ACTIVATION:-1}"
RUN_FREQUENCY_ACTIVATION="${RUN_FREQUENCY_ACTIVATION:-1}"
RUN_FREQUENCY_INPUT_VISUALS="${RUN_FREQUENCY_INPUT_VISUALS:-1}"
RUN_FREQUENCY_FIGURES="${RUN_FREQUENCY_FIGURES:-1}"
RUN_DETECTION_FREQUENCY_EVAL="${RUN_DETECTION_FREQUENCY_EVAL:-0}"
RUN_FOURIER_SENSITIVITY="${RUN_FOURIER_SENSITIVITY:-0}"
RUN_TSNE="${RUN_TSNE:-0}"

SAMPLE_ROOT="${OUT_ROOT}/sample"
FREQUENCY_ROOT="${OUT_ROOT}/frequency_inputs"
FEATURE_ROOT="${OUT_ROOT}/feature_store"
ANALYSIS_ROOT="${OUT_ROOT}/analysis"
MODELS_CONFIG="${OUT_ROOT}/models.json"

mkdir -p "${OUT_ROOT}" "${ANALYSIS_ROOT}"

if [[ -n "${MODELS_CONFIG_INPUT}" ]]; then
  MODELS_CONFIG="$(readlink -f "${MODELS_CONFIG_INPUT}")"
  [[ -f "${MODELS_CONFIG}" ]] || {
    echo "Error: models config not found: ${MODELS_CONFIG}" >&2
    exit 1
  }
  echo "External model configuration: ${MODELS_CONFIG}"
else
  [[ -n "${CASCADE_CHECKPOINT}" ]] || {
    echo "Error: set CASCADE_CHECKPOINT when MODELS_CONFIG_INPUT is empty" >&2
    exit 1
  }
  python - \
    "${MODELS_CONFIG}" \
    "${BACKBONE_CONFIG}" "${BACKBONE_CHECKPOINT}" \
    "${CASCADE_CONFIG}" "${CASCADE_CHECKPOINT}" \
    "${BACKBONE_STATE_KEY}" "${BACKBONE_PREFIX}" \
    "${BACKBONE_MINIMUM_MATCH}" <<'PY'
import json
import sys

(
    output, backbone_config, backbone_checkpoint,
    cascade_config, cascade_checkpoint,
    state_key, prefix, minimum_match,
) = sys.argv[1:]
backbone = {
    'id': 'imagenet_backbone',
    'kind': 'backbone',
    'config': backbone_config,
    'minimum_match_ratio': float(minimum_match),
    'layers': {
        'res2': 'layer1',
        'res3': 'layer2',
        'res4': 'layer3',
        'res5': 'layer4',
    },
}
if backbone_checkpoint:
    backbone['checkpoint'] = backbone_checkpoint
if state_key:
    backbone['state_dict_key'] = state_key
if prefix:
    backbone['checkpoint_prefix'] = prefix
models = {
    'models': [
        backbone,
        {
            'id': 'cascade_ruod_backbone',
            'kind': 'detector',
            'config': cascade_config,
            'checkpoint': cascade_checkpoint,
            'layers': {
                'res2': 'layer1',
                'res3': 'layer2',
                'res4': 'layer3',
                'res5': 'layer4',
            },
        },
    ],
}
with open(output, 'w', encoding='utf-8') as handle:
    json.dump(models, handle, indent=2)
    handle.write('\n')
print(f'Model configuration: {output}')
PY
fi

overwrite_args=()
if [[ "${OVERWRITE}" == 1 ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ "${RUN_SAMPLE}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.sample_ruod \
    --annotation-file "${RUOD_ANN}" \
    --image-root "${RUOD_IMAGE_ROOT}" \
    --out-dir "${SAMPLE_ROOT}" \
    --samples "${SAMPLES}" \
    --seed "${SEED}" \
    --materialize "${MATERIALIZE}" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FREQUENCY_IMAGES}" == 1 ]]; then
  frequency_band_args=(--band-policy "${FREQUENCY_BAND_POLICY}")
  if [[ "${FREQUENCY_BAND_POLICY}" == dataset-energy ]]; then
    frequency_band_args+=(
      --energy-quantiles "${FREQUENCY_ENERGY_QUANTILES}"
      --energy-bins "${FREQUENCY_ENERGY_BINS}"
      --energy-color-space "${FREQUENCY_ENERGY_COLOR_SPACE}"
    )
    if [[ -n "${FREQUENCY_CALIBRATION_MANIFEST}" ]]; then
      frequency_band_args+=(
        --calibration-manifest "${FREQUENCY_CALIBRATION_MANIFEST}")
    fi
  else
    frequency_band_args+=(--bands "${BANDS}")
  fi
  python -m tools.exp_2.backbone_analysis.generate_frequency_bands \
    --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
    --out-dir "${FREQUENCY_ROOT}" \
    --method "${FREQUENCY_METHOD}" \
    "${frequency_band_args[@]}" \
    --transition-ratio "${FREQUENCY_TRANSITION_RATIO}" \
    --resize "${FREQUENCY_RESIZE}" \
    --pad-fraction "${FREQUENCY_PAD_FRACTION}" \
    --model-input-mode "${FREQUENCY_MODEL_INPUT_MODE}" \
    --save-raw \
    --save-band-stop \
    --save-visualizations \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FEATURES}" == 1 ]]; then
  spatial_args=(--no-save-spatial)
  if [[ "${SAVE_SPATIAL}" == 1 ]]; then
    spatial_args=(--save-spatial --spatial-samples "${SPATIAL_SAMPLES}")
  fi
  python -m tools.exp_2.backbone_analysis.extract_backbone_features \
    --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --models-config "${MODELS_CONFIG}" \
    --out-dir "${FEATURE_ROOT}" \
    --variants "${VARIANTS}" \
    --layers "${LAYERS}" \
    --device "${DEVICE}" \
    "${spatial_args[@]}" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_CKA}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.compute_cka \
    --feature-root "${FEATURE_ROOT}" \
    --y-models "${CKA_Y_MODELS}" \
    --x-model "${CKA_X_MODEL}" \
    --variant clean \
    --layers-a "${LAYERS}" \
    --layers-b "${LAYERS}" \
    --out-dir "${ANALYSIS_ROOT}/cka" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FREQUENCY_RESPONSE}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.compute_frequency_response \
    --feature-root "${FEATURE_ROOT}" \
    --frequency-manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --models "${ANALYSIS_MODELS}" \
    --layers "${LAYERS}" \
    --clean-variant clean \
    --frequency-variants "${FREQUENCY_RESPONSE_VARIANTS}" \
    --out-dir "${ANALYSIS_ROOT}/frequency_response" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_ACTIVATION}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.render_feature_activation \
    --feature-root "${FEATURE_ROOT}" \
    --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --models "${ANALYSIS_MODELS}" \
    --layers "${LAYERS}" \
    --variant clean \
    --out-dir "${ANALYSIS_ROOT}/activation" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FREQUENCY_ACTIVATION}" == 1 ]]; then
  IFS=',' read -r -a activation_variants <<< "${VARIANTS}"
  for variant in "${activation_variants[@]}"; do
    python -m tools.exp_2.backbone_analysis.render_feature_activation \
      --feature-root "${FEATURE_ROOT}" \
      --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
      --models "${ANALYSIS_MODELS}" \
      --layers "${LAYERS}" \
      --variant "${variant}" \
      --out-dir "${ANALYSIS_ROOT}/activation_by_frequency/${variant}" \
      "${overwrite_args[@]}"
  done
fi

if [[ "${RUN_FREQUENCY_INPUT_VISUALS}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.render_frequency_inputs \
    --frequency-root "${FREQUENCY_ROOT}" \
    --out-dir "${ANALYSIS_ROOT}/frequency_inputs" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_DETECTION_FREQUENCY_EVAL}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.evaluate_frequency_detection \
    --frequency-manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --annotation-file "${RUOD_ANN}" \
    --models-config "${MODELS_CONFIG}" \
    --variants clean,remove_low,remove_mid,remove_high \
    --device "${DEVICE}" \
    --out-dir "${ANALYSIS_ROOT}/frequency_detection" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FOURIER_SENSITIVITY}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.compute_fourier_basis_sensitivity \
    --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --models-config "${MODELS_CONFIG}" \
    --layers "${LAYERS}" \
    --filter-config "${FREQUENCY_ROOT}/filter_config.json" \
    --device "${DEVICE}" \
    --out-dir "${ANALYSIS_ROOT}/fourier_basis_sensitivity" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FREQUENCY_FIGURES}" == 1 ]]; then
  frequency_figure_args=(
    --response-dir "${ANALYSIS_ROOT}/frequency_response"
    --out-dir "${ANALYSIS_ROOT}/frequency_figures"
  )
  if [[ -d "${ANALYSIS_ROOT}/activation_by_frequency" ]]; then
    frequency_figure_args+=(
      --activation-root "${ANALYSIS_ROOT}/activation_by_frequency")
  fi
  if [[ -f "${ANALYSIS_ROOT}/frequency_detection/frequency_detection_metrics.tsv" ]]; then
    frequency_figure_args+=(
      --detection-metrics \
      "${ANALYSIS_ROOT}/frequency_detection/frequency_detection_metrics.tsv")
  fi
  python -m tools.exp_2.backbone_analysis.render_frequency_response \
    "${frequency_figure_args[@]}" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_TSNE}" == 1 ]]; then
  TSNE_LAYER="${TSNE_LAYER:-res5}"
  python -m tools.exp_2.backbone_analysis.compute_tsne \
    --feature-root "${FEATURE_ROOT}" \
    --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
    --models "${ANALYSIS_MODELS}" \
    --layer "${TSNE_LAYER}" \
    --variant clean \
    --seed "${SEED}" \
    --out-dir "${ANALYSIS_ROOT}/tsne_${TSNE_LAYER}" \
    "${overwrite_args[@]}"
fi

echo "Pipeline completed: ${OUT_ROOT}"
