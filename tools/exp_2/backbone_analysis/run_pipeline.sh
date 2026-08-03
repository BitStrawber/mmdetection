#!/usr/bin/env bash
set -euo pipefail

# Modular ImageNet-backbone vs RUOD-Cascade-backbone analysis pipeline.
# Every stage can be disabled and run independently with RUN_* variables.

RUOD_ANN="${RUOD_ANN:?Set RUOD_ANN to a COCO annotation JSON}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:?Set RUOD_IMAGE_ROOT to the image directory}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:?Set BACKBONE_CONFIG to a detector config used for preprocessing}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:?Set BACKBONE_CHECKPOINT}"
CASCADE_CONFIG="${CASCADE_CONFIG:?Set CASCADE_CONFIG}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:?Set CASCADE_CHECKPOINT}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"

SAMPLES="${SAMPLES:-50}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda:0}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
BANDS="${BANDS:-low:0.0:0.15,mid:0.15:0.40,high:0.40:1.0}"
VARIANTS="${VARIANTS:-clean,low,mid,high}"
BACKBONE_STATE_KEY="${BACKBONE_STATE_KEY:-}"
BACKBONE_PREFIX="${BACKBONE_PREFIX:-}"
BACKBONE_MINIMUM_MATCH="${BACKBONE_MINIMUM_MATCH:-0.5}"
MATERIALIZE="${MATERIALIZE:-copy}"
SAVE_SPATIAL="${SAVE_SPATIAL:-1}"
SPATIAL_SAMPLES="${SPATIAL_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"

RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_FREQUENCY_IMAGES="${RUN_FREQUENCY_IMAGES:-1}"
RUN_FEATURES="${RUN_FEATURES:-1}"
RUN_CKA="${RUN_CKA:-1}"
RUN_FREQUENCY_RESPONSE="${RUN_FREQUENCY_RESPONSE:-1}"
RUN_ACTIVATION="${RUN_ACTIVATION:-1}"
RUN_TSNE="${RUN_TSNE:-0}"

SAMPLE_ROOT="${OUT_ROOT}/sample"
FREQUENCY_ROOT="${OUT_ROOT}/frequency_inputs"
FEATURE_ROOT="${OUT_ROOT}/feature_store"
ANALYSIS_ROOT="${OUT_ROOT}/analysis"
MODELS_CONFIG="${OUT_ROOT}/models.json"

mkdir -p "${OUT_ROOT}" "${ANALYSIS_ROOT}"

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
    'checkpoint': backbone_checkpoint,
    'minimum_match_ratio': float(minimum_match),
    'layers': {
        'res2': 'layer1',
        'res3': 'layer2',
        'res4': 'layer3',
        'res5': 'layer4',
    },
}
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
  python -m tools.exp_2.backbone_analysis.generate_frequency_bands \
    --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
    --out-dir "${FREQUENCY_ROOT}" \
    --bands "${BANDS}" \
    --reconstruction mean-preserve \
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
    --model-a imagenet_backbone \
    --model-b cascade_ruod_backbone \
    --variant clean \
    --layers-a "${LAYERS}" \
    --layers-b "${LAYERS}" \
    --out-dir "${ANALYSIS_ROOT}/cka" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_FREQUENCY_RESPONSE}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.compute_frequency_response \
    --feature-root "${FEATURE_ROOT}" \
    --models imagenet_backbone,cascade_ruod_backbone \
    --layers "${LAYERS}" \
    --clean-variant clean \
    --frequency-variants low,mid,high \
    --out-dir "${ANALYSIS_ROOT}/frequency_response" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_ACTIVATION}" == 1 ]]; then
  python -m tools.exp_2.backbone_analysis.render_feature_activation \
    --feature-root "${FEATURE_ROOT}" \
    --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" \
    --models imagenet_backbone,cascade_ruod_backbone \
    --layers "${LAYERS}" \
    --variant clean \
    --out-dir "${ANALYSIS_ROOT}/activation" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_TSNE}" == 1 ]]; then
  TSNE_LAYER="${TSNE_LAYER:-res5}"
  python -m tools.exp_2.backbone_analysis.compute_tsne \
    --feature-root "${FEATURE_ROOT}" \
    --manifest "${SAMPLE_ROOT}/manifest.jsonl" \
    --models imagenet_backbone,cascade_ruod_backbone \
    --layer "${TSNE_LAYER}" \
    --variant clean \
    --seed "${SEED}" \
    --out-dir "${ANALYSIS_ROOT}/tsne_${TSNE_LAYER}" \
    "${overwrite_args[@]}"
fi

echo "Pipeline completed: ${OUT_ROOT}"
