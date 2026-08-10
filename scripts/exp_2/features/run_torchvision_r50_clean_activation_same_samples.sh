#!/usr/bin/env bash
set -euo pipefail

# Extract torchvision ImageNet-supervised ResNet-50 features for the existing
# RUOD sample manifest, then render only clean activation maps with the two
# ImageNet-reference normalization scopes used by the backbone analysis.

REPO_ROOT="${REPO_ROOT:-$HOME/xcx/exp_2/mmdetection}"
RUN_NAME="${RUN_NAME:-dino100e_five_models_ruod100_20260809_162028}"
SOURCE_POLICY_ROOT="${SOURCE_POLICY_ROOT:-/media/SSD2/XCX/exp_2/backbone_analysis/${RUN_NAME}/fixed}"
MANIFEST="${MANIFEST:-${SOURCE_POLICY_ROOT}/frequency_inputs/frequency_manifest.jsonl}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-${REPO_ROOT}/configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py}"
MODEL_ID="${MODEL_ID:-torchvision_imagenet_resnet50}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLES="${SAMPLES:-100}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-3}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/backbone_analysis/${RUN_NAME}/torchvision_r50_clean_activation}"
FEATURE_ROOT="${FEATURE_ROOT:-${OUT_ROOT}/feature_store}"
MODELS_CONFIG="${MODELS_CONFIG:-${OUT_ROOT}/torchvision_r50_model.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${OUT_ROOT}/analysis}"
OVERWRITE="${OVERWRITE:-0}"

cd "${REPO_ROOT}"

for required in "${MANIFEST}" "${BACKBONE_CONFIG}"; do
  [[ -s "${required}" ]] || {
    echo "Error: required file not found: ${required}" >&2
    exit 1
  }
done

actual_samples="$(wc -l < "${MANIFEST}")"
[[ "${actual_samples}" -eq "${SAMPLES}" ]] || {
  echo "Error: manifest has ${actual_samples} rows; expected ${SAMPLES}" >&2
  exit 1
}

mkdir -p "${OUT_ROOT}" "${ANALYSIS_ROOT}"

python - "${MODELS_CONFIG}" "${BACKBONE_CONFIG}" "${MODEL_ID}" <<'PY'
import json
import os
import sys
from pathlib import Path

output, config, model_id = sys.argv[1:]
payload = {
    'models': [{
        'id': model_id,
        'kind': 'backbone',
        'config': str(Path(config).resolve()),
        'layers': {
            'res2': 'layer1',
            'res3': 'layer2',
            'res4': 'layer3',
            'res5': 'layer4',
        },
    }],
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + '.tmp')
with temporary.open('w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2)
    handle.write('\n')
os.replace(str(temporary), str(path))
print('Model configuration: {}'.format(path))
PY

expected_spatial="$((SAMPLES * 4))"
current_spatial="$({
  find "${FEATURE_ROOT}/spatial/${MODEL_ID}/clean" \
    -type f -name '*.npz' 2>/dev/null || true
} | wc -l)"

if [[ "${current_spatial}" -eq "${expected_spatial}" ]] && \
   [[ -s "${FEATURE_ROOT}/model_load_reports.json" ]]; then
  echo "Reusing complete torchvision feature store: ${FEATURE_ROOT}"
else
  [[ ! -e "${FEATURE_ROOT}" ]] || {
    if [[ "${OVERWRITE}" == 1 ]]; then
      echo "Error: OVERWRITE=1 is intentionally not destructive here." >&2
      echo "Move the partial feature root aside before rerunning:" >&2
      echo "  ${FEATURE_ROOT}" >&2
    else
      echo "Error: partial feature root exists (${current_spatial}/${expected_spatial}):" >&2
      echo "  ${FEATURE_ROOT}" >&2
    fi
    exit 1
  }

  echo "============================================================"
  echo "Extract torchvision ResNet-50 clean spatial features"
  echo "============================================================"
  date

  python -m tools.exp_2.backbone_analysis.extract_backbone_features \
    --manifest "${MANIFEST}" \
    --models-config "${MODELS_CONFIG}" \
    --out-dir "${FEATURE_ROOT}" \
    --models "${MODEL_ID}" \
    --variants clean \
    --layers "${LAYERS}" \
    --device "${DEVICE}" \
    --pooling avg \
    --save-spatial \
    --spatial-samples "${SAMPLES}" \
    --spatial-dtype float16
fi

current_spatial="$(find \
  "${FEATURE_ROOT}/spatial/${MODEL_ID}/clean" \
  -type f -name '*.npz' | wc -l)"
[[ "${current_spatial}" -eq "${expected_spatial}" ]] || {
  echo "Error: spatial feature count ${current_spatial}/${expected_spatial}" >&2
  exit 1
}

render_mode() {
  local mode="$1"
  local destination="$2"

  if [[ -s "${destination}/activation_metadata.json" ]]; then
    echo "Reusing complete-looking activation output: ${destination}"
    return
  fi
  [[ ! -e "${destination}" ]] || {
    echo "Error: partial activation output exists: ${destination}" >&2
    exit 1
  }

  echo
  echo "============================================================"
  echo "Render ${mode}"
  echo "============================================================"
  date

  python -m tools.exp_2.backbone_analysis.render_feature_activation \
    --feature-root "${FEATURE_ROOT}" \
    --manifest "${MANIFEST}" \
    --models "${MODEL_ID}" \
    --layers "${LAYERS}" \
    --variant clean \
    --normalization-mode "${mode}" \
    --normalization-reference-model "${MODEL_ID}" \
    --low-percentile "${LOW_PERCENTILE}" \
    --high-percentile "${HIGH_PERCENTILE}" \
    --skip-raw-activation \
    --png-compress-level "${PNG_COMPRESS_LEVEL}" \
    --out-dir "${destination}"
}

render_mode \
  reference-dataset-per-layer \
  "${ANALYSIS_ROOT}/torchvision_reference_dataset_p01_p99"

render_mode \
  reference-per-sample-layer \
  "${ANALYSIS_ROOT}/torchvision_reference_per_sample_p01_p99"

for mode in \
  torchvision_reference_dataset_p01_p99 \
  torchvision_reference_per_sample_p01_p99
do
  root="${ANALYSIS_ROOT}/${mode}"
  without="$(find "${root}" -path '*/without_boxes/*' -name '*.png' | wc -l)"
  with_gt="$(find "${root}" -path '*/with_gt_boxes/*' -name '*.png' | wc -l)"
  panels="$(find "${root}/panels" -name '*.png' | wc -l)"
  raw="$(find "${root}" -name '*.npy' | wc -l)"
  [[ "${without}" -eq "${expected_spatial}" ]] || exit 1
  [[ "${with_gt}" -eq "${expected_spatial}" ]] || exit 1
  [[ "${panels}" -eq "${expected_spatial}" ]] || exit 1
  [[ "${raw}" -eq 0 ]] || exit 1
  echo "${mode}: without=${without}, with_gt=${with_gt}, panels=${panels}, raw=${raw}"
done

cat > "${OUT_ROOT}/COMPLETE.txt" <<EOF
completed=$(date --iso-8601=seconds)
model=${MODEL_ID}
weights=torchvision://resnet50
manifest=${MANIFEST}
samples=${SAMPLES}
layers=${LAYERS}
feature_root=${FEATURE_ROOT}
analysis_root=${ANALYSIS_ROOT}
EOF

echo
echo "============================================================"
echo "TORCHVISION CLEAN ACTIVATION EXPERIMENT COMPLETE"
echo "============================================================"
echo "Output: ${OUT_ROOT}"
date
