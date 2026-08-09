#!/usr/bin/env bash
set -euo pipefail

# Run fixed and dataset-energy frequency experiments sequentially. Within one
# policy, assign one model to one GPU and extract all missing variants in
# parallel. Analysis starts only after every model/variant passes validation.

BASE_OUT_ROOT="${BASE_OUT_ROOT:?Set BASE_OUT_ROOT}"
FIXED_OUT_ROOT="${FIXED_OUT_ROOT:-${BASE_OUT_ROOT}/fixed}"
DATASET_ENERGY_OUT_ROOT="${DATASET_ENERGY_OUT_ROOT:-${BASE_OUT_ROOT}/dataset_energy}"
RUOD_ANN="${RUOD_ANN:?Set RUOD_ANN}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:?Set RUOD_IMAGE_ROOT}"
MODELS_CONFIG_INPUT="${MODELS_CONFIG_INPUT:?Set MODELS_CONFIG_INPUT}"

PIPELINE="${PIPELINE:-tools/exp_2/backbone_analysis/run_pipeline.sh}"
GPUS="${GPUS:-2,3,4,5,6,7}"
POLICIES="${POLICIES:-fixed,dataset-energy}"
SAMPLES="${SAMPLES:-100}"
SPATIAL_SAMPLES="${SPATIAL_SAMPLES:-${SAMPLES}}"
SEED="${SEED:-2026}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
VARIANTS="${VARIANTS:-clean,low,mid,high,remove_low,remove_mid,remove_high}"
ANALYSIS_MODELS="${ANALYSIS_MODELS:-}"
CKA_Y_MODELS="${CKA_Y_MODELS:-}"
CKA_X_MODEL="${CKA_X_MODEL:-}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cuda:0}"
OVERWRITE_ANALYSIS="${OVERWRITE_ANALYSIS:-0}"

FIXED_BANDS="${FIXED_BANDS:-low:0:1/32,mid:1/32:1/8,high:1/8:max}"
FREQUENCY_METHOD="${FREQUENCY_METHOD:-soft-cpp}"
FREQUENCY_TRANSITION_RATIO="${FREQUENCY_TRANSITION_RATIO:-0.25}"
FREQUENCY_RESIZE="${FREQUENCY_RESIZE:-1333x800}"
FREQUENCY_PAD_FRACTION="${FREQUENCY_PAD_FRACTION:-0.05}"
FREQUENCY_MODEL_INPUT_MODE="${FREQUENCY_MODEL_INPUT_MODE:-natural-energy}"
FREQUENCY_ENERGY_QUANTILES="${FREQUENCY_ENERGY_QUANTILES:-1/3,2/3}"
FREQUENCY_ENERGY_BINS="${FREQUENCY_ENERGY_BINS:-1024}"
FREQUENCY_ENERGY_COLOR_SPACE="${FREQUENCY_ENERGY_COLOR_SPACE:-rgb}"
FREQUENCY_CALIBRATION_MANIFEST="${FREQUENCY_CALIBRATION_MANIFEST:-}"

RUN_CKA="${RUN_CKA:-1}"
RUN_FREQUENCY_RESPONSE="${RUN_FREQUENCY_RESPONSE:-1}"
RUN_ACTIVATION="${RUN_ACTIVATION:-1}"
RUN_FREQUENCY_ACTIVATION="${RUN_FREQUENCY_ACTIVATION:-1}"
RUN_FREQUENCY_INPUT_VISUALS="${RUN_FREQUENCY_INPUT_VISUALS:-1}"
RUN_FREQUENCY_FIGURES="${RUN_FREQUENCY_FIGURES:-1}"
RUN_DETECTION_FREQUENCY_EVAL="${RUN_DETECTION_FREQUENCY_EVAL:-0}"
RUN_FOURIER_SENSITIVITY="${RUN_FOURIER_SENSITIVITY:-0}"
RUN_TSNE="${RUN_TSNE:-1}"
ACTIVATION_JOBS="${ACTIVATION_JOBS:-7}"
FIXED_ACTIVATION_JOBS="${FIXED_ACTIVATION_JOBS:-${ACTIVATION_JOBS}}"
DATASET_ENERGY_ACTIVATION_JOBS="${DATASET_ENERGY_ACTIVATION_JOBS:-3}"
ACTIVATION_PNG_COMPRESS_LEVEL="${ACTIVATION_PNG_COMPRESS_LEVEL:-1}"
ACTIVATION_REUSE_COMPLETE="${ACTIVATION_REUSE_COMPLETE:-1}"

[[ -f "${PIPELINE}" ]] || {
  echo "Error: pipeline not found: ${PIPELINE}" >&2
  exit 1
}
[[ -f "${MODELS_CONFIG_INPUT}" ]] || {
  echo "Error: models config not found: ${MODELS_CONFIG_INPUT}" >&2
  exit 1
}

IFS=',' read -r -a gpu_list <<< "${GPUS}"
IFS=',' read -r -a policy_list <<< "${POLICIES}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a layer_list <<< "${LAYERS}"

mapfile -t model_list < <(
  python - "${MODELS_CONFIG_INPUT}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    payload = json.load(handle)
models = payload.get('models', [])
if not models:
    raise SystemExit('models config contains no models')
for spec in models:
    model_id = str(spec['id'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', model_id):
        raise SystemExit('unsafe model id: {}'.format(model_id))
    print(model_id)
PY
)

if (( ${#gpu_list[@]} < ${#model_list[@]} )); then
  echo "Error: ${#model_list[@]} models require at least that many GPUs; got ${#gpu_list[@]}" >&2
  exit 1
fi
ANALYSIS_GPU="${ANALYSIS_GPU:-${gpu_list[$((${#gpu_list[@]} - 1))]}}"

if [[ -z "${ANALYSIS_MODELS}" ]]; then
  ANALYSIS_MODELS="$(IFS=,; echo "${model_list[*]}")"
fi
if [[ -z "${CKA_X_MODEL}" ]]; then
  CKA_X_MODEL="${model_list[$((${#model_list[@]} - 1))]}"
fi
if [[ -z "${CKA_Y_MODELS}" ]]; then
  cka_y=()
  for model_id in "${model_list[@]}"; do
    [[ "${model_id}" == "${CKA_X_MODEL}" ]] || cka_y+=("${model_id}")
  done
  CKA_Y_MODELS="$(IFS=,; echo "${cka_y[*]}")"
fi

expected_spatial_samples="${SPATIAL_SAMPLES}"
if (( expected_spatial_samples <= 0 || expected_spatial_samples > SAMPLES )); then
  expected_spatial_samples="${SAMPLES}"
fi
expected_spatial=$((expected_spatial_samples * ${#layer_list[@]}))

common_pipeline_env=(
  RUOD_ANN="${RUOD_ANN}"
  RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT}"
  MODELS_CONFIG_INPUT="${MODELS_CONFIG_INPUT}"
  SAMPLES="${SAMPLES}"
  SEED="${SEED}"
  LAYERS="${LAYERS}"
  VARIANTS="${VARIANTS}"
  FREQUENCY_RESPONSE_VARIANTS="${FREQUENCY_RESPONSE_VARIANTS:-low,mid,high,remove_low,remove_mid,remove_high}"
  ANALYSIS_MODELS="${ANALYSIS_MODELS}"
  CKA_Y_MODELS="${CKA_Y_MODELS}"
  CKA_X_MODEL="${CKA_X_MODEL}"
  FREQUENCY_METHOD="${FREQUENCY_METHOD}"
  FREQUENCY_TRANSITION_RATIO="${FREQUENCY_TRANSITION_RATIO}"
  FREQUENCY_RESIZE="${FREQUENCY_RESIZE}"
  FREQUENCY_PAD_FRACTION="${FREQUENCY_PAD_FRACTION}"
  FREQUENCY_MODEL_INPUT_MODE="${FREQUENCY_MODEL_INPUT_MODE}"
  FREQUENCY_ENERGY_QUANTILES="${FREQUENCY_ENERGY_QUANTILES}"
  FREQUENCY_ENERGY_BINS="${FREQUENCY_ENERGY_BINS}"
  FREQUENCY_ENERGY_COLOR_SPACE="${FREQUENCY_ENERGY_COLOR_SPACE}"
  ACTIVATION_PNG_COMPRESS_LEVEL="${ACTIVATION_PNG_COMPRESS_LEVEL}"
  ACTIVATION_REUSE_COMPLETE="${ACTIVATION_REUSE_COMPLETE}"
)
if [[ -n "${FREQUENCY_CALIBRATION_MANIFEST}" ]]; then
  common_pipeline_env+=(
    FREQUENCY_CALIBRATION_MANIFEST="${FREQUENCY_CALIBRATION_MANIFEST}")
fi

validate_task() {
  local feature_root="$1"
  local model_id="$2"
  local variant="$3"
  python - \
    "${feature_root}" "${model_id}" "${variant}" \
    "${SAMPLES}" "${expected_spatial}" "${LAYERS}" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np

root = Path(sys.argv[1])
model, variant = sys.argv[2:4]
samples, expected_spatial = map(int, sys.argv[4:6])
layers = [item for item in sys.argv[6].split(',') if item]
feature_dir = root / 'features' / model / variant

for layer in layers:
    pooled = feature_dir / (layer + '.npy')
    norms = feature_dir / (layer + '.spatial_norm.npy')
    if not pooled.is_file() or not norms.is_file():
        raise SystemExit(1)
    pooled_array = np.load(str(pooled), mmap_mode='r')
    norm_array = np.load(str(norms), mmap_mode='r')
    if pooled_array.ndim != 2 or pooled_array.shape[0] != samples:
        raise SystemExit(1)
    if norm_array.shape != (samples,):
        raise SystemExit(1)

metadata = root / 'metadata' / model / (variant + '.json')
if not metadata.is_file():
    raise SystemExit(1)
with metadata.open('r', encoding='utf-8') as handle:
    if len(json.load(handle)) != samples:
        raise SystemExit(1)

spatial_root = root / 'spatial' / model / variant
spatial = [path for path in spatial_root.rglob('*.npz') if path.stat().st_size > 0]
if len(spatial) != expected_spatial:
    raise SystemExit(1)
PY
}

prepare_policy() {
  local policy="$1"
  local out_root="$2"
  local run_sample=0
  local run_frequency=0

  [[ -s "${out_root}/sample/manifest.jsonl" ]] || run_sample=1
  [[ -s "${out_root}/frequency_inputs/frequency_manifest.jsonl" ]] || run_frequency=1

  if (( run_sample == 0 && run_frequency == 0 )); then
    echo "reuse prepared inputs: ${out_root}"
    return
  fi

  env "${common_pipeline_env[@]}" \
    OUT_ROOT="${out_root}" \
    DEVICE="${ANALYSIS_DEVICE}" \
    FREQUENCY_BAND_POLICY="${policy}" \
    BANDS="${FIXED_BANDS}" \
    RUN_SAMPLE="${run_sample}" \
    RUN_FREQUENCY_IMAGES="${run_frequency}" \
    RUN_FEATURES=0 \
    RUN_CKA=0 \
    RUN_FREQUENCY_RESPONSE=0 \
    RUN_ACTIVATION=0 \
    RUN_FREQUENCY_ACTIVATION=0 \
    RUN_FREQUENCY_INPUT_VISUALS=0 \
    RUN_FREQUENCY_FIGURES=0 \
    RUN_DETECTION_FREQUENCY_EVAL=0 \
    RUN_FOURIER_SENSITIVITY=0 \
    RUN_TSNE=0 \
    OVERWRITE=0 \
    bash "${PIPELINE}"
}

extract_model() {
  local policy="$1"
  local out_root="$2"
  local model_id="$3"
  local physical_gpu="$4"
  local feature_root="${out_root}/feature_store"
  local stage_root="${out_root}/.parallel_feature_stage/${model_id}"
  local log_root="${out_root}/logs/parallel_features"
  local variant stage log pending_csv
  local -a pending_variants=()

  mkdir -p "${stage_root}" "${log_root}"
  for variant in "${variant_list[@]}"; do
    if validate_task "${feature_root}" "${model_id}" "${variant}"; then
      echo "reuse ${policy}/${model_id}/${variant}"
    else
      pending_variants+=("${variant}")
    fi
  done

  if (( ${#pending_variants[@]} == 0 )); then
    echo "all variants complete: ${policy}/${model_id}"
    return
  fi

  pending_csv="$(IFS=,; echo "${pending_variants[*]}")"
  stage="${stage_root}/pending"
  log="${log_root}/${model_id}_gpu${physical_gpu}.log"
  echo "extract ${policy}/${model_id} variants=${pending_csv} on GPU ${physical_gpu}"

  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    python -m tools.exp_2.backbone_analysis.extract_backbone_features \
      --manifest "${out_root}/frequency_inputs/frequency_manifest.jsonl" \
      --models-config "${MODELS_CONFIG_INPUT}" \
      --out-dir "${stage}" \
      --models "${model_id}" \
      --variants "${pending_csv}" \
      --layers "${LAYERS}" \
      --device cuda:0 \
      --save-spatial \
      --spatial-samples "${SPATIAL_SAMPLES}" \
      --spatial-dtype float16 \
      --overwrite \
      >"${log}" 2>&1

  for variant in "${pending_variants[@]}"; do
    validate_task "${stage}" "${model_id}" "${variant}" || {
      echo "Error: staged task failed validation: ${policy}/${model_id}/${variant}" >&2
      echo "Log: ${log}" >&2
      return 1
    }

    mkdir -p \
      "${feature_root}/features/${model_id}" \
      "${feature_root}/spatial/${model_id}" \
      "${feature_root}/metadata/${model_id}" \
      "${feature_root}/parallel_audit/${model_id}/${variant}"
    rsync -a \
      "${stage}/features/${model_id}/${variant}/" \
      "${feature_root}/features/${model_id}/${variant}/"
    rsync -a \
      "${stage}/spatial/${model_id}/${variant}/" \
      "${feature_root}/spatial/${model_id}/${variant}/"
    cp -f \
      "${stage}/metadata/${model_id}/${variant}.json" \
      "${feature_root}/metadata/${model_id}/${variant}.json"
    cp -f "${stage}/source.json" \
      "${feature_root}/parallel_audit/${model_id}/${variant}/source.json"
    cp -f "${stage}/model_load_reports.json" \
      "${feature_root}/parallel_audit/${model_id}/${variant}/model_load_reports.json"
    cp -f "${stage}/qa/feature_summary.tsv" \
      "${feature_root}/parallel_audit/${model_id}/${variant}/feature_summary.all_pending.tsv"
    cp -f "${stage}/sample_ids.npy" "${feature_root}/sample_ids.npy"

    validate_task "${feature_root}" "${model_id}" "${variant}" || {
      echo "Error: merged task failed validation: ${policy}/${model_id}/${variant}" >&2
      return 1
    }
    echo "completed ${policy}/${model_id}/${variant}"
  done
}

validate_policy() {
  local policy="$1"
  local out_root="$2"
  local feature_root="${out_root}/feature_store"
  local failed=0 model_id variant
  for model_id in "${model_list[@]}"; do
    for variant in "${variant_list[@]}"; do
      if validate_task "${feature_root}" "${model_id}" "${variant}"; then
        echo "OK ${policy}/${model_id}/${variant}"
      else
        echo "MISSING ${policy}/${model_id}/${variant}" >&2
        failed=1
      fi
    done
  done
  (( failed == 0 ))
}

run_analysis() {
  local policy="$1"
  local out_root="$2"
  local activation_jobs="$3"
  CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}" \
  env "${common_pipeline_env[@]}" \
    OUT_ROOT="${out_root}" \
    DEVICE="${ANALYSIS_DEVICE}" \
    FREQUENCY_BAND_POLICY="${policy}" \
    BANDS="${FIXED_BANDS}" \
    RUN_SAMPLE=0 \
    RUN_FREQUENCY_IMAGES=0 \
    RUN_FEATURES=0 \
    RUN_CKA="${RUN_CKA}" \
    RUN_FREQUENCY_RESPONSE="${RUN_FREQUENCY_RESPONSE}" \
    RUN_ACTIVATION="${RUN_ACTIVATION}" \
    RUN_FREQUENCY_ACTIVATION="${RUN_FREQUENCY_ACTIVATION}" \
    RUN_FREQUENCY_INPUT_VISUALS="${RUN_FREQUENCY_INPUT_VISUALS}" \
    RUN_FREQUENCY_FIGURES="${RUN_FREQUENCY_FIGURES}" \
    RUN_DETECTION_FREQUENCY_EVAL="${RUN_DETECTION_FREQUENCY_EVAL}" \
    RUN_FOURIER_SENSITIVITY="${RUN_FOURIER_SENSITIVITY}" \
    RUN_TSNE="${RUN_TSNE}" \
    ACTIVATION_JOBS="${activation_jobs}" \
    OVERWRITE="${OVERWRITE_ANALYSIS}" \
    bash "${PIPELINE}"
}

mkdir -p "${BASE_OUT_ROOT}"
cat <<EOF
============================================================
Sequential-policy, parallel-model backbone analysis
============================================================
BASE_OUT_ROOT:      ${BASE_OUT_ROOT}
FIXED_OUT_ROOT:     ${FIXED_OUT_ROOT}
ENERGY_OUT_ROOT:    ${DATASET_ENERGY_OUT_ROOT}
POLICIES:           ${POLICIES}
MODELS:             ${model_list[*]}
GPUS:               ${gpu_list[*]}
ANALYSIS_GPU:       ${ANALYSIS_GPU}
VARIANTS:           ${VARIANTS}
SAMPLES:            ${SAMPLES}
SPATIAL_SAMPLES:    ${SPATIAL_SAMPLES}
CKA_X_MODEL:        ${CKA_X_MODEL}
CKA_Y_MODELS:       ${CKA_Y_MODELS}
OVERWRITE_ANALYSIS: ${OVERWRITE_ANALYSIS}
FIXED_ACT_JOBS:     ${FIXED_ACTIVATION_JOBS}
ENERGY_ACT_JOBS:    ${DATASET_ENERGY_ACTIVATION_JOBS}
PNG_COMPRESSION:    ${ACTIVATION_PNG_COMPRESS_LEVEL}
============================================================
EOF

for policy in "${policy_list[@]}"; do
  case "${policy}" in
    fixed)
      policy_root="${FIXED_OUT_ROOT}"
      policy_activation_jobs="${FIXED_ACTIVATION_JOBS}"
      ;;
    dataset-energy|dataset_energy)
      policy="dataset-energy"
      policy_root="${DATASET_ENERGY_OUT_ROOT}"
      policy_activation_jobs="${DATASET_ENERGY_ACTIVATION_JOBS}"
      ;;
    *)
      echo "Error: unsupported policy: ${policy}" >&2
      exit 1
      ;;
  esac

  echo
  echo "===== prepare ${policy} ====="
  prepare_policy "${policy}" "${policy_root}"

  echo
  echo "===== parallel feature extraction: ${policy} ====="
  worker_pids=()
  worker_names=()
  for index in "${!model_list[@]}"; do
    model_id="${model_list[$index]}"
    gpu="${gpu_list[$index]}"
    worker_log="${policy_root}/logs/parallel_features/worker_${model_id}_gpu${gpu}.log"
    mkdir -p "$(dirname "${worker_log}")"
    (
      trap '' HUP
      extract_model "${policy}" "${policy_root}" "${model_id}" "${gpu}"
    ) >"${worker_log}" 2>&1 &
    worker_pids+=("$!")
    worker_names+=("${model_id}:gpu${gpu}")
    echo "started ${model_id} on GPU ${gpu}: pid=$! log=${worker_log}"
  done

  worker_failed=0
  for index in "${!worker_pids[@]}"; do
    if wait "${worker_pids[$index]}"; then
      echo "finished ${worker_names[$index]}"
    else
      echo "FAILED ${worker_names[$index]}" >&2
      worker_failed=1
    fi
  done
  (( worker_failed == 0 )) || {
    echo "Error: one or more ${policy} model workers failed" >&2
    exit 1
  }

  echo
  echo "===== validate ${policy} feature store ====="
  validate_policy "${policy}" "${policy_root}"
  date --iso-8601=seconds > "${policy_root}/feature_store/PARALLEL_EXTRACTION_COMPLETE"

  echo
  echo "===== unified ${policy} analysis ====="
  run_analysis "${policy}" "${policy_root}" "${policy_activation_jobs}"
  date --iso-8601=seconds > "${policy_root}/ANALYSIS_COMPLETE"
  echo "completed policy: ${policy}"
done

echo "All requested frequency policies completed: ${BASE_OUT_ROOT}"
