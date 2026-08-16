#!/usr/bin/env bash
# Full RUOD analysis suite: sample -> energy bands -> features -> CKA/frequency/CAM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco}"
RUOD_IMAGE_ROOT="${RUOD_IMAGE_ROOT:-${RUOD_ROOT}/val}"
RUOD_ANN="${RUOD_ANN:-${RUOD_ROOT}/annotations/instances_val.json}"
FEATURE_MODELS_CONFIG="${FEATURE_MODELS_CONFIG:-${SCRIPT_DIR}/models.features.example.json}"
CAM_MODELS_CONFIG="${CAM_MODELS_CONFIG:-${SCRIPT_DIR}/models.cam.example.json}"
SAMPLES="${SAMPLES:-20}"
SEED="${SEED:-2026}"
CAM_SAMPLES="${CAM_SAMPLES:-${SAMPLES}}"
CAM_SEED="${CAM_SEED:-$((SEED + 1009))}"
MATERIALIZE="${MATERIALIZE:-none}"
LAYERS="${LAYERS:-res2,res3,res4,res5}"
FEATURE_VARIANTS="${FEATURE_VARIANTS:-clean,low,mid,high,remove_low,remove_mid,remove_high}"
ENERGY_CALIBRATION_MANIFEST="${ENERGY_CALIBRATION_MANIFEST:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-ruod${SAMPLES}_energy_cam_cka_frequency_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/analysis/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
SAMPLE_ROOT="${SAMPLE_ROOT:-${OUT_ROOT}/sample}"
CAM_SAMPLE_ROOT="${CAM_SAMPLE_ROOT:-${OUT_ROOT}/cam_sample}"
CAM_OUTPUT_ROOT="${CAM_OUTPUT_ROOT:-${OUT_ROOT}/cam_prediction}"
FREQUENCY_ROOT="${FREQUENCY_ROOT:-${OUT_ROOT}/frequency_inputs}"
FEATURE_ROOT="${FEATURE_ROOT:-${OUT_ROOT}/feature_store}"
FREQUENCY_ANALYSIS_ROOT="${FREQUENCY_ANALYSIS_ROOT:-${OUT_ROOT}/frequency}"
FREQUENCY_REUSE_PER_SAMPLE="${FREQUENCY_REUSE_PER_SAMPLE:-}"

PRETRAINED_MODELS="${PRETRAINED_MODELS:-imagenet_dino100e_backbone,realuw_dino100e_backbone,synthetic5_dino100e_backbone,imagenet_dino100e_dfui_backbone}"
DETECTOR_MODELS="${DETECTOR_MODELS:-imagenet_dino100e_ruod_cascade,realuw_dino100e_ruod_cascade,synthetic5_dino100e_ruod_cascade,imagenet_dino100e_dfui_ruod_cascade}"
ALL_MODELS="${ALL_MODELS:-${PRETRAINED_MODELS},${DETECTOR_MODELS}}"
CKA_REFERENCE_MODEL="${CKA_REFERENCE_MODEL:-imagenet_dino100e_ruod_cascade}"
FEATURE_GPUS="${FEATURE_GPUS:-cuda:2,cuda:3,cuda:6,cuda:7}"
FEATURE_POOLING="${FEATURE_POOLING:-avg}"
SPATIAL_DTYPE="${SPATIAL_DTYPE:-float16}"
SPATIAL_SAMPLES="${SPATIAL_SAMPLES:-0}"
CAM_DEVICES="${CAM_DEVICES:-cuda:0}"
CAM_PARALLEL_MODELS="${CAM_PARALLEL_MODELS:-1}"
CAM_SCORE_THRESHOLD="${CAM_SCORE_THRESHOLD:-0.05}"
CAM_MAX_PREDICTIONS="${CAM_MAX_PREDICTIONS:-30}"
CPU_WORKERS="${CPU_WORKERS:-16}"
FREQUENCY_BAND_WORKERS="${FREQUENCY_BAND_WORKERS:-${CPU_WORKERS}}"
FREQUENCY_MODEL_WORKERS="${FREQUENCY_MODEL_WORKERS:-4}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_BANDS="${RUN_BANDS:-1}"
RUN_FEATURES="${RUN_FEATURES:-1}"
RUN_CKA="${RUN_CKA:-1}"
RUN_FREQUENCY="${RUN_FREQUENCY:-1}"
RUN_CAM="${RUN_CAM:-1}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
require_file() { [[ -s "$1" ]] || { echo "Error: required file is missing or empty: $1" >&2; exit 1; }; }
for switch in "${RUN_SAMPLE}" "${RUN_BANDS}" "${RUN_FEATURES}" "${RUN_CKA}" "${RUN_FREQUENCY}" "${RUN_CAM}"; do
    [[ "${switch}" == 0 || "${switch}" == 1 ]] || { echo "Error: stage switches must be 0 or 1" >&2; exit 1; }
done
require_file "${RUOD_ANN}"
require_file "${FEATURE_MODELS_CONFIG}"
require_file "${CAM_MODELS_CONFIG}"

echo "============================================================"
echo "RUOD analysis suite"
echo "RUN_NAME:               ${RUN_NAME}"
echo "OUT_ROOT:               ${OUT_ROOT}"
echo "Samples / seed:         ${SAMPLES} / ${SEED}"
echo "CAM samples / seed:     ${CAM_SAMPLES} / ${CAM_SEED}"
echo "Frequency policy:       dataset-energy (RGB, q=1/3,2/3)"
echo "Feature models:         ${ALL_MODELS}"
echo "Feature GPUs:           ${FEATURE_GPUS} (four models per wave)"
echo "Frequency band workers: ${FREQUENCY_BAND_WORKERS}; metric workers: ${FREQUENCY_MODEL_WORKERS}"
echo "Frequency analysis root: ${FREQUENCY_ANALYSIS_ROOT}"
echo "CAM aggregation/style:  prediction max / JET / per-image min-max"
echo "CAM output root:        ${CAM_OUTPUT_ROOT}"
echo "============================================================"

if [[ "${RUN_SAMPLE}" == 1 ]]; then
    sample_args=(-m tools.exp_2.backbone_analysis.sample_ruod --annotation-file "${RUOD_ANN}" --image-root "${RUOD_IMAGE_ROOT}" --out-dir "${SAMPLE_ROOT}" --samples "${SAMPLES}" --seed "${SEED}" --materialize "${MATERIALIZE}")
    [[ "${OVERWRITE}" == 1 ]] && sample_args+=(--overwrite)
    python "${sample_args[@]}" 2>&1 | tee "${LOG_ROOT}/01_sample.log"
fi
require_file "${SAMPLE_ROOT}/manifest.jsonl"

if [[ "${RUN_BANDS}" == 1 ]]; then
    band_args=(-m tools.exp_2.backbone_analysis.generate_frequency_bands --manifest "${SAMPLE_ROOT}/manifest.jsonl" --out-dir "${FREQUENCY_ROOT}" --method soft-cpp --band-policy dataset-energy --energy-quantiles 1/3,2/3 --energy-color-space rgb --model-input-mode natural-energy --save-band-stop --copy-clean --png-compress-level 3 --workers "${FREQUENCY_BAND_WORKERS}")
    [[ -n "${ENERGY_CALIBRATION_MANIFEST}" ]] && band_args+=(--calibration-manifest "${ENERGY_CALIBRATION_MANIFEST}")
    [[ "${OVERWRITE}" == 1 ]] && band_args+=(--overwrite)
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "${band_args[@]}" 2>&1 | tee "${LOG_ROOT}/02_frequency_bands.log"
fi
require_file "${FREQUENCY_ROOT}/frequency_manifest.jsonl"

if [[ "${RUN_FEATURES}" == 1 ]]; then
    IFS=',' read -r -a feature_gpus <<< "${FEATURE_GPUS}"
    IFS=',' read -r -a feature_models <<< "${ALL_MODELS}"
    [[ "${#feature_gpus[@]}" -eq 4 ]] || { echo "Error: FEATURE_GPUS must contain exactly four GPUs" >&2; exit 1; }
    [[ "${#feature_models[@]}" -eq 8 ]] || { echo "Error: ALL_MODELS must contain exactly eight models" >&2; exit 1; }
    mkdir -p "${FEATURE_ROOT}/qa" "${OUT_ROOT}/feature_workers"
    if [[ "${OVERWRITE}" == 1 || ! -s "${FEATURE_ROOT}/qa/feature_summary.tsv" ]]; then
        : > "${FEATURE_ROOT}/qa/feature_summary.tsv"
    fi
    feature_failures=0
    for wave in 0 1; do
        pids=()
        worker_models=()
        for slot in 0 1 2 3; do
            index=$((wave * 4 + slot))
            model="${feature_models[$index]}"
            gpu="$(echo "${feature_gpus[$slot]}" | xargs)"
            worker_root="${OUT_ROOT}/feature_workers/${model}"
            worker_log="${LOG_ROOT}/03_extract_${model}.log"
            if [[ -s "${FEATURE_ROOT}/features/${model}/clean/res2.npy" && "${OVERWRITE}" != 1 ]]; then
                echo "REUSE features for ${model}"
                continue
            fi
            [[ ! -e "${worker_root}" || "${OVERWRITE}" == 1 ]] || { echo "Error: worker output exists: ${worker_root}" >&2; exit 1; }
            feature_args=(-m tools.exp_2.backbone_analysis.extract_backbone_features --manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" --models-config "${FEATURE_MODELS_CONFIG}" --models "${model}" --out-dir "${worker_root}" --variants "${FEATURE_VARIANTS}" --layers "${LAYERS}" --device "${gpu}" --pooling "${FEATURE_POOLING}" --save-spatial --spatial-samples "${SPATIAL_SAMPLES}" --spatial-dtype "${SPATIAL_DTYPE}")
            [[ "${OVERWRITE}" == 1 ]] && feature_args+=(--overwrite)
            echo "START feature worker model=${model} gpu=${gpu} wave=${wave}"
            env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "${feature_args[@]}" > "${worker_log}" 2>&1 &
            pids+=("$!")
            worker_models+=("${model}")
        done
        for slot in "${!pids[@]}"; do
            if wait "${pids[$slot]}"; then
                model="${worker_models[$slot]}"
                worker_root="${OUT_ROOT}/feature_workers/${model}"
                for section in features spatial metadata; do
                    source="${worker_root}/${section}/${model}"
                    target="${FEATURE_ROOT}/${section}/${model}"
                    [[ -d "${source}" ]] || { echo "Error: missing ${source}" >&2; exit 1; }
                    [[ ! -e "${target}" ]] || { echo "Error: refusing to replace existing ${target}" >&2; exit 1; }
                    mkdir -p "${FEATURE_ROOT}/${section}"
                    mv -- "${source}" "${target}"
                done
                if [[ ! -s "${FEATURE_ROOT}/qa/feature_summary.tsv" ]]; then
                    cp -- "${worker_root}/qa/feature_summary.tsv" "${FEATURE_ROOT}/qa/feature_summary.tsv"
                else
                    tail -n +2 "${worker_root}/qa/feature_summary.tsv" >> "${FEATURE_ROOT}/qa/feature_summary.tsv"
                fi
                cp -- "${worker_root}/model_load_reports.json" "${FEATURE_ROOT}/qa/model_load_report_${model}.json"
                echo "DONE feature worker model=${model}"
            else
                echo "FAILED feature worker model=${worker_models[$slot]}" >&2
                feature_failures=$((feature_failures + 1))
            fi
        done
        [[ "${feature_failures}" -eq 0 ]] || exit 1
    done
fi
require_file "${FEATURE_ROOT}/qa/feature_summary.tsv"

if [[ "${RUN_CKA}" == 1 ]]; then
    for group in pretrained detector; do
        if [[ "${group}" == pretrained ]]; then models="${PRETRAINED_MODELS}"; reference="${CKA_REFERENCE_MODEL}"; else models="${DETECTOR_MODELS}"; reference="${CKA_REFERENCE_MODEL}"; fi
        cka_args=(-m tools.analysis.compute_same_layer_cka --feature-root "${FEATURE_ROOT}" --models "${models}" --reference-model "${reference}" --layers "${LAYERS}" --variant clean --out-dir "${OUT_ROOT}/cka/${group}")
        [[ "${OVERWRITE}" == 1 ]] && cka_args+=(--overwrite)
        python "${cka_args[@]}" 2>&1 | tee "${LOG_ROOT}/04_cka_${group}.log"
    done
fi

if [[ "${RUN_FREQUENCY}" == 1 ]]; then
    frequency_args=(-m tools.analysis.compute_frequency_metrics --feature-root "${FEATURE_ROOT}" --frequency-manifest "${FREQUENCY_ROOT}/frequency_manifest.jsonl" --models "${ALL_MODELS}" --layers "${LAYERS}" --pretrained-models "${PRETRAINED_MODELS}" --detector-models "${DETECTOR_MODELS}" --variants "${FEATURE_VARIANTS}" --model-workers "${FREQUENCY_MODEL_WORKERS}" --out-dir "${FREQUENCY_ANALYSIS_ROOT}")
    [[ -n "${FREQUENCY_REUSE_PER_SAMPLE}" ]] && frequency_args+=(--reuse-per-sample "${FREQUENCY_REUSE_PER_SAMPLE}")
    [[ "${OVERWRITE}" == 1 ]] && frequency_args+=(--overwrite)
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "${frequency_args[@]}" 2>&1 | tee "${LOG_ROOT}/05_frequency_metrics.log"
fi

if [[ "${RUN_CAM}" == 1 ]]; then
    if [[ "${CAM_SAMPLES}" -eq "${SAMPLES}" ]]; then
        active_cam_sample_root="${SAMPLE_ROOT}"
    else
        if [[ -s "${CAM_SAMPLE_ROOT}/manifest.jsonl" && "${OVERWRITE}" != 1 ]]; then
            echo "REUSE CAM subset: ${CAM_SAMPLE_ROOT}/manifest.jsonl"
        else
            cam_subset_args=(-m tools.analysis.sample_manifest_subset --manifest "${SAMPLE_ROOT}/manifest.jsonl" --out-dir "${CAM_SAMPLE_ROOT}" --samples "${CAM_SAMPLES}" --seed "${CAM_SEED}")
            [[ "${OVERWRITE}" == 1 ]] && cam_subset_args+=(--overwrite)
            python "${cam_subset_args[@]}" 2>&1 | tee "${LOG_ROOT}/06a_cam_subset.log"
        fi
        active_cam_sample_root="${CAM_SAMPLE_ROOT}"
    fi
    env RUOD_ROOT="${RUOD_ROOT}" RUOD_ANN="${RUOD_ANN}" CAM_MODELS_CONFIG="${CAM_MODELS_CONFIG}" SAMPLE_ROOT="${active_cam_sample_root}" CAM_OUT_ROOT="${CAM_OUTPUT_ROOT}" LAYERS="${LAYERS}" CAM_DEVICES="${CAM_DEVICES}" CAM_PARALLEL_MODELS="${CAM_PARALLEL_MODELS}" CAM_SCORE_THRESHOLD="${CAM_SCORE_THRESHOLD}" CAM_MAX_PREDICTIONS="${CAM_MAX_PREDICTIONS}" bash "${SCRIPT_DIR}/run_prediction_cam.sh" 2>&1 | tee "${LOG_ROOT}/06_prediction_cam.log"
fi

cat > "${OUT_ROOT}/COMPLETE.env" <<EOF
STATUS=complete
RUN_NAME=${RUN_NAME}
SAMPLE_ROOT=${SAMPLE_ROOT}
CAM_SAMPLE_ROOT=${CAM_SAMPLE_ROOT}
FREQUENCY_ROOT=${FREQUENCY_ROOT}
FEATURE_ROOT=${FEATURE_ROOT}
CKA_ROOT=${OUT_ROOT}/cka
FREQUENCY_ANALYSIS_ROOT=${FREQUENCY_ANALYSIS_ROOT}
CAM_ROOT=${CAM_OUTPUT_ROOT}/jet_per_image_max
SAMPLES=${SAMPLES}
CAM_SAMPLES=${CAM_SAMPLES}
CKA_REFERENCE_MODEL=${CKA_REFERENCE_MODEL}
FEATURE_GPUS=${FEATURE_GPUS}
SEED=${SEED}
FREQUENCY_POLICY=dataset-energy
CAM_METHOD=prediction-conditioned XGradCAM; max aggregation; JET; per-image/model/layer min-max
EOF
echo "Analysis suite complete: ${OUT_ROOT}"
