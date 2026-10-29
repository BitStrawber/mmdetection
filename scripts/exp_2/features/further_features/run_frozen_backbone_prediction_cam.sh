#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/frozen_backbone_ruod_uiis_20261028}"
CAM_BASE="${CAM_BASE:-/media/HDD2/XCX/exp_2/frozen_backbone_prediction_cam_$(date +%Y%m%d_%H%M%S)}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco}"
UIIS_ROOT="${UIIS_ROOT:-/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco}"
SAMPLES="${SAMPLES:-100}"
SEED="${SEED:-2026}"
DEVICES="${DEVICES:-cuda:4,cuda:5,cuda:6}"
PARALLEL_MODELS="${PARALLEL_MODELS:-3}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE:-30}"
RESUME="${RESUME:-1}"
CHECK_GPU_FREE="${CHECK_GPU_FREE:-1}"

RUNNER="${SCRIPT_DIR}/run_prediction_xgradcam_analysis.sh"
SAMPLER="${SCRIPT_DIR}/sample_fixed_gt_ruod.py"

[[ -s "${RUNNER}" ]] || {
    echo "Error: runner is missing: ${RUNNER}" >&2
    exit 1
}
[[ -s "${SAMPLER}" ]] || {
    echo "Error: sampler is missing: ${SAMPLER}" >&2
    exit 1
}
[[ -f "${OUTPUT_ROOT}/status/matrix.complete" ]] || {
    echo "Error: training matrix is incomplete: ${OUTPUT_ROOT}/status/matrix.complete" >&2
    exit 1
}
[[ -f "${RUOD_ROOT}/annotations/instances_val.json" ]] || {
    echo "Error: missing RUOD val annotation" >&2
    exit 1
}
[[ -f "${UIIS_ROOT}/annotations/instances_val.json" ]] || {
    echo "Error: missing UIIS10K val annotation" >&2
    exit 1
}

if [[ "${CHECK_GPU_FREE}" == 1 ]]; then
    for device in ${DEVICES//,/ }; do
        gpu_id="${device#cuda:}"
        used_mib="$(
            nvidia-smi -i "${gpu_id}" \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits
        )"
        if (( used_mib > 4000 )); then
            echo "Error: GPU ${gpu_id} already uses ${used_mib} MiB." >&2
            echo "Release the occupancy process or set CHECK_GPU_FREE=0." >&2
            exit 1
        fi
    done
fi

mkdir -p \
    "${CAM_BASE}/samples" \
    "${CAM_BASE}/configs" \
    "${CAM_BASE}/logs" \
    "${CAM_BASE}/output"

sample_dataset() {
    local name="$1"
    local dataset_root="$2"
    local sample_root="${CAM_BASE}/samples/${name}_${SAMPLES}"

    if [[ "${RESUME}" == 1 && -s "${sample_root}/manifest.jsonl" ]]; then
        echo "REUSE sample ${sample_root}/manifest.jsonl"
        return
    fi

    python "${SAMPLER}" \
        --annotation-file "${dataset_root}/annotations/instances_val.json" \
        --image-root "${dataset_root}/val" \
        --out-dir "${sample_root}" \
        --samples "${SAMPLES}" \
        --seed "${SEED}" \
        --minimum-instances 1 \
        --materialize none \
        2>&1 | tee "${CAM_BASE}/logs/sample_${name}.log"
}

sample_dataset ruod "${RUOD_ROOT}"
sample_dataset uiis10k "${UIIS_ROOT}"

python - "${OUTPUT_ROOT}" "${CAM_BASE}" <<'PY'
import json
import sys
from pathlib import Path

training_root = Path(sys.argv[1])
cam_base = Path(sys.argv[2])
sources = ("imagenet", "realuw", "synthetic5")

for dataset in ("ruod", "uiis10k"):
    for arch, layer_id, module, layout in (
        ("r50", "res4", "layer3", "channels_first"),
        ("vits", "block12", "blocks.11", "channels_last"),
    ):
        entries = []
        for source in sources:
            task = f"{dataset}_{source}_{arch}"
            work_dir = training_root / "work_dirs" / task
            configs = sorted(
                work_dir.glob("*/vis_data/config.py"),
                key=lambda path: path.stat().st_mtime,
            )
            checkpoints = sorted(
                work_dir.glob("best_coco_*.pth"),
                key=lambda path: path.stat().st_mtime,
            )
            if not configs or not checkpoints:
                raise SystemExit(
                    f"{task}: missing runtime config or best checkpoint"
                )
            entries.append({
                "id": task,
                "kind": "detector",
                "config": str(configs[-1].resolve()),
                "checkpoint": str(checkpoints[-1].resolve()),
                "feature_layout": layout,
                "layers": {layer_id: module},
            })

        output = cam_base / "configs" / f"models.{dataset}.{arch}.json"
        output.write_text(
            json.dumps({"models": entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"MODEL_CONFIG={output}")
PY

run_cam() {
    local dataset="$1"
    local arch="$2"
    local layer="$3"
    local data_root="$4"
    local reference="$5"

    echo "============================================================"
    echo "START dataset=${dataset} arch=${arch} layer=${layer}"
    echo "============================================================"

    env \
        RUOD_ROOT="${data_root}" \
        RUOD_ANN="${data_root}/annotations/instances_val.json" \
        SAMPLE_ROOT="${CAM_BASE}/samples/${dataset}_${SAMPLES}" \
        MODELS_CONFIG="${CAM_BASE}/configs/models.${dataset}.${arch}.json" \
        OUT_ROOT="${CAM_BASE}/output/${dataset}_${arch}" \
        LAYERS="${layer}" \
        CASCADE_STAGE=-1 \
        SCORE_THRESHOLD="${SCORE_THRESHOLD}" \
        INFERENCE_SCORE_THRESHOLD="${SCORE_THRESHOLD}" \
        MAX_PREDICTIONS_PER_IMAGE="${MAX_PREDICTIONS_PER_IMAGE}" \
        MINIMUM_BOX_AREA=4 \
        MATCH_IOU_THRESHOLD=0.50 \
        TARGET_MODE=bbox \
        REFERENCE_MODEL="${reference}" \
        DEVICES="${DEVICES}" \
        PARALLEL_MODELS="${PARALLEL_MODELS}" \
        RESUME="${RESUME}" \
        bash "${RUNNER}"
}

run_cam ruod r50 res4 "${RUOD_ROOT}" ruod_imagenet_r50
run_cam ruod vits block12 "${RUOD_ROOT}" ruod_imagenet_vits
run_cam uiis10k r50 res4 "${UIIS_ROOT}" uiis10k_imagenet_r50
run_cam uiis10k vits block12 "${UIIS_ROOT}" uiis10k_imagenet_vits

cat > "${CAM_BASE}/COMPLETE.env" <<EOF
STATUS=complete
METHOD=prediction_conditioned_xgradcam
TRAINING_ROOT=${OUTPUT_ROOT}
SAMPLES=${SAMPLES}
SEED=${SEED}
RUOD_SAMPLE=${CAM_BASE}/samples/ruod_${SAMPLES}/manifest.jsonl
UIIS_SAMPLE=${CAM_BASE}/samples/uiis10k_${SAMPLES}/manifest.jsonl
OUTPUT=${CAM_BASE}/output
EOF

echo "Prediction CAM matrix complete: ${CAM_BASE}"
