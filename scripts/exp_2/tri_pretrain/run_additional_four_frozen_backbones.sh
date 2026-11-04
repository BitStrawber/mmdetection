#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-detector}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/frozen_backbone_additional_20261104}"
CONFIG_ROOT="${CONFIG_ROOT:-${OUTPUT_ROOT}/configs}"
BACKBONE_ROOT="${BACKBONE_ROOT:-${OUTPUT_ROOT}/backbones}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/work_dirs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
PREPARE="${REPO_ROOT}/tools/exp_2/frozen_backbone_training/prepare_frozen_training.py"

R50_GPUS="${R50_GPUS:-2,3}"
VITS_RUOD_GPUS="${VITS_RUOD_GPUS:-4,5}"
VITS_UIIS_GPUS="${VITS_UIIS_GPUS:-6,7}"

R50_RUOD_CONFIG="${R50_RUOD_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py}"
R50_UIIS_CONFIG="${R50_UIIS_CONFIG:-configs/exp_2/mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py}"
VITS_RUOD_CONFIG="${VITS_RUOD_CONFIG:-configs/exp_2/tri_pretrain/cascade-rcnn_vit-small_dino_fpn_24e_ruod_control100k.py}"
VITS_UIIS_CONFIG="${VITS_UIIS_CONFIG:-configs/exp_2/tri_pretrain/mask-rcnn_vit-small_dino_fpn_24e_uiis10k_control100k.py}"

DFUI_R50="${DFUI_R50:-/media/HDD2/XCX/exp_2/hf_downloads/realuw_selected_best/realuw_random_task_best/dfui_realuw_task/imagenet_r50/DFUI/weights/best_coco_bbox_mAP_epoch_38.pth}"
SFUI_R50="${SFUI_R50:-/media/HDD2/XCX/exp_2/hf_downloads/realuw_selected_best/realuw_random_task_best/sfui_realuw_random/imagenet_r50/SFUI/weights/best_coco_segm_mAP_epoch_38.pth}"
SFUI_VITS="${SFUI_VITS:-/media/HDD2/XCX/exp_2/hf_downloads/realuw_selected_best/realuw_random_task_best/sfui_realuw_random/imagenet_vits/SFUI/weights/best_coco_segm_mAP_epoch_40.pth}"
DFUI_VITS="${DFUI_VITS:-/media/HDD2/XCX/exp_2/hf_downloads/realuw_selected_best/realuw_random_task_best/dfui_realuw_random/imagenet_vits/DFUI/weights/best_coco_bbox_mAP_epoch_36.pth}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-0}"
RESUME="${RESUME:-1}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    set +u
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
fi

mkdir -p "${CONFIG_ROOT}" "${BACKBONE_ROOT}" "${WORK_ROOT}" "${LOG_ROOT}"

for file in \
    "${R50_RUOD_CONFIG}" "${R50_UIIS_CONFIG}" \
    "${VITS_RUOD_CONFIG}" "${VITS_UIIS_CONFIG}" \
    "${DFUI_R50}" "${SFUI_R50}" "${SFUI_VITS}" "${DFUI_VITS}"; do
    [[ -s "${file}" ]] || {
        echo "Error: missing required file: ${file}" >&2
        exit 1
    }
done

prepare_one() {
    local name="$1"
    local base_config="$2"
    local detector_checkpoint="$3"
    local backbone_type="$4"
    local save_best="$5"
    local config_output="${CONFIG_ROOT}/${name}.py"
    local backbone_output="${BACKBONE_ROOT}/${name}_backbone.pth"

    if [[ -s "${config_output}" && -s "${backbone_output}" ]]; then
        echo "REUSE prepared model=${name}"
        return
    fi

    local overwrite_args=()
    [[ "${PREPARE_OVERWRITE}" == 1 ]] && overwrite_args+=(--overwrite)
    python "${PREPARE}" prepare \
        --name "${name}" \
        --base-config "${base_config}" \
        --detector-checkpoint "${detector_checkpoint}" \
        --backbone-output "${backbone_output}" \
        --config-output "${config_output}" \
        --backbone-type "${backbone_type}" \
        --save-best "${save_best}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${LOG_ROOT}/prepare_${name}.log"
}

gpu_count() {
    local devices="$1"
    local -a values
    IFS=',' read -r -a values <<< "${devices}"
    echo "${#values[@]}"
}

run_train_job() {
    local name="$1"
    local config="${CONFIG_ROOT}/${name}.py"
    local backbone="${BACKBONE_ROOT}/${name}_backbone.pth"
    local work_dir="${WORK_ROOT}/${name}"
    local log="${LOG_ROOT}/train_${name}.log"
    local devices="$2"
    local port="$3"
    local num_gpus
    num_gpus="$(gpu_count "${devices}")"

    if [[ -s "${work_dir}/COMPLETE.env" ]]; then
        echo "SKIP completed model=${name}"
        return
    fi

    rm -f "${work_dir}/COMPLETE.env"
    mkdir -p "${work_dir}"
    local train_args=(
        "${config}"
        "${num_gpus}"
        --work-dir "${work_dir}"
    )
    if [[ "${RESUME}" == 1 && -s "${work_dir}/latest.pth" ]]; then
        train_args+=(--resume)
    fi

    echo "START model=${name} devices=${devices} port=${port}"
    CUDA_VISIBLE_DEVICES="${devices}" PORT="${port}" \
        bash tools/dist_train.sh "${train_args[@]}" \
        2>&1 | tee "${log}"

    local best
    best="$(find "${work_dir}" -maxdepth 1 -type f -name 'best_coco_*.pth' \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    [[ -s "${best}" ]] || {
        echo "Error: no best checkpoint produced for ${name}" >&2
        exit 1
    }

    python "${PREPARE}" verify \
        --checkpoint "${best}" \
        --backbone "${backbone}" \
        > "${LOG_ROOT}/verify_${name}.log" 2>&1

    cat > "${work_dir}/COMPLETE.env" <<EOF
STATUS=complete
MODEL=${name}
CONFIG=${config}
BACKBONE=${backbone}
BEST_CHECKPOINT=${best}
GPUS=${devices}
EOF
    echo "DONE model=${name} best=${best}"
}

if [[ "${RUN_PREPARE}" == 1 ]]; then
    prepare_one ruod_dfui_r50 "${R50_RUOD_CONFIG}" "${DFUI_R50}" resnet50 coco/bbox_mAP
    prepare_one uiis10k_sfui_r50 "${R50_UIIS_CONFIG}" "${SFUI_R50}" resnet50 coco/segm_mAP
    prepare_one ruod_dfui_vits "${VITS_RUOD_CONFIG}" "${DFUI_VITS}" vit_small coco/bbox_mAP
    prepare_one uiis10k_sfui_vits "${VITS_UIIS_CONFIG}" "${SFUI_VITS}" vit_small coco/segm_mAP
fi

if [[ "${RUN_TRAIN}" == 1 ]]; then
    run_r50_serial() {
        # Both R50 jobs use the same GPU pair and are intentionally serial.
        run_train_job ruod_dfui_r50 "${R50_GPUS}" 29510
        run_train_job uiis10k_sfui_r50 "${R50_GPUS}" 29511
    }

    # Start the R50 serial queue and both ViT jobs at the same time.
    run_r50_serial &
    r50_pid=$!

    run_train_job ruod_dfui_vits "${VITS_RUOD_GPUS}" 29512 &
    vit_ruod_pid=$!
    run_train_job uiis10k_sfui_vits "${VITS_UIIS_GPUS}" 29513 &
    vit_uiis_pid=$!

    failed=0
    wait "${r50_pid}" || failed=1
    wait "${vit_ruod_pid}" || failed=1
    wait "${vit_uiis_pid}" || failed=1
    if [[ "${failed}" -ne 0 ]]; then
        exit 1
    fi
fi

echo "Additional frozen-backbone training complete: ${OUTPUT_ROOT}"
