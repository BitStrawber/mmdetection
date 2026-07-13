#!/usr/bin/env bash
set -euo pipefail

# Final UWDF full ImageNet generation.
# Fixed default recipe selected from visual ablations:
#   source img2img + blurred underwater reference + depth ControlNet
#   IP-Adapter style mode, IP scale 2.0, ControlNet scale 0.85, strength 0.75.
#
# Example:
#   conda activate /media/SSD1/conda_envs/uwdf
#   GPU_IDS="3 5 6 7" SPLITS="train val" bash scripts/exp_2/synthesis/run_uwdf_final_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/uwdf_controlnet_ipadapter}"
REF_WORK_ROOT="${REF_WORK_ROOT:-${WORK_ROOT}/uwdf_reference_blur_final}"
REFERENCE_DIR="${REFERENCE_DIR:-${REF_WORK_ROOT}/qingxi}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/uwdf_final_full_$(date +%Y%m%d_%H%M%S)}"

SPLITS="${SPLITS:-train val}"
GPU_IDS_RAW="${GPU_IDS:-3 5 6 7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-2026}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.75}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-8.0}"
IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-2.0}"
IP_ADAPTER_SCALE_MODE="${IP_ADAPTER_SCALE_MODE:-style}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.85}"
CONTROL_GUIDANCE_START="${CONTROL_GUIDANCE_START:-0.0}"
CONTROL_GUIDANCE_END="${CONTROL_GUIDANCE_END:-1.0}"
REFERENCE_MODE="${REFERENCE_MODE:-round_robin}"
RESIZE_MODE="${RESIZE_MODE:-pad}"
RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE:-1}"
PROMPT="${PROMPT:-a realistic underwater photograph}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, unrealistic image, artificial colors, object deformation, changed object identity, extra objects, text, watermark, low quality, worst quality}"

BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE:-64}"
BLUR_RADIUS="${BLUR_RADIUS:-28}"
REF_OVERWRITE="${REF_OVERWRITE:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "Error: no GPUs provided. Set GPU_IDS=\"3 5 6 7\"" >&2
  exit 1
fi
if (( PROCS_PER_GPU < 1 )); then
  echo "Error: PROCS_PER_GPU must be >= 1" >&2
  exit 1
fi
EXPANDED_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  for _ in $(seq 1 "${PROCS_PER_GPU}"); do
    EXPANDED_GPU_IDS+=("${gpu}")
  done
done
NUM_SHARDS="${NUM_SHARDS:-${#EXPANDED_GPU_IDS[@]}}"
if [[ "${NUM_SHARDS}" -ne "${#EXPANDED_GPU_IDS[@]}" ]]; then
  echo "Error: NUM_SHARDS (${NUM_SHARDS}) must equal launched process count (${#EXPANDED_GPU_IDS[@]})." >&2
  exit 1
fi

check_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

count_images() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
}

check_dir "${UWDF_DIR}" "UWDF_DIR"
check_dir "${SOURCE_ROOT}/train" "UWDF train source"
check_dir "${SOURCE_ROOT}/val" "UWDF val source"
check_dir "${DEPTH_ROOT}/train" "UWDF train depth"
check_dir "${DEPTH_ROOT}/val" "UWDF val depth"
check_dir "${REFERENCE_ROOT}" "underwater reference root"
if [[ ! -f "${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" ]]; then
  echo "Error: missing UWDF wrapper: ${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${REF_WORK_ROOT}"

cat <<EOF
=========================================
UWDF final full generation
=========================================
UWDF_DIR:              ${UWDF_DIR}
SOURCE_ROOT:           ${SOURCE_ROOT}
DEPTH_ROOT:            ${DEPTH_ROOT}
REFERENCE_ROOT:        ${REFERENCE_ROOT}
REFERENCE_DIR:         ${REFERENCE_DIR}
WORK_ROOT:             ${WORK_ROOT}
OUT_ROOT:              ${OUT_ROOT}
LOG_DIR:               ${LOG_DIR}
SPLITS:                ${SPLITS}
GPU_IDS:               ${GPU_IDS[*]}
PROCS_PER_GPU:         ${PROCS_PER_GPU}
NUM_SHARDS:            ${NUM_SHARDS}
LIMIT:                 ${LIMIT}
SIZE/STEPS/STRENGTH:   ${WIDTH}x${HEIGHT}/${STEPS}/${STRENGTH}
GUIDANCE_SCALE:        ${GUIDANCE_SCALE}
IP_ADAPTER_SCALE:      ${IP_ADAPTER_SCALE}
IP_ADAPTER_MODE:       ${IP_ADAPTER_SCALE_MODE}
CONTROLNET_SCALE:      ${CONTROLNET_SCALE}
REFERENCE_MODE:        ${REFERENCE_MODE}
RESIZE_MODE:           ${RESIZE_MODE}
RESTORE_SOURCE_SIZE:   ${RESTORE_SOURCE_SIZE}
BLUR_DOWNSAMPLE/RADIUS:${BLUR_DOWNSAMPLE}/${BLUR_RADIUS}
REF_OVERWRITE:         ${REF_OVERWRITE}
RESET_OUTPUTS:         ${RESET_OUTPUTS}
=========================================
EOF

prepare_blur_ref() {
  local existing
  existing="$(count_images "${REFERENCE_DIR}")"
  if [[ "${existing}" != "0" && "${REF_OVERWRITE}" != "1" ]]; then
    echo "Skip blur reference preparation; existing images: ${existing}"
    return
  fi
  echo "Prepare blurred reference images: ${REFERENCE_DIR}"
  REFERENCE_ROOT="${REFERENCE_ROOT}" \
  REFERENCE_DIR="${REFERENCE_DIR}" \
  BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE}" \
  BLUR_RADIUS="${BLUR_RADIUS}" \
  REF_OVERWRITE="${REF_OVERWRITE}" \
  python - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

src = Path(os.environ["REFERENCE_ROOT"])
dst = Path(os.environ["REFERENCE_DIR"])
down = int(os.environ["BLUR_DOWNSAMPLE"])
radius = float(os.environ["BLUR_RADIUS"])
overwrite = os.environ.get("REF_OVERWRITE", "0") == "1"
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
imgs = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if not imgs:
    raise SystemExit(f"No reference images found under {src}")
dst.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(imgs):
    out = dst / f"{i:08d}.png"
    if out.exists() and not overwrite:
        continue
    with Image.open(p) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    small = rgb.resize((down, down), Image.Resampling.BICUBIC)
    blur = small.resize(rgb.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=radius))
    blur.save(out)
print(f"blurred_reference_count={len(list(dst.glob('*.png')))}")
PY
}

run_split() {
  local split="$1"
  local source_dir="${SOURCE_ROOT}/${split}"
  local depth_dir="${DEPTH_ROOT}/${split}"
  local out_dir="${OUT_ROOT}/${split}"
  local split_log_dir="${LOG_DIR}/${split}"
  mkdir -p "${split_log_dir}" "${out_dir}"

  if [[ "${RESET_OUTPUTS}" == "1" ]]; then
    case "${out_dir}" in
      "${WORK_ROOT}"/uwdf_controlnet_ipadapter/*|"${OUT_ROOT}"/*)
        echo "Reset output dir: ${out_dir}"
        rm -rf "${out_dir}"
        mkdir -p "${out_dir}"
        ;;
      *)
        echo "Refuse to reset unexpected OUT_DIR: ${out_dir}" >&2
        return 1
        ;;
    esac
  fi

  echo
  echo "========================================="
  echo "Launch UWDF ${split}: ${NUM_SHARDS} shards"
  echo "source: ${source_dir} ($(count_images "${source_dir}"))"
  echo "depth:  ${depth_dir} ($(count_images "${depth_dir}"))"
  echo "out:    ${out_dir}"
  echo "========================================="

  pids=()
  for idx in "${!EXPANDED_GPU_IDS[@]}"; do
    local gpu="${EXPANDED_GPU_IDS[$idx]}"
    local shard_log="${split_log_dir}/uwdf_${split}_shard${idx}of${NUM_SHARDS}.log"
    echo "  shard ${idx}/${NUM_SHARDS} -> GPU ${gpu}; log=${shard_log}"
    (
      cd "${UWDF_DIR}"
      GPU="${gpu}" \
      SOURCE_DIR="${source_dir}" \
      DEPTH_DIR="${depth_dir}" \
      REFERENCE_DIR="${REFERENCE_DIR}" \
      REFERENCE_MODE="${REFERENCE_MODE}" \
      OUT_DIR="${out_dir}" \
      LIMIT="${LIMIT}" \
      NUM_SHARDS="${NUM_SHARDS}" \
      SHARD_INDEX="${idx}" \
      SEED="${SEED}" \
      HEIGHT="${HEIGHT}" \
      WIDTH="${WIDTH}" \
      STEPS="${STEPS}" \
      STRENGTH="${STRENGTH}" \
      GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
      IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE}" \
      IP_ADAPTER_SCALE_MODE="${IP_ADAPTER_SCALE_MODE}" \
      CONTROLNET_SCALE="${CONTROLNET_SCALE}" \
      CONTROL_GUIDANCE_START="${CONTROL_GUIDANCE_START}" \
      CONTROL_GUIDANCE_END="${CONTROL_GUIDANCE_END}" \
      RESIZE_MODE="${RESIZE_MODE}" \
      RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE}" \
      PROMPT="${PROMPT}" \
      NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
      SAVE_COMPARISON=0 \
      bash scripts/run_ipadapter_controlnet_depth_generate.sh
    ) > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  local generated_count
  generated_count="$(count_images "${out_dir}/generated")"
  echo "UWDF ${split} generated_count=${generated_count}" | tee "${split_log_dir}/completion.txt"
  if [[ "${failed}" != "0" ]]; then
    echo "Warning: one or more UWDF ${split} shards failed. Check ${split_log_dir}" >&2
    return 1
  fi
  return 0
}

prepare_blur_ref

failed_any=0
for split in ${SPLITS}; do
  if ! run_split "${split}"; then
    failed_any=1
  fi
done

cat <<EOF
=========================================
UWDF final full generation finished
=========================================
OUT_ROOT:      ${OUT_ROOT}
train count:   $(count_images "${OUT_ROOT}/train/generated")
val count:     $(count_images "${OUT_ROOT}/val/generated")
logs:          ${LOG_DIR}
=========================================
EOF

if [[ "${failed_any}" != "0" ]]; then
  exit 1
fi