#!/usr/bin/env bash
set -euo pipefail

# Run seven UWDF SDXL img2img condition-linkage ablations, then export grids.
# Layout intent:
#   conditions: source | raw underwater reference | processed blur reference | depth
#   results: original stable | style only | depth only | style+depth |
#            text-style linked | text-depth linked | text-style-depth linked

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SPLIT="${SPLIT:-train}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/${SPLIT}}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf/${SPLIT}}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"

WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_condition_linkage_seven_ablation}"
EXP_ROOT="${EXP_ROOT:-${WORK_ROOT}/experiments}"
SELECT_ROOT="${SELECT_ROOT:-${WORK_ROOT}/selected}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_condition_linkage_seven_ablation_grid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
LOG_DIR="${WORK_ROOT}/logs"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.75}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-8.0}"
IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.35}"
IP_ADAPTER_SCALE_MODE="${IP_ADAPTER_SCALE_MODE:-style}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.85}"
CONTROL_GUIDANCE_START="${CONTROL_GUIDANCE_START:-0.0}"
CONTROL_GUIDANCE_END="${CONTROL_GUIDANCE_END:-1.0}"
REF_INPUT_MODE="${REF_INPUT_MODE:-blur}"
REFERENCE_MODE="${REFERENCE_MODE:-round_robin}"
RESIZE_MODE="${RESIZE_MODE:-pad}"
RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE:-1}"

LIGHTFIELD_SIGMAS="${LIGHTFIELD_SIGMAS:-15 60 90}"
LIGHTFIELD_RESIZE_RATIO="${LIGHTFIELD_RESIZE_RATIO:-0.3}"
BLUR_RADIUS="${BLUR_RADIUS:-28}"
BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE:-64}"

BASE_PROMPT="${BASE_PROMPT:-a realistic underwater photograph}"
STYLE_LINK_PROMPT="${STYLE_LINK_PROMPT:-a realistic underwater photograph, use the reference image as the guidance for the global underwater appearance and scene-level environmental direction of the entire source scene}"
DEPTH_LINK_PROMPT="${DEPTH_LINK_PROMPT:-a realistic underwater photograph, use the depth map to guide spatial structure, scene geometry, object layout, and depth-aware underwater appearance across the source scene}"
STYLE_DEPTH_LINK_PROMPT="${STYLE_DEPTH_LINK_PROMPT:-a realistic underwater photograph, use the reference image as the guidance for the global underwater appearance and scene-level environmental direction of the entire source scene, and use the depth map to guide spatial structure, scene geometry, object layout, and depth-aware underwater appearance across the source scene}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, unrealistic image, artificial colors, object deformation, changed object identity, extra objects, text, watermark, low quality, worst quality}"

RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
UPLOAD="${UPLOAD:-1}"
OVERWRITE="${OVERWRITE:-1}"
TILE_SIZE="${TILE_SIZE:-1024}"
CONDITION_TILE_SIZE="${CONDITION_TILE_SIZE:-248}"
LABEL_H="${LABEL_H:-32}"
PANEL_FORMAT="${PANEL_FORMAT:-png}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-0}"
TILE_MODE="${TILE_MODE:-cover}"

EXPERIMENTS="e1_original_stable e2_style_only e3_depth_only e4_style_depth e5_text_style_linked e6_text_depth_linked e7_text_style_depth_linked"
SELECTED_SOURCE_DIR="${SELECT_ROOT}/source/${SPLIT}"
SELECTED_DEPTH_DIR="${SELECT_ROOT}/depth/${SPLIT}"
SELECTED_REFERENCE_RAW_DIR="${SELECT_ROOT}/reference_raw/qingxi"
SELECTED_REFERENCE_BLUR_DIR="${SELECT_ROOT}/reference_blur/qingxi"
SELECTED_REFERENCE_LIGHTFIELD_DIR="${SELECT_ROOT}/reference_lightfield/qingxi"
case "${REF_INPUT_MODE}" in
  raw)
    SELECTED_REFERENCE_RUN_DIR="${SELECTED_REFERENCE_RAW_DIR}"
    ;;
  blur)
    SELECTED_REFERENCE_RUN_DIR="${SELECTED_REFERENCE_BLUR_DIR}"
    ;;
  lightfield)
    SELECTED_REFERENCE_RUN_DIR="${SELECTED_REFERENCE_LIGHTFIELD_DIR}"
    ;;
  *)
    echo "Error: REF_INPUT_MODE must be raw, blur, or lightfield, got: ${REF_INPUT_MODE}" >&2
    exit 1
    ;;
esac

cat <<EOF
=========================================
UWDF condition-linkage seven ablation
=========================================
UWDF_DIR:              ${UWDF_DIR}
SOURCE_ROOT:           ${SOURCE_ROOT}
DEPTH_ROOT:            ${DEPTH_ROOT}
REFERENCE_ROOT:        ${REFERENCE_ROOT}
WORK_ROOT:             ${WORK_ROOT}
EXP_ROOT:              ${EXP_ROOT}
OUT_ROOT:              ${OUT_ROOT}
NUM/SEED:              ${NUM}/${SEED}
GPU_IDS:               ${GPU_IDS}
SIZE/STEPS/STRENGTH:   ${WIDTH}x${HEIGHT}/${STEPS}/${STRENGTH}
GUIDANCE_SCALE:        ${GUIDANCE_SCALE}
IP_ADAPTER_SCALE:      ${IP_ADAPTER_SCALE}
IP_ADAPTER_MODE:       ${IP_ADAPTER_SCALE_MODE}
CONTROLNET_SCALE:      ${CONTROLNET_SCALE}
REF_INPUT_MODE:        ${REF_INPUT_MODE}
REF_RUN_DIR:           ${SELECTED_REFERENCE_RUN_DIR}
REFERENCE_MODE:        ${REFERENCE_MODE}
LIGHTFIELD_SIGMAS:     ${LIGHTFIELD_SIGMAS}
LIGHTFIELD_RATIO:      ${LIGHTFIELD_RESIZE_RATIO}
RESIZE_MODE:           ${RESIZE_MODE}
RESTORE_SOURCE_SIZE:   ${RESTORE_SOURCE_SIZE}
UPLOAD:                ${UPLOAD}
RCLONE_DEST:           ${RCLONE_DEST}
=========================================
EOF

if [[ ! -d "${UWDF_DIR}" ]]; then
  echo "Error: UWDF_DIR not found: ${UWDF_DIR}" >&2
  exit 1
fi
if [[ ! -f "${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" ]]; then
  echo "Error: missing UWDF controlnet wrapper: ${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" >&2
  exit 1
fi
for required in "${SOURCE_ROOT}" "${DEPTH_ROOT}" "${REFERENCE_ROOT}"; do
  if [[ ! -d "${required}" ]]; then
    echo "Error: required directory not found: ${required}" >&2
    exit 1
  fi
done

GPU_IDS="${GPU_IDS//,/ }"
read -r -a gpu_array <<< "${GPU_IDS}"
if [[ "${#gpu_array[@]}" -lt 1 ]]; then
  echo "Error: GPU_IDS is empty" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${WORK_ROOT}" "${OUT_ROOT}" "${ARCHIVE_PATH}"
fi
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${SELECTED_SOURCE_DIR}" "${SELECTED_DEPTH_DIR}" \
  "${SELECTED_REFERENCE_RAW_DIR}" "${SELECTED_REFERENCE_BLUR_DIR}" "${SELECTED_REFERENCE_LIGHTFIELD_DIR}"

echo
echo "Step 1/3: Select shared source/depth/reference samples and build blur reference variants"
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
REFERENCE_ROOT="${REFERENCE_ROOT}" \
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
SELECTED_DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
SELECTED_REFERENCE_RAW_DIR="${SELECTED_REFERENCE_RAW_DIR}" \
SELECTED_REFERENCE_BLUR_DIR="${SELECTED_REFERENCE_BLUR_DIR}" \
SELECTED_REFERENCE_LIGHTFIELD_DIR="${SELECTED_REFERENCE_LIGHTFIELD_DIR}" \
WORK_ROOT="${WORK_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
BLUR_RADIUS="${BLUR_RADIUS}" \
BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE}" \
LIGHTFIELD_SIGMAS="${LIGHTFIELD_SIGMAS}" \
LIGHTFIELD_RESIZE_RATIO="${LIGHTFIELD_RESIZE_RATIO}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import json
import numpy as np
import os
import random

source_root = Path(os.environ["SOURCE_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
reference_root = Path(os.environ["REFERENCE_ROOT"])
source_out = Path(os.environ["SELECTED_SOURCE_DIR"])
depth_out = Path(os.environ["SELECTED_DEPTH_DIR"])
ref_raw_out = Path(os.environ["SELECTED_REFERENCE_RAW_DIR"])
ref_blur_out = Path(os.environ["SELECTED_REFERENCE_BLUR_DIR"])
ref_lightfield_out = Path(os.environ["SELECTED_REFERENCE_LIGHTFIELD_DIR"])
work_root = Path(os.environ["WORK_ROOT"])
num = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
blur_radius = float(os.environ["BLUR_RADIUS"])
blur_downsample = int(os.environ["BLUR_DOWNSAMPLE"])
lightfield_sigmas = [float(x) for x in os.environ["LIGHTFIELD_SIGMAS"].split()]
lightfield_resize_ratio = float(os.environ["LIGHTFIELD_RESIZE_RATIO"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clear(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass

def uwnr_luminance_field(rgb: Image.Image, sigmas, resize_ratio: float) -> Image.Image:
    width, height = rgb.size
    small_size = (max(1, int(round(width * resize_ratio))), max(1, int(round(height * resize_ratio))))
    small = rgb.resize(small_size, Image.Resampling.BICUBIC)
    luminance = np.ones_like(np.asarray(small), dtype=np.float32)
    for sigma in sigmas:
        blurred = small.filter(ImageFilter.GaussianBlur(radius=sigma))
        arr = np.asarray(blurred, dtype=np.float32)
        with np.errstate(divide="ignore"):
            arr = np.log10(arr)
        arr = np.nan_to_num(arr, neginf=0.0, posinf=255.0)
        arr = np.clip(arr, 0.0, 255.0)
        luminance += arr
    luminance = luminance / max(1, len(sigmas))
    low = float(np.min(luminance))
    high = float(np.max(luminance))
    field = (luminance - low) / (high - low + 0.0001)
    field = np.uint8(np.clip(field * 255.0, 0, 255))
    return Image.fromarray(field).resize((width, height), Image.Resampling.BICUBIC)

for path in [source_out, depth_out, ref_raw_out, ref_blur_out, ref_lightfield_out]:
    clear(path)
    path.mkdir(parents=True, exist_ok=True)

by_cls = {}
for p in source_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        rel = p.relative_to(source_root)
        cls = rel.parts[0] if len(rel.parts) >= 2 else "_flat"
        depth = (depth_root / rel).with_suffix(".png")
        if depth.exists():
            by_cls.setdefault(cls, []).append((p, depth, rel))

classes = sorted(by_cls)
if not classes:
    raise RuntimeError(f"No source/depth pairs found under {source_root} and {depth_root}")

rng = random.Random(seed)
if len(classes) >= num:
    picked_classes = rng.sample(classes, num)
    picked = [rng.choice(sorted(by_cls[cls], key=lambda x: str(x[0]))) for cls in picked_classes]
else:
    all_pairs = sorted((x for xs in by_cls.values() for x in xs), key=lambda x: str(x[0]))
    picked = rng.sample(all_pairs, min(num, len(all_pairs)))

refs_all = sorted(p for p in reference_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if not refs_all:
    raise RuntimeError(f"No reference images found under {reference_root}")
picked_refs = rng.sample(refs_all, len(picked)) if len(refs_all) >= len(picked) else [rng.choice(refs_all) for _ in picked]

records = []
for idx, ((source, depth, rel), ref) in enumerate(zip(picked, picked_refs)):
    source_dst = source_out / rel
    depth_dst = (depth_out / rel).with_suffix(".png")
    ref_raw_dst = ref_raw_out / f"{idx:08d}{ref.suffix.lower()}"
    ref_blur_dst = ref_blur_out / f"{idx:08d}.png"
    ref_lightfield_dst = ref_lightfield_out / f"{idx:08d}.png"
    source_dst.parent.mkdir(parents=True, exist_ok=True)
    depth_dst.parent.mkdir(parents=True, exist_ok=True)
    for dst in [source_dst, depth_dst, ref_raw_dst, ref_blur_dst, ref_lightfield_dst]:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    os.symlink(source, source_dst)
    os.symlink(depth, depth_dst)
    os.symlink(ref, ref_raw_dst)
    with Image.open(ref) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    small = rgb.resize((blur_downsample, blur_downsample), Image.Resampling.BICUBIC)
    blur = small.resize(rgb.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blur.save(ref_blur_dst)
    lightfield = uwnr_luminance_field(rgb, lightfield_sigmas, lightfield_resize_ratio)
    lightfield.save(ref_lightfield_dst)
    records.append({
        "index": idx,
        "relative": str(rel).replace("\\", "/"),
        "class": rel.parts[0] if len(rel.parts) >= 2 else "_flat",
        "source": str(source),
        "depth": str(depth),
        "reference_raw": str(ref),
        "selected_source": str(source_dst),
        "selected_depth": str(depth_dst),
        "selected_reference_raw": str(ref_raw_dst),
        "selected_reference_blur": str(ref_blur_dst),
        "selected_reference_lightfield": str(ref_lightfield_dst),
    })

manifest = {
    "num": len(records),
    "requested_num": num,
    "seed": seed,
    "source_root": str(source_root),
    "depth_root": str(depth_root),
    "reference_root": str(reference_root),
    "selected_source_dir": str(source_out),
    "selected_depth_dir": str(depth_out),
    "selected_reference_raw_dir": str(ref_raw_out),
    "selected_reference_blur_dir": str(ref_blur_out),
    "selected_reference_lightfield_dir": str(ref_lightfield_out),
    "blur_radius": blur_radius,
    "blur_downsample": blur_downsample,
    "lightfield_sigmas": lightfield_sigmas,
    "lightfield_resize_ratio": lightfield_resize_ratio,
    "records": records,
}
(work_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"selected": len(records), "manifest": str(work_root / "selection_manifest.json")}, indent=2))
PY

run_exp() {
  local exp_name="$1"
  local gpu="$2"
  local prompt="$3"
  local ip_scale="$4"
  local control_scale="$5"
  local out_dir="${EXP_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "Launch ${exp_name} on GPU ${gpu}; ip=${ip_scale}; depth=${control_scale}; log=${log_file}"
  (
    cd "${UWDF_DIR}"
    GPU="${gpu}" \
    SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
    DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
    REFERENCE_DIR="${SELECTED_REFERENCE_RUN_DIR}" \
    REFERENCE_MODE="${REFERENCE_MODE}" \
    OUT_DIR="${out_dir}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    STRENGTH="${STRENGTH}" \
    GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
    IP_ADAPTER_SCALE="${ip_scale}" \
    IP_ADAPTER_SCALE_MODE="${IP_ADAPTER_SCALE_MODE}" \
    CONTROLNET_SCALE="${control_scale}" \
    CONTROL_GUIDANCE_START="${CONTROL_GUIDANCE_START}" \
    CONTROL_GUIDANCE_END="${CONTROL_GUIDANCE_END}" \
    RESIZE_MODE="${RESIZE_MODE}" \
    RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE}" \
    STEPS="${STEPS}" \
    LIMIT="${NUM}" \
    SEED="${SEED}" \
    PROMPT="${prompt}" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
    SAVE_COMPARISON=0 \
    bash scripts/run_ipadapter_controlnet_depth_generate.sh
  ) > "${log_file}" 2>&1 &
}

exp_names=(
  e1_original_stable
  e2_style_only
  e3_depth_only
  e4_style_depth
  e5_text_style_linked
  e6_text_depth_linked
  e7_text_style_depth_linked
)
exp_prompts=(
  "${BASE_PROMPT}"
  "${BASE_PROMPT}"
  "${BASE_PROMPT}"
  "${BASE_PROMPT}"
  "${STYLE_LINK_PROMPT}"
  "${DEPTH_LINK_PROMPT}"
  "${STYLE_DEPTH_LINK_PROMPT}"
)
exp_ip_scales=(0.0 "${IP_ADAPTER_SCALE}" 0.0 "${IP_ADAPTER_SCALE}" "${IP_ADAPTER_SCALE}" 0.0 "${IP_ADAPTER_SCALE}")
exp_control_scales=(0.0 0.0 "${CONTROLNET_SCALE}" "${CONTROLNET_SCALE}" 0.0 "${CONTROLNET_SCALE}" "${CONTROLNET_SCALE}")

echo
echo "Step 2/3: Run seven experiments with available GPUs"
pids=()
pid_names=()
for i in "${!exp_names[@]}"; do
  gpu="${gpu_array[$((i % ${#gpu_array[@]}))]}"
  run_exp "${exp_names[$i]}" "${gpu}" "${exp_prompts[$i]}" "${exp_ip_scales[$i]}" "${exp_control_scales[$i]}"
  pids+=("$!")
  pid_names+=("${exp_names[$i]}")
  if [[ "${#pids[@]}" -ge "${#gpu_array[@]}" ]]; then
    failed=0
    for j in "${!pids[@]}"; do
      if wait "${pids[$j]}"; then
        echo "OK: ${pid_names[$j]}"
      else
        echo "FAILED: ${pid_names[$j]}. Check ${LOG_DIR}/${pid_names[$j]}.log" >&2
        failed=1
      fi
    done
    if [[ "${failed}" != "0" ]]; then
      exit 1
    fi
    pids=()
    pid_names=()
  fi
done

failed=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "OK: ${pid_names[$j]}"
  else
    echo "FAILED: ${pid_names[$j]}. Check ${LOG_DIR}/${pid_names[$j]}.log" >&2
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  exit 1
fi

echo
echo "Step 3/3: Build condition-linkage grids and upload"
EXP_ROOT="${EXP_ROOT}" \
EXPERIMENTS="${EXPERIMENTS}" \
SELECTION_MANIFEST="${WORK_ROOT}/selection_manifest.json" \
OUT_ROOT="${OUT_ROOT}" \
ARCHIVE_PATH="${ARCHIVE_PATH}" \
LOG_ROOT="${LOG_DIR}" \
MAX_IMAGES="${NUM}" \
TILE_SIZE="${TILE_SIZE}" \
CONDITION_TILE_SIZE="${CONDITION_TILE_SIZE}" \
LABEL_H="${LABEL_H}" \
TILE_MODE="${TILE_MODE}" \
PANEL_FORMAT="${PANEL_FORMAT}" \
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL}" \
UPLOAD="${UPLOAD}" \
RCLONE_DEST="${RCLONE_DEST}" \
OVERWRITE=1 \
bash scripts/exp_2/synthesis/export_uwdf_condition_linkage_grid_to_gdrive.sh \
  2>&1 | tee "${LOG_DIR}/export_condition_linkage_grid.log"

echo
echo "Done."
echo "Experiments: ${EXP_ROOT}"
echo "Selection:   ${WORK_ROOT}/selection_manifest.json"
echo "Panels:      ${OUT_ROOT}/panels"
echo "Archive:     ${ARCHIVE_PATH}"