#!/usr/bin/env bash
set -euo pipefail

# Five-way UWDF ablation for environment-style transfer.
#
# Goal:
#   Keep ImageNet object identity as much as possible while testing whether
#   text prompt, text guidance, blurred reference, and weak IP-Adapter can shift
#   global underwater lighting/environment style.
#
# All five experiments use:
#   source image + depth ControlNet + IP-Adapter reference + text
#
# Experiments:
#   e1_env_prompt_base
#     adjusted environment prompt
#   e2_env_prompt_textstrong
#     adjusted environment prompt + stronger text guidance
#   e3_blur_ref_base
#     default/simple prompt + blurred reference
#   e4_env_textstrong_blur_ref
#     adjusted prompt + stronger text guidance + blurred reference
#   e5_env_textstrong_blur_ref_lowip
#     adjusted prompt + stronger text guidance + blurred reference + lower IP scale

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SPLIT="${SPLIT:-train}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/${SPLIT}}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf/${SPLIT}}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"

WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_environment_style_five_ablation}"
EXP_ROOT="${EXP_ROOT:-${WORK_ROOT}/experiments}"
SELECT_ROOT="${SELECT_ROOT:-${WORK_ROOT}/selected}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_environment_style_five_ablation_multigrid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU_IDS="${GPU_IDS:-2 4 5 6 7}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.30}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.85}"
BASE_IP_ADAPTER_SCALE="${BASE_IP_ADAPTER_SCALE:-0.35}"
LOW_IP_ADAPTER_SCALE="${LOW_IP_ADAPTER_SCALE:-0.15}"
BASE_GUIDANCE_SCALE="${BASE_GUIDANCE_SCALE:-5.0}"
HIGH_GUIDANCE_SCALE="${HIGH_GUIDANCE_SCALE:-8.0}"
BLUR_RADIUS="${BLUR_RADIUS:-28}"
BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE:-64}"

BASE_PROMPT="${BASE_PROMPT:-a realistic underwater photograph}"
ENV_PROMPT="${ENV_PROMPT:-a realistic photograph of the same object with underwater ambient lighting, blue-green water color cast, mild haze, reduced contrast, natural light attenuation, preserve the original object identity, preserve the original object shape}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-changed object identity, deformed object, duplicated object, extra object, fish, coral, diver, cartoon, painting, illustration, text, watermark, low quality, worst quality}"

RESIZE_MODE="${RESIZE_MODE:-pad}"
RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE:-1}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
UPLOAD="${UPLOAD:-1}"
PACKAGE_EXPORT="${PACKAGE_EXPORT:-0}"
TILE_SIZE="${TILE_SIZE:-1024}"
GRID_COLUMNS="${GRID_COLUMNS:-4}"
TILE_MODE="${TILE_MODE:-cover}"
PANEL_FORMAT="${PANEL_FORMAT:-png}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-0}"
OVERWRITE="${OVERWRITE:-1}"

EXPERIMENTS="e1_env_prompt_base e2_env_prompt_textstrong e3_blur_ref_base e4_env_textstrong_blur_ref e5_env_textstrong_blur_ref_lowip"
LOG_DIR="${WORK_ROOT}/logs"
SELECTED_SOURCE_DIR="${SELECT_ROOT}/source/${SPLIT}"
SELECTED_DEPTH_DIR="${SELECT_ROOT}/depth/${SPLIT}"
SELECTED_REFERENCE_RAW_DIR="${SELECT_ROOT}/reference_raw/qingxi"
SELECTED_REFERENCE_BLUR_DIR="${SELECT_ROOT}/reference_blur/qingxi"

echo "========================================="
echo "UWDF environment-style five ablation"
echo "========================================="
echo "UWDF_DIR:              ${UWDF_DIR}"
echo "SOURCE_ROOT:           ${SOURCE_ROOT}"
echo "DEPTH_ROOT:            ${DEPTH_ROOT}"
echo "REFERENCE_ROOT:        ${REFERENCE_ROOT}"
echo "WORK_ROOT:             ${WORK_ROOT}"
echo "EXP_ROOT:              ${EXP_ROOT}"
echo "OUT_ROOT:              ${OUT_ROOT}"
echo "NUM:                   ${NUM}"
echo "SEED:                  ${SEED}"
echo "GPU_IDS:               ${GPU_IDS}"
echo "SIZE:                  ${WIDTH}x${HEIGHT}"
echo "STRENGTH:              ${STRENGTH}"
echo "CONTROLNET_SCALE:      ${CONTROLNET_SCALE}"
echo "BASE_IP_SCALE:         ${BASE_IP_ADAPTER_SCALE}"
echo "LOW_IP_SCALE:          ${LOW_IP_ADAPTER_SCALE}"
echo "BASE_GUIDANCE_SCALE:   ${BASE_GUIDANCE_SCALE}"
echo "HIGH_GUIDANCE_SCALE:   ${HIGH_GUIDANCE_SCALE}"
echo "BLUR_RADIUS:           ${BLUR_RADIUS}"
echo "BLUR_DOWNSAMPLE:       ${BLUR_DOWNSAMPLE}"
echo "RESIZE_MODE:           ${RESIZE_MODE}"
echo "RESTORE_SOURCE_SIZE:   ${RESTORE_SOURCE_SIZE}"
echo "ENV_PROMPT:            ${ENV_PROMPT}"
echo "NEGATIVE_PROMPT:       ${NEGATIVE_PROMPT}"
echo "========================================="

if [[ ! -d "${UWDF_DIR}" ]]; then
  echo "Error: UWDF_DIR not found: ${UWDF_DIR}" >&2
  exit 1
fi
if [[ ! -f "${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" ]]; then
  echo "Error: missing UWDF controlnet script: ${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" || ! -d "${DEPTH_ROOT}" || ! -d "${REFERENCE_ROOT}" ]]; then
  echo "Error: source/depth/reference root missing." >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${WORK_ROOT}" "${OUT_ROOT}" "${ARCHIVE_PATH}"
fi
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${SELECTED_SOURCE_DIR}" "${SELECTED_DEPTH_DIR}" \
  "${SELECTED_REFERENCE_RAW_DIR}" "${SELECTED_REFERENCE_BLUR_DIR}"

echo
echo "Step 1/3: Select shared source/depth/reference samples and build blurred refs"
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
REFERENCE_ROOT="${REFERENCE_ROOT}" \
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
SELECTED_DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
SELECTED_REFERENCE_RAW_DIR="${SELECTED_REFERENCE_RAW_DIR}" \
SELECTED_REFERENCE_BLUR_DIR="${SELECTED_REFERENCE_BLUR_DIR}" \
WORK_ROOT="${WORK_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
BLUR_RADIUS="${BLUR_RADIUS}" \
BLUR_DOWNSAMPLE="${BLUR_DOWNSAMPLE}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import json
import os
import random

source_root = Path(os.environ["SOURCE_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
reference_root = Path(os.environ["REFERENCE_ROOT"])
source_out = Path(os.environ["SELECTED_SOURCE_DIR"])
depth_out = Path(os.environ["SELECTED_DEPTH_DIR"])
ref_raw_out = Path(os.environ["SELECTED_REFERENCE_RAW_DIR"])
ref_blur_out = Path(os.environ["SELECTED_REFERENCE_BLUR_DIR"])
work_root = Path(os.environ["WORK_ROOT"])
num = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
blur_radius = float(os.environ["BLUR_RADIUS"])
blur_downsample = int(os.environ["BLUR_DOWNSAMPLE"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clear(path: Path):
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

for p in [source_out, depth_out, ref_raw_out, ref_blur_out]:
    clear(p)
    p.mkdir(parents=True, exist_ok=True)

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
    source_dst.parent.mkdir(parents=True, exist_ok=True)
    depth_dst.parent.mkdir(parents=True, exist_ok=True)

    for dst in [source_dst, depth_dst, ref_raw_dst, ref_blur_dst]:
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
    "blur_radius": blur_radius,
    "blur_downsample": blur_downsample,
    "records": records,
}
(work_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({
    "selected": len(records),
    "classes": len(set(r["class"] for r in records)),
    "manifest": str(work_root / "selection_manifest.json"),
}, indent=2, ensure_ascii=False))
PY

read -r -a gpu_array <<< "${GPU_IDS}"
if [[ "${#gpu_array[@]}" -lt 5 ]]; then
  echo "Error: need at least 5 GPU ids in GPU_IDS, got: ${GPU_IDS}" >&2
  exit 1
fi

run_exp() {
  local exp_name="$1"
  local gpu="$2"
  local prompt="$3"
  local guidance_scale="$4"
  local ref_dir="$5"
  local ip_scale="$6"
  local out_dir="${EXP_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "Launch ${exp_name} on GPU ${gpu}; log=${log_file}"
  (
    cd "${UWDF_DIR}"
    GPU="${gpu}" \
    SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
    DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
    REFERENCE_DIR="${ref_dir}" \
    OUT_DIR="${out_dir}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    STRENGTH="${STRENGTH}" \
    GUIDANCE_SCALE="${guidance_scale}" \
    IP_ADAPTER_SCALE="${ip_scale}" \
    CONTROLNET_SCALE="${CONTROLNET_SCALE}" \
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

echo
echo "Step 2/3: Run five experiments in parallel"
run_exp "e1_env_prompt_base" "${gpu_array[0]}" "${ENV_PROMPT}" "${BASE_GUIDANCE_SCALE}" "${SELECTED_REFERENCE_RAW_DIR}" "${BASE_IP_ADAPTER_SCALE}"
pid1=$!
run_exp "e2_env_prompt_textstrong" "${gpu_array[1]}" "${ENV_PROMPT}" "${HIGH_GUIDANCE_SCALE}" "${SELECTED_REFERENCE_RAW_DIR}" "${BASE_IP_ADAPTER_SCALE}"
pid2=$!
run_exp "e3_blur_ref_base" "${gpu_array[2]}" "${BASE_PROMPT}" "${BASE_GUIDANCE_SCALE}" "${SELECTED_REFERENCE_BLUR_DIR}" "${BASE_IP_ADAPTER_SCALE}"
pid3=$!
run_exp "e4_env_textstrong_blur_ref" "${gpu_array[3]}" "${ENV_PROMPT}" "${HIGH_GUIDANCE_SCALE}" "${SELECTED_REFERENCE_BLUR_DIR}" "${BASE_IP_ADAPTER_SCALE}"
pid4=$!
run_exp "e5_env_textstrong_blur_ref_lowip" "${gpu_array[4]}" "${ENV_PROMPT}" "${HIGH_GUIDANCE_SCALE}" "${SELECTED_REFERENCE_BLUR_DIR}" "${LOW_IP_ADAPTER_SCALE}"
pid5=$!

failed=0
for pair in \
  "e1_env_prompt_base:${pid1}" \
  "e2_env_prompt_textstrong:${pid2}" \
  "e3_blur_ref_base:${pid3}" \
  "e4_env_textstrong_blur_ref:${pid4}" \
  "e5_env_textstrong_blur_ref_lowip:${pid5}"
do
  name="${pair%%:*}"
  pid="${pair#*:}"
  if wait "${pid}"; then
    echo "OK: ${name}"
  else
    echo "FAILED: ${name}. Check ${LOG_DIR}/${name}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "One or more UWDF experiments failed; skip export." >&2
  exit 1
fi

echo
echo "Step 3/3: Build multi-panel comparison and upload"
EXP_ROOT="${EXP_ROOT}" \
EXPERIMENTS="${EXPERIMENTS}" \
OUT_ROOT="${OUT_ROOT}" \
ARCHIVE_PATH="${ARCHIVE_PATH}" \
LOG_ROOT="${LOG_DIR}" \
DEPTH_ROOT="${SELECTED_DEPTH_DIR}" \
MAX_IMAGES="${NUM}" \
TILE_SIZE="${TILE_SIZE}" \
GRID_COLUMNS="${GRID_COLUMNS}" \
TILE_MODE="${TILE_MODE}" \
PANEL_FORMAT="${PANEL_FORMAT}" \
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL}" \
UPLOAD="${UPLOAD}" \
PACKAGE_EXPORT="${PACKAGE_EXPORT}" \
RCLONE_DEST="${RCLONE_DEST}" \
OVERWRITE=1 \
bash scripts/exp_2/synthesis/export_uwdf_depth_ablation_multigrid_to_gdrive.sh \
  2>&1 | tee "${LOG_DIR}/export_multigrid.log"

echo
echo "Done."
echo "Experiments: ${EXP_ROOT}"
echo "Selection:   ${WORK_ROOT}/selection_manifest.json"
echo "Panels:      ${OUT_ROOT}/multi_panel"
if [[ "${PACKAGE_EXPORT}" == "1" ]]; then
  echo "Archive:     ${ARCHIVE_PATH}"
else
  echo "Archive:     skipped"
fi
