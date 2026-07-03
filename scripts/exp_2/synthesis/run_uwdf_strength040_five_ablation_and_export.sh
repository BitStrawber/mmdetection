#!/usr/bin/env bash
set -euo pipefail

# Run five UWDF SDXL ablation experiments at STRENGTH=0.40 in parallel,
# then build multi-panel comparison grids and upload them.
#
# Experiments:
#   1. e1_image_only_empty_prompt_oldneg
#      image latent only, empty positive prompt, old negative prompt
#   2. e2_text_oldneg
#      image latent + text, old negative prompt
#   3. e3_text_ref_oldneg
#      image latent + text + IP-Adapter reference, old negative prompt
#   4. e4_text_ref_depth_oldneg
#      image latent + text + reference + ControlNet depth, old negative prompt
#   5. e5_text_ref_depth_cleanneg
#      image latent + text + reference + ControlNet depth, cleaner negative prompt
#
# Each experiment uses a different GPU from GPU_IDS and the same selected
# source/reference/depth samples for a fair visual comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SPLIT="${SPLIT:-train}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/${SPLIT}}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf/${SPLIT}}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"

WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_strength040_five_ablation}"
EXP_ROOT="${EXP_ROOT:-${WORK_ROOT}/experiments}"
SELECT_ROOT="${SELECT_ROOT:-${WORK_ROOT}/selected}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_strength040_five_ablation_multigrid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU_IDS="${GPU_IDS:-2 4 5 6 7}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.75}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.65}"
PROMPT="${PROMPT:-a realistic underwater photograph}"
EMPTY_PROMPT="${EMPTY_PROMPT:-}"
OLD_NEGATIVE_PROMPT="${OLD_NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, extra objects, fish, coral, diver, text, watermark, blurry, low quality, worst quality}"
CLEAN_NEGATIVE_PROMPT="${CLEAN_NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, duplicated object, text, watermark, low quality, worst quality}"

RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"
UPLOAD="${UPLOAD:-1}"
TILE_SIZE="${TILE_SIZE:-640}"
GRID_COLUMNS="${GRID_COLUMNS:-3}"
OVERWRITE="${OVERWRITE:-1}"

EXPERIMENTS="e1_image_only_empty_prompt_oldneg e2_text_oldneg e3_text_ref_oldneg e4_text_ref_depth_oldneg e5_text_ref_depth_cleanneg"
LOG_DIR="${WORK_ROOT}/logs"
SELECTED_SOURCE_DIR="${SELECT_ROOT}/source/${SPLIT}"
SELECTED_DEPTH_DIR="${SELECT_ROOT}/depth/${SPLIT}"
SELECTED_REFERENCE_DIR="${SELECT_ROOT}/reference/qingxi"

echo "========================================="
echo "UWDF strength=0.40 five ablation pipeline"
echo "========================================="
echo "UWDF_DIR:         ${UWDF_DIR}"
echo "SOURCE_ROOT:      ${SOURCE_ROOT}"
echo "DEPTH_ROOT:       ${DEPTH_ROOT}"
echo "REFERENCE_ROOT:   ${REFERENCE_ROOT}"
echo "WORK_ROOT:        ${WORK_ROOT}"
echo "EXP_ROOT:         ${EXP_ROOT}"
echo "SELECT_ROOT:      ${SELECT_ROOT}"
echo "OUT_ROOT:         ${OUT_ROOT}"
echo "ARCHIVE_PATH:     ${ARCHIVE_PATH}"
echo "NUM:              ${NUM}"
echo "SEED:             ${SEED}"
echo "GPU_IDS:          ${GPU_IDS}"
echo "SIZE:             ${WIDTH}x${HEIGHT}"
echo "STEPS:            ${STEPS}"
echo "STRENGTH:         ${STRENGTH}"
echo "GUIDANCE_SCALE:   ${GUIDANCE_SCALE}"
echo "IP_ADAPTER_SCALE: ${IP_ADAPTER_SCALE}"
echo "CONTROLNET_SCALE: ${CONTROLNET_SCALE}"
echo "PROMPT:           ${PROMPT}"
echo "OLD_NEG_PROMPT:   ${OLD_NEGATIVE_PROMPT}"
echo "CLEAN_NEG_PROMPT: ${CLEAN_NEGATIVE_PROMPT}"
echo "UPLOAD:           ${UPLOAD}"
echo "RCLONE_DEST:      ${RCLONE_DEST}"
echo "========================================="

if [[ ! -d "${UWDF_DIR}" ]]; then
  echo "Error: UWDF_DIR not found: ${UWDF_DIR}" >&2
  exit 1
fi
if [[ ! -f "${UWDF_DIR}/scripts/run_ipadapter_img2img_generate.sh" ]]; then
  echo "Error: missing UWDF img2img script: ${UWDF_DIR}/scripts/run_ipadapter_img2img_generate.sh" >&2
  exit 1
fi
if [[ ! -f "${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" ]]; then
  echo "Error: missing UWDF controlnet script: ${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Error: SOURCE_ROOT not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DEPTH_ROOT}" ]]; then
  echo "Error: DEPTH_ROOT not found: ${DEPTH_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_ROOT}" ]]; then
  echo "Error: REFERENCE_ROOT not found: ${REFERENCE_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${WORK_ROOT}" "${OUT_ROOT}" "${ARCHIVE_PATH}"
fi
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${SELECTED_SOURCE_DIR}" "${SELECTED_DEPTH_DIR}" "${SELECTED_REFERENCE_DIR}"

echo
echo "Step 1/3: Select shared source/depth/reference samples"
SOURCE_ROOT="${SOURCE_ROOT}" \
DEPTH_ROOT="${DEPTH_ROOT}" \
REFERENCE_ROOT="${REFERENCE_ROOT}" \
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
SELECTED_DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
SELECTED_REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
WORK_ROOT="${WORK_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
python - <<'PY'
from pathlib import Path
import json
import os
import random

source_root = Path(os.environ["SOURCE_ROOT"])
depth_root = Path(os.environ["DEPTH_ROOT"])
reference_root = Path(os.environ["REFERENCE_ROOT"])
source_out = Path(os.environ["SELECTED_SOURCE_DIR"])
depth_out = Path(os.environ["SELECTED_DEPTH_DIR"])
reference_out = Path(os.environ["SELECTED_REFERENCE_DIR"])
work_root = Path(os.environ["WORK_ROOT"])
num = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clear(path):
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

for p in [source_out, depth_out, reference_out]:
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
if len(refs_all) >= len(picked):
    picked_refs = rng.sample(refs_all, len(picked))
else:
    picked_refs = [rng.choice(refs_all) for _ in picked]

records = []
for idx, ((source, depth, rel), ref) in enumerate(zip(picked, picked_refs)):
    source_dst = source_out / rel
    depth_dst = (depth_out / rel).with_suffix(".png")
    ref_dst = reference_out / f"{idx:08d}{ref.suffix.lower()}"
    source_dst.parent.mkdir(parents=True, exist_ok=True)
    depth_dst.parent.mkdir(parents=True, exist_ok=True)
    if source_dst.exists() or source_dst.is_symlink():
        source_dst.unlink()
    if depth_dst.exists() or depth_dst.is_symlink():
        depth_dst.unlink()
    if ref_dst.exists() or ref_dst.is_symlink():
        ref_dst.unlink()
    os.symlink(source, source_dst)
    os.symlink(depth, depth_dst)
    os.symlink(ref, ref_dst)
    records.append({
        "index": idx,
        "relative": str(rel).replace("\\", "/"),
        "class": rel.parts[0] if len(rel.parts) >= 2 else "_flat",
        "source": str(source),
        "depth": str(depth),
        "reference": str(ref),
        "selected_source": str(source_dst),
        "selected_depth": str(depth_dst),
        "selected_reference": str(ref_dst),
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
    "selected_reference_dir": str(reference_out),
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

run_img2img_exp() {
  local exp_name="$1"
  local gpu="$2"
  local prompt="$3"
  local negative_prompt="$4"
  local ip_scale="$5"
  local out_dir="${EXP_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "Launch ${exp_name} on GPU ${gpu}; log=${log_file}"
  (
    cd "${UWDF_DIR}"
    GPU="${gpu}" \
    SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
    REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
    OUT_DIR="${out_dir}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    STRENGTH="${STRENGTH}" \
    GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
    IP_ADAPTER_SCALE="${ip_scale}" \
    STEPS="${STEPS}" \
    LIMIT="${NUM}" \
    SEED="${SEED}" \
    PROMPT="${prompt}" \
    NEGATIVE_PROMPT="${negative_prompt}" \
    SAVE_COMPARISON=0 \
    bash scripts/run_ipadapter_img2img_generate.sh
  ) > "${log_file}" 2>&1 &
}

run_controlnet_exp() {
  local exp_name="$1"
  local gpu="$2"
  local prompt="$3"
  local negative_prompt="$4"
  local ip_scale="$5"
  local control_scale="$6"
  local out_dir="${EXP_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "Launch ${exp_name} on GPU ${gpu}; log=${log_file}"
  (
    cd "${UWDF_DIR}"
    GPU="${gpu}" \
    SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
    DEPTH_DIR="${SELECTED_DEPTH_DIR}" \
    REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
    OUT_DIR="${out_dir}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    STRENGTH="${STRENGTH}" \
    GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
    IP_ADAPTER_SCALE="${ip_scale}" \
    CONTROLNET_SCALE="${control_scale}" \
    STEPS="${STEPS}" \
    LIMIT="${NUM}" \
    SEED="${SEED}" \
    PROMPT="${prompt}" \
    NEGATIVE_PROMPT="${negative_prompt}" \
    SAVE_COMPARISON=0 \
    bash scripts/run_ipadapter_controlnet_depth_generate.sh
  ) > "${log_file}" 2>&1 &
}

echo
echo "Step 2/3: Run five experiments in parallel"
run_img2img_exp "e1_image_only_empty_prompt_oldneg" "${gpu_array[0]}" "${EMPTY_PROMPT}" "${OLD_NEGATIVE_PROMPT}" "0.0"
pid1=$!
run_img2img_exp "e2_text_oldneg" "${gpu_array[1]}" "${PROMPT}" "${OLD_NEGATIVE_PROMPT}" "0.0"
pid2=$!
run_img2img_exp "e3_text_ref_oldneg" "${gpu_array[2]}" "${PROMPT}" "${OLD_NEGATIVE_PROMPT}" "${IP_ADAPTER_SCALE}"
pid3=$!
run_controlnet_exp "e4_text_ref_depth_oldneg" "${gpu_array[3]}" "${PROMPT}" "${OLD_NEGATIVE_PROMPT}" "${IP_ADAPTER_SCALE}" "${CONTROLNET_SCALE}"
pid4=$!
run_controlnet_exp "e5_text_ref_depth_cleanneg" "${gpu_array[4]}" "${PROMPT}" "${CLEAN_NEGATIVE_PROMPT}" "${IP_ADAPTER_SCALE}" "${CONTROLNET_SCALE}"
pid5=$!

failed=0
for pair in \
  "e1_image_only_empty_prompt_oldneg:${pid1}" \
  "e2_text_oldneg:${pid2}" \
  "e3_text_ref_oldneg:${pid3}" \
  "e4_text_ref_depth_oldneg:${pid4}" \
  "e5_text_ref_depth_cleanneg:${pid5}"
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
UPLOAD="${UPLOAD}" \
RCLONE_DEST="${RCLONE_DEST}" \
OVERWRITE=1 \
bash scripts/exp_2/synthesis/export_uwdf_depth_ablation_multigrid_to_gdrive.sh \
  2>&1 | tee "${LOG_DIR}/export_multigrid.log"

echo
echo "Done."
echo "Experiments: ${EXP_ROOT}"
echo "Selection:   ${WORK_ROOT}/selection_manifest.json"
echo "Panels:      ${OUT_ROOT}/multi_panel"
echo "Archive:     ${ARCHIVE_PATH}"
