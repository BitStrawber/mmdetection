#!/usr/bin/env bash
set -euo pipefail

# Prepare SSD-side sources and temporary inputs for the five ImageNet
# underwater synthesis baselines:
#
#   1. uwnr                  -> source copy + MegaDepth + flat clean/depth pairs + RUOD refs
#   2. syreanet              -> source copy + resized flat input for SyreaNet test.py
#   3. syreanet_synthesis    -> source copy + MegaDepth + flat image/depth pairs
#   4. cut                   -> CUT unaligned train/test folders with RUOD trainB/testB
#   5. watergan              -> source copy + MegaDepth + air_images/air_depth/water_images
#   6. stable_diffusion_img2img -> source copy only; its ImageNet sample source is uwdf/source
#
# The project currently has five model families, but SyreaNet has two runnable
# routes: enhancement and physical synthesis. Keep both prepared because they
# need different temp layouts.
#
# Smoke:
#   conda activate /media/SSD1/conda_envs/syreanet
#   MODE=smoke GPU=2 bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh
#
# Full:
#   conda activate /media/SSD1/conda_envs/syreanet
#   MODE=full GPU=2 bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh
#
# Only selected methods:
#   METHODS="uwnr stable_diffusion_img2img" MODE=smoke bash scripts/exp_2/synthesis/prepare_synthesis_ssd_inputs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
MODE="${MODE:-smoke}"
SPLITS="${SPLITS:-train val}"
GPU="${GPU:-2}"
METHODS="${METHODS:-uwnr syreanet syreanet_synthesis cut watergan stable_diffusion_img2img}"
SOURCE_ONLY="${SOURCE_ONLY:-0}"

SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-200}"
SMOKE_VAL_LIMIT="${SMOKE_VAL_LIMIT:-50}"
FULL_LIMIT="${FULL_LIMIT:-0}"
COPY_MODE="${COPY_MODE:-copy}"
OVERWRITE="${OVERWRITE:-1}"

UWNR_TEST_SIZE="${UWNR_TEST_SIZE:-256}"
SYREANET_IMG_SIZE="${SYREANET_IMG_SIZE:-512}"
SYREANET_SYN_PREP_SIZE="${SYREANET_SYN_PREP_SIZE:-512}"

CUT_TRAIN_B_LIMIT="${CUT_TRAIN_B_LIMIT:-0}"
CUT_TEST_B_LIMIT="${CUT_TEST_B_LIMIT:-1000}"
WATERGAN_WATER_LIMIT="${WATERGAN_WATER_LIMIT:-0}"

MEGADEPTH_DIR="${MEGADEPTH_DIR:-/home/fcp/xcx/exp_2/syn/MegaDepth}"
MEGADEPTH_CKPT="${MEGADEPTH_CKPT:-${MEGADEPTH_DIR}/checkpoints/best_generalization_net_G.pth}"
UWNR_DIR="${UWNR_DIR:-/home/fcp/xcx/exp_2/syn/UWNR}"
UWNR_CKPT="${UWNR_CKPT:-${UWNR_DIR}/checkpoints/uwnr_pretrained.pk}"
SYREANET_DIR="${SYREANET_DIR:-/home/fcp/xcx/exp_2/syn/SyreaNet}"
SYREANET_CKPT="${SYREANET_CKPT:-${SYREANET_DIR}/checkpoints/pretrained.pth}"
SYREANET_BASE_CONFIG="${SYREANET_BASE_CONFIG:-${SYREANET_DIR}/configs/syreanet_test.yaml}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_prepare}"
mkdir -p "${LOG_DIR}" "${WORK_ROOT}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Error: MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi
if [[ "${COPY_MODE}" != "copy" && "${COPY_MODE}" != "symlink" && "${COPY_MODE}" != "hardlink" ]]; then
  echo "Error: COPY_MODE must be copy, symlink, or hardlink." >&2
  exit 1
fi

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

split_limit() {
  local split="$1"
  if [[ "${MODE}" == "full" ]]; then
    echo "${FULL_LIMIT}"
  elif [[ "${split}" == "val" ]]; then
    echo "${SMOKE_VAL_LIMIT}"
  else
    echo "${SMOKE_TRAIN_LIMIT}"
  fi
}

method_source_name() {
  local method="$1"
  case "${method}" in
    stable_diffusion_img2img)
      echo "uwdf"
      ;;
    syreanet_synthesis)
      echo "syreanet"
      ;;
    *)
      echo "${method}"
      ;;
  esac
}

copy_sampled_source() {
  local method="$1"
  local split="$2"
  local source_name
  source_name="$(method_source_name "${method}")"
  local src="${SOURCE_ROOT}/${source_name}/source/${split}"
  local dst="${WORK_ROOT}/sources/${method}/${split}"
  local limit
  limit="$(split_limit "${split}")"

  check_path "${src}" "${method} sampled source for ${split}"
  mkdir -p "${dst}"

  echo
  echo "[${method}/${split}] Copy sampled source to SSD"
  echo "  src:       ${src}"
  echo "  dst:       ${dst}"
  echo "  limit:     ${limit}"
  echo "  mode:      ${COPY_MODE}"
  echo "  overwrite: ${OVERWRITE}"

  SRC="${src}" DST="${dst}" LIMIT="${limit}" COPY_MODE="${COPY_MODE}" OVERWRITE="${OVERWRITE}" python - <<'PY'
from pathlib import Path
import json
import os
import shutil

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])
limit = int(os.environ["LIMIT"])
copy_mode = os.environ["COPY_MODE"]
overwrite = os.environ.get("OVERWRITE", "1") == "1"
image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print(f"scanning images: {src}", flush=True)
images = []
for path in tqdm(src.rglob("*"), desc=f"scan {src.name}", unit="entry"):
    if path.is_file() and path.suffix.lower() in image_suffixes:
        images.append(path)
images.sort()
total_before_limit = len(images)
if limit > 0:
    images = images[:limit]

dst.mkdir(parents=True, exist_ok=True)
if overwrite:
    for old in tqdm(list(dst.rglob("*")), desc="clear old files", unit="entry"):
        if old.is_file() or old.is_symlink():
            old.unlink()

records = []
written = skipped = failed = 0
for path in tqdm(images, desc="materialize source", unit="image"):
    rel = path.relative_to(src)
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        skipped += 1
        continue
    try:
        if copy_mode == "copy":
            shutil.copy2(path, out)
        elif copy_mode == "hardlink":
            os.link(path, out)
        elif copy_mode == "symlink":
            os.symlink(path, out)
        else:
            raise ValueError(copy_mode)
        written += 1
        records.append({
            "relative": str(rel).replace("\\", "/"),
            "synset": rel.parts[0] if len(rel.parts) > 1 else "unknown",
            "source": str(path),
            "ssd_path": str(out),
        })
    except Exception as exc:
        failed += 1
        records.append({
            "relative": str(rel).replace("\\", "/"),
            "source": str(path),
            "error": repr(exc),
        })

manifest = dst / "source_manifest.jsonl"
with manifest.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

summary = {
    "src": str(src),
    "dst": str(dst),
    "copy_mode": copy_mode,
    "limit": limit,
    "total_before_limit": total_before_limit,
    "selected": len(images),
    "written": written,
    "skipped_existing": skipped,
    "failed": failed,
    "manifest": str(manifest),
}
(dst / "source_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
if failed:
    raise SystemExit(f"failed to materialize {failed} source images")
PY
}

prepare_uwnr() {
  local split="$1"
  local source="${WORK_ROOT}/sources/uwnr/${split}"
  local depth="${WORK_ROOT}/uwnr_ruod_ref/megadepth/${split}"
  local prep="${WORK_ROOT}/uwnr_ruod_ref/prepared/${split}"
  local ref_root="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}"
  local fid_ref="${WORK_ROOT}/uwnr_ruod_ref/ruod_reference_${split}_fid_resized"
  local flat="${WORK_ROOT}/uwnr_ruod_ref/generated_flat/${split}"
  local restore="${SYN_ROOT}/uwnr_ruod_ref/generated/${split}"
  local limit
  limit="$(split_limit "${split}")"

  check_path "${UWNR_DIR}/test.py" "UWNR test.py"
  check_path "${UWNR_CKPT}" "UWNR checkpoint"
  echo
  echo "[uwnr/${split}] Prepare MegaDepth, flat pairs, and RUOD refs on SSD"
  SOURCE_DIR="${source}" \
  DEPTH_DIR="${depth}" \
  PREP_DIR="${prep}" \
  RUOD_REF_ROOT="${ref_root}" \
  FID_REF_DIR="${fid_ref}" \
  FLAT_SAVE_DIR="${flat}" \
  RESTORE_DIR="${restore}" \
  RUOD_REF_SRC="${RUOD_REF_SRC}" \
  UWNR_DIR="${UWNR_DIR}" \
  UWNR_CKPT="${UWNR_CKPT}" \
  MEGADEPTH_DIR="${MEGADEPTH_DIR}" \
  MEGADEPTH_CKPT="${MEGADEPTH_CKPT}" \
  SPLIT="${split}" \
  GPU="${GPU}" \
  LIMIT="${limit}" \
  TEST_SIZE="${UWNR_TEST_SIZE}" \
  RUN_DEPTH=1 RUN_PREPARE=1 RUN_RUOD_REF=1 RUN_UWNR=0 RUN_RESTORE=0 \
  bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh \
    2>&1 | tee "${LOG_DIR}/prepare_uwnr_${split}_${MODE}.log"
}

prepare_syreanet() {
  local split="$1"
  local source="${WORK_ROOT}/sources/syreanet/${split}"
  local prep="${WORK_ROOT}/syreanet/prepared/${split}"
  local flat="${WORK_ROOT}/syreanet/generated_flat/${split}"
  local final="${SYN_ROOT}/syreanet/generated/${split}"
  local limit
  limit="$(split_limit "${split}")"

  check_path "${SYREANET_DIR}/test.py" "SyreaNet test.py"
  check_path "${SYREANET_CKPT}" "SyreaNet checkpoint"
  check_path "${SYREANET_BASE_CONFIG}" "SyreaNet base config"
  echo
  echo "[syreanet/${split}] Prepare resized flat input/config on SSD"
  SOURCE_DIR="${source}" \
  PREP_DIR="${prep}" \
  FLAT_SAVE_DIR="${flat}" \
  FINAL_SAVE_DIR="${final}" \
  SYREANET_DIR="${SYREANET_DIR}" \
  SYREANET_CKPT="${SYREANET_CKPT}" \
  SYREANET_BASE_CONFIG="${SYREANET_BASE_CONFIG}" \
  SPLIT="${split}" \
  GPU="${GPU}" \
  LIMIT="${limit}" \
  IMG_SIZE="${SYREANET_IMG_SIZE}" \
  RUN_PREPARE=1 RUN_SYREANET=0 RUN_RESTORE=0 \
  bash scripts/exp_2/synthesis/run_syreanet_generate.sh \
    2>&1 | tee "${LOG_DIR}/prepare_syreanet_${split}_${MODE}.log"
}

prepare_syreanet_synthesis() {
  local split="$1"
  local source="${WORK_ROOT}/sources/syreanet_synthesis/${split}"
  local depth="${WORK_ROOT}/syreanet_synthesis/depth/${split}"
  local prep="${WORK_ROOT}/syreanet_synthesis/prepared/${split}"
  local flat="${WORK_ROOT}/syreanet_synthesis/generated_flat/${split}"
  local restore="${SYN_ROOT}/syreanet_synthesis/generated/${split}"
  local limit
  limit="$(split_limit "${split}")"

  check_path "${SYREANET_DIR}/synthesize/synthesize.py" "SyreaNet synthesize.py"
  echo
  echo "[syreanet_synthesis/${split}] Prepare MegaDepth and flat image/depth pairs on SSD"
  SOURCE_DIR="${source}" \
  DEPTH_DIR="${depth}" \
  PREP_DIR="${prep}" \
  FLAT_SAVE_DIR="${flat}" \
  RESTORE_DIR="${restore}" \
  SYREANET_DIR="${SYREANET_DIR}" \
  MEGADEPTH_DIR="${MEGADEPTH_DIR}" \
  MEGADEPTH_CKPT="${MEGADEPTH_CKPT}" \
  SPLIT="${split}" \
  GPU="${GPU}" \
  LIMIT="${limit}" \
  PREP_SIZE="${SYREANET_SYN_PREP_SIZE}" \
  RUN_DEPTH=1 RUN_PREPARE=1 RUN_SYREANET=0 RUN_RESTORE=0 \
  bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh \
    2>&1 | tee "${LOG_DIR}/prepare_syreanet_synthesis_${split}_${MODE}.log"
}

prepare_cut() {
  local train_limit val_limit data_name data_root train_a test_a
  train_limit="$(split_limit train)"
  val_limit="$(split_limit val)"
  data_name="imagenet_ruod_cut_${MODE}_ssd"
  data_root="${WORK_ROOT}/cut/datasets/${data_name}"
  train_a="${WORK_ROOT}/sources/cut/train"
  test_a="${WORK_ROOT}/sources/cut/val"

  echo
  echo "[cut] Prepare CUT unaligned train/test folders on SSD"
  DATA_NAME="${data_name}" \
  DATA_ROOT="${data_root}" \
  TRAIN_A_SOURCE="${train_a}" \
  TEST_A_SOURCE="${test_a}" \
  TRAIN_B_SOURCE="${RUOD_REF_SRC}" \
  TEST_B_SOURCE="${RUOD_REF_SRC}" \
  TRAIN_A_LIMIT="${train_limit}" \
  TEST_A_LIMIT="${val_limit}" \
  TRAIN_B_LIMIT="${CUT_TRAIN_B_LIMIT}" \
  TEST_B_LIMIT="${CUT_TEST_B_LIMIT}" \
  LINK_MODE="${COPY_MODE}" \
  OVERWRITE="${OVERWRITE}" \
  bash scripts/exp_2/synthesis/run_cut_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/prepare_cut_${MODE}.log"
}

prepare_watergan() {
  local split="$1"
  local source="${WORK_ROOT}/sources/watergan/${split}"
  local depth="${WORK_ROOT}/watergan/depth/${split}"
  local data_name="imagenet_ruod_watergan_${split}_${MODE}_ssd"
  local data_root="${WORK_ROOT}/watergan/datasets/${data_name}"
  local limit
  limit="$(split_limit "${split}")"

  echo
  echo "[watergan/${split}] Prepare WaterGAN air/depth/water folders on SSD"
  SYN_ROOT="${SYN_ROOT}" \
  WORK_ROOT="${WORK_ROOT}" \
  SOURCE_DIR="${source}" \
  WATER_SOURCE="${RUOD_REF_SRC}" \
  DATA_NAME="${data_name}" \
  DEPTH_DIR="${depth}" \
  DATA_ROOT="${data_root}" \
  MEGADEPTH_DIR="${MEGADEPTH_DIR}" \
  MEGADEPTH_CKPT="${MEGADEPTH_CKPT}" \
  SPLIT="${split}" \
  GPU="${GPU}" \
  AIR_LIMIT="${limit}" \
  WATER_LIMIT="${WATERGAN_WATER_LIMIT}" \
  OVERWRITE="${OVERWRITE}" \
  RUN_DEPTH=1 \
  bash scripts/exp_2/synthesis/run_watergan_prepare_dataset.sh \
    2>&1 | tee "${LOG_DIR}/prepare_watergan_${split}_${MODE}.log"
}

prepare_stable_diffusion() {
  local split="$1"
  local source="${WORK_ROOT}/sources/stable_diffusion_img2img/${split}"
  local generated="${WORK_ROOT}/stable_diffusion_img2img/generated/${split}"
  mkdir -p "${generated}"
  echo
  echo "[stable_diffusion_img2img/${split}] SSD source prepared"
  echo "  source:    ${source}"
  echo "  generated: ${generated}"
  echo "  use with:"
  echo "    SOURCE_DIR=${source} OUT_DIR=${generated} SPLIT=${split} bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh"
}

echo "========================================="
echo "Prepare synthesis SSD inputs"
echo "========================================="
echo "SYN_ROOT:       ${SYN_ROOT}"
echo "SOURCE_ROOT:    ${SOURCE_ROOT}"
echo "WORK_ROOT:      ${WORK_ROOT}"
echo "RUOD_REF_SRC:   ${RUOD_REF_SRC}"
echo "MODE:           ${MODE}"
echo "SPLITS:         ${SPLITS}"
echo "METHODS:        ${METHODS}"
echo "SOURCE_ONLY:    ${SOURCE_ONLY}"
echo "GPU:            ${GPU}"
echo "COPY_MODE:      ${COPY_MODE}"
echo "OVERWRITE:      ${OVERWRITE}"
echo "SMOKE_LIMITS:   train=${SMOKE_TRAIN_LIMIT}, val=${SMOKE_VAL_LIMIT}"
echo "FULL_LIMIT:     ${FULL_LIMIT}"
echo "LOG_DIR:        ${LOG_DIR}"
echo "========================================="

check_path "${SYN_ROOT}" "synthetic ImageNet root"
check_path "${RUOD_REF_SRC}" "RUOD reference source"
check_path "${MEGADEPTH_DIR}" "MegaDepth repository"
check_path "${MEGADEPTH_CKPT}" "MegaDepth checkpoint"

for method in ${METHODS}; do
  case "${method}" in
    cut)
      # CUT needs both train and val/test sources before one dataset prepare call.
      for split in train val; do
        copy_sampled_source "${method}" "${split}"
      done
      if [[ "${SOURCE_ONLY}" == "1" ]]; then
        continue
      fi
      prepare_cut
      ;;
    uwnr)
      for split in ${SPLITS}; do
        copy_sampled_source "${method}" "${split}"
        if [[ "${SOURCE_ONLY}" == "1" ]]; then
          continue
        fi
        prepare_uwnr "${split}"
      done
      ;;
    syreanet)
      for split in ${SPLITS}; do
        copy_sampled_source "${method}" "${split}"
        if [[ "${SOURCE_ONLY}" == "1" ]]; then
          continue
        fi
        prepare_syreanet "${split}"
      done
      ;;
    syreanet_synthesis)
      for split in ${SPLITS}; do
        copy_sampled_source "${method}" "${split}"
        if [[ "${SOURCE_ONLY}" == "1" ]]; then
          continue
        fi
        prepare_syreanet_synthesis "${split}"
      done
      ;;
    watergan)
      for split in ${SPLITS}; do
        copy_sampled_source "${method}" "${split}"
        if [[ "${SOURCE_ONLY}" == "1" ]]; then
          continue
        fi
        prepare_watergan "${split}"
      done
      ;;
    stable_diffusion_img2img)
      for split in ${SPLITS}; do
        copy_sampled_source "${method}" "${split}"
        if [[ "${SOURCE_ONLY}" == "1" ]]; then
          continue
        fi
        prepare_stable_diffusion "${split}"
      done
      ;;
    *)
      echo "Error: unknown method: ${method}" >&2
      exit 1
      ;;
  esac
done

echo
echo "========================================="
echo "SSD preparation complete"
echo "========================================="
echo "Prepared sources: ${WORK_ROOT}/sources"
echo "Prepared temps:   ${WORK_ROOT}"
echo "Logs:             ${LOG_DIR}"
