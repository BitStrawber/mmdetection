#!/usr/bin/env bash
set -euo pipefail

# Full ImageNet -> underwater synthesis with official UWNR inference logic and
# RUOD real-underwater reference images.
#
# This script intentionally keeps generation and evaluation separate:
# - It runs official UWNR test.py, with only compatibility patches required for
#   current Python/NumPy and optional FID skipping/resized FID references.
# - It restores UWNR's flat output order back to ImageNet synset directories
#   using pair_manifest.jsonl.
#
# Typical use:
#   SPLIT=train GPU=2 bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh
#   SPLIT=val   GPU=2 bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh
#   LIMIT=200   GPU=2 bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
UWNR_DIR="${UWNR_DIR:-/home/fcp/xcx/exp_2/syn/UWNR}"
UWNR_CKPT="${UWNR_CKPT:-${UWNR_DIR}/checkpoints/uwnr_pretrained.pk}"
MEGADEPTH_DIR="${MEGADEPTH_DIR:-/home/fcp/xcx/exp_2/syn/MegaDepth}"
MEGADEPTH_CKPT="${MEGADEPTH_CKPT:-${MEGADEPTH_DIR}/checkpoints/best_generalization_net_G.pth}"

SPLIT="${SPLIT:-train}"
GPU="${GPU:-2}"
LIMIT="${LIMIT:-0}"
TEST_SIZE="${TEST_SIZE:-256}"
N_CPU="${N_CPU:-4}"
FID_SIZE="${FID_SIZE:-${TEST_SIZE}}"

SOURCE_DIR="${SOURCE_DIR:-${SYN_ROOT}/uwnr/source/${SPLIT}}"
DEPTH_DIR="${DEPTH_DIR:-${SYN_ROOT}/uwnr_ruod_ref/megadepth/${SPLIT}}"
PREP_DIR="${PREP_DIR:-${SYN_ROOT}/uwnr_ruod_ref/prepared/${SPLIT}}"
RUOD_REF_ROOT="${RUOD_REF_ROOT:-${SYN_ROOT}/uwnr_ruod_ref/ruod_reference_${SPLIT}}"
RUOD_REF_DIR="${RUOD_REF_DIR:-}"
if [[ -z "${RUOD_REF_DIR}" || "${RUOD_REF_DIR}" == "${RUOD_REF_ROOT}/images" ]]; then
  RUOD_REF_DIR="${RUOD_REF_ROOT}/qingxi"
fi
FID_REF_DIR="${FID_REF_DIR:-${SYN_ROOT}/uwnr_ruod_ref/ruod_reference_${SPLIT}_fid_resized}"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${SYN_ROOT}/uwnr_ruod_ref/generated_flat/${SPLIT}}"
RESTORE_DIR="${RESTORE_DIR:-${SYN_ROOT}/uwnr_ruod_ref/generated/${SPLIT}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PY_COMPAT_DIR="${PY_COMPAT_DIR:-${SYN_ROOT}/uwnr_ruod_ref/python_compat}"

RUN_DEPTH="${RUN_DEPTH:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_RUOD_REF="${RUN_RUOD_REF:-1}"
RUN_UWNR="${RUN_UWNR:-1}"
RUN_RESTORE="${RUN_RESTORE:-1}"
SKIP_FID="${SKIP_FID:-1}"
CLEAR_FLAT_OUTPUT="${CLEAR_FLAT_OUTPUT:-1}"
CLEAR_RESTORE_OUTPUT="${CLEAR_RESTORE_OUTPUT:-0}"

mkdir -p "${LOG_DIR}"

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

count_manifest() {
  local manifest="$1"
  if [[ ! -f "${manifest}" ]]; then
    echo "0"
  else
    wc -l < "${manifest}" | tr -d ' '
  fi
}

echo "========================================="
echo "UWNR + RUOD reference generation pipeline"
echo "========================================="
echo "SPLIT:                ${SPLIT}"
echo "SYN_ROOT:             ${SYN_ROOT}"
echo "SOURCE_DIR:           ${SOURCE_DIR}"
echo "DEPTH_DIR:            ${DEPTH_DIR}"
echo "PREP_DIR:             ${PREP_DIR}"
echo "RUOD_REF_SRC:         ${RUOD_REF_SRC}"
echo "RUOD_REF_ROOT:        ${RUOD_REF_ROOT}"
echo "RUOD_REF_DIR:         ${RUOD_REF_DIR}"
echo "FID_REF_DIR:          ${FID_REF_DIR}"
echo "FLAT_SAVE_DIR:        ${FLAT_SAVE_DIR}"
echo "RESTORE_DIR:          ${RESTORE_DIR}"
echo "UWNR_DIR:             ${UWNR_DIR}"
echo "UWNR_CKPT:            ${UWNR_CKPT}"
echo "MEGADEPTH_DIR:        ${MEGADEPTH_DIR}"
echo "MEGADEPTH_CKPT:       ${MEGADEPTH_CKPT}"
echo "GPU:                  ${GPU}"
echo "LIMIT:                ${LIMIT}"
echo "TEST_SIZE:            ${TEST_SIZE}"
echo "N_CPU:                ${N_CPU}"
echo "FID_SIZE:             ${FID_SIZE}"
echo "SKIP_FID:             ${SKIP_FID}"
echo "CLEAR_FLAT_OUTPUT:    ${CLEAR_FLAT_OUTPUT}"
echo "CLEAR_RESTORE_OUTPUT: ${CLEAR_RESTORE_OUTPUT}"
echo "========================================="

check_path "${SOURCE_DIR}" "ImageNet UWNR source directory"
check_path "${RUOD_REF_SRC}" "RUOD reference source directory"
check_path "${UWNR_DIR}/test.py" "UWNR test.py"
check_path "${UWNR_CKPT}" "UWNR pretrained checkpoint"
check_path "${MEGADEPTH_DIR}" "MegaDepth repository"
check_path "${MEGADEPTH_CKPT}" "MegaDepth checkpoint"

if [[ "${RUN_DEPTH}" == "1" ]]; then
  echo
  echo "Step 1/5: Generate MegaDepth maps"
  python tools/generate_megadepth_maps.py \
    --image-dir "${SOURCE_DIR}" \
    --out-dir "${DEPTH_DIR}" \
    --megadepth-dir "${MEGADEPTH_DIR}" \
    --checkpoint "${MEGADEPTH_CKPT}" \
    --device "cuda:${GPU}" \
    --limit "${LIMIT}" \
    2>&1 | tee "${LOG_DIR}/uwnr_ruod_ref_megadepth_${SPLIT}.log"
else
  echo
  echo "Step 1/5: Skip MegaDepth generation"
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo
  echo "Step 2/5: Prepare flat clean/depth pairs"
  mkdir -p "${PREP_DIR}/clean" "${PREP_DIR}/depth"
  SOURCE_DIR="${SOURCE_DIR}" DEPTH_DIR="${DEPTH_DIR}" PREP_DIR="${PREP_DIR}" LIMIT="${LIMIT}" python - <<'PY'
from pathlib import Path
import json
import os

source_root = Path(os.environ["SOURCE_DIR"])
depth_root = Path(os.environ["DEPTH_DIR"])
out = Path(os.environ["PREP_DIR"])
limit = int(os.environ["LIMIT"])

clean_out = out / "clean"
depth_out = out / "depth"
manifest = out / "pair_manifest.jsonl"

clean_out.mkdir(parents=True, exist_ok=True)
depth_out.mkdir(parents=True, exist_ok=True)
for directory in (clean_out, depth_out):
    for old in directory.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if limit > 0:
    images = images[:limit]

records = []
missing_depth = 0
for source_index, image_path in enumerate(images):
    rel = image_path.relative_to(source_root)
    depth_path = depth_root / rel.with_suffix(".png")
    if not depth_path.exists():
        missing_depth += 1
        print(f"missing depth: {rel}")
        continue

    index = len(records)
    stem = f"{index:08d}"
    clean_link = clean_out / f"{stem}{image_path.suffix.lower()}"
    depth_link = depth_out / f"{stem}.png"
    os.symlink(image_path, clean_link)
    os.symlink(depth_path, depth_link)

    records.append({
        "index": index,
        "source_index": source_index,
        "relative": str(rel).replace("\\", "/"),
        "synset": rel.parts[0],
        "original_name": rel.name,
        "source": str(image_path),
        "depth": str(depth_path),
        "clean_link": str(clean_link),
        "depth_link": str(depth_link),
    })

with manifest.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"candidate images: {len(images)}")
print(f"prepared pairs: {len(records)}")
print(f"missing depth: {missing_depth}")
print(f"manifest: {manifest}")
if not records:
    raise SystemExit("No clean/depth pairs prepared.")
PY
else
  echo
  echo "Step 2/5: Skip flat pair preparation"
fi

PAIR_MANIFEST="${PREP_DIR}/pair_manifest.jsonl"
PAIR_COUNT="$(count_manifest "${PAIR_MANIFEST}")"
if [[ "${PAIR_COUNT}" == "0" ]]; then
  echo "Error: no prepared pairs found in ${PAIR_MANIFEST}" >&2
  exit 1
fi
echo "Prepared pair count: ${PAIR_COUNT}"

if [[ "${RUN_RUOD_REF}" == "1" ]]; then
  echo
  echo "Step 3/5: Build RUOD reference images aligned to prepared pair count"
  mkdir -p "${RUOD_REF_DIR}"
  RUOD_REF_SRC="${RUOD_REF_SRC}" RUOD_REF_DIR="${RUOD_REF_DIR}" PAIR_COUNT="${PAIR_COUNT}" python - <<'PY'
from pathlib import Path
import os

src = Path(os.environ["RUOD_REF_SRC"])
dst = Path(os.environ["RUOD_REF_DIR"])
count = int(os.environ["PAIR_COUNT"])
dst.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if not files:
    raise SystemExit(f"No RUOD reference images found under {src}")

for old in dst.iterdir():
    if old.is_file() or old.is_symlink():
        old.unlink()

for i in range(count):
    source = files[i % len(files)]
    os.symlink(source, dst / f"{i:08d}{source.suffix.lower()}")

print(f"RUOD source images: {len(files)}")
print(f"reference images ready: {count}")
print(f"reference dir: {dst}")
PY

  if [[ "${SKIP_FID}" != "1" ]]; then
    echo
    echo "Step 3b/5: Build resized RUOD reference images for FID"
    mkdir -p "${FID_REF_DIR}"
    RUOD_REF_DIR="${RUOD_REF_DIR}" FID_REF_DIR="${FID_REF_DIR}" FID_SIZE="${FID_SIZE}" python - <<'PY'
from pathlib import Path
from PIL import Image
import os

src = Path(os.environ["RUOD_REF_DIR"])
dst = Path(os.environ["FID_REF_DIR"])
size = int(os.environ["FID_SIZE"])
dst.mkdir(parents=True, exist_ok=True)

for old in dst.iterdir():
    if old.is_file() or old.is_symlink():
        old.unlink()

images = sorted(p for p in src.iterdir()
                if (p.is_file() or p.is_symlink()) and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
for i, path in enumerate(images):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        image.save(dst / f"{i:08d}.png")

print(f"FID resized references: {len(images)}")
print(f"FID reference dir: {dst}")
PY
  fi
else
  echo
  echo "Step 3/5: Skip RUOD reference build"
fi

if [[ "${RUN_UWNR}" == "1" ]]; then
  echo
  echo "Step 4/5: Run official UWNR test.py with RUOD reference"
  mkdir -p "${FLAT_SAVE_DIR}" "${PY_COMPAT_DIR}"
  if [[ "${CLEAR_FLAT_OUTPUT}" == "1" ]]; then
    FLAT_SAVE_DIR="${FLAT_SAVE_DIR}" python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["FLAT_SAVE_DIR"])
root.mkdir(parents=True, exist_ok=True)
for path in root.rglob("*"):
    if path.is_file() or path.is_symlink():
        path.unlink()
PY
  fi

  cat > "${PY_COMPAT_DIR}/sitecustomize.py" <<'PY'
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
PY

  UWNR_TEST_SCRIPT="${UWNR_DIR}/test.py"
  FID_PATH="${FID_REF_DIR}"
  if [[ "${SKIP_FID}" == "1" ]]; then
    UWNR_TEST_SCRIPT="${PY_COMPAT_DIR}/test_skip_fid.py"
    UWNR_DIR="${UWNR_DIR}" UWNR_TEST_SCRIPT="${UWNR_TEST_SCRIPT}" python - <<'PY'
from pathlib import Path
import os

src = Path(os.environ["UWNR_DIR"]) / "test.py"
dst = Path(os.environ["UWNR_TEST_SCRIPT"])
text = src.read_text(encoding="utf-8")
text = text.replace(
    "    fid = calculate_fid_given_paths([opt.save_path,opt.fid_gt_path],50,'cuda:0',2048,1)\n",
    "    fid = float('nan')\n"
    "    print('Skipping FID during generation; run resized FID evaluation separately if needed.')\n",
)
dst.write_text(text, encoding="utf-8")
print(f"Using FID-skipping UWNR test copy: {dst}")
PY
  fi

  (
    cd "${UWNR_DIR}"
    PYTHONPATH="${PY_COMPAT_DIR}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" python "${UWNR_TEST_SCRIPT}" \
      --cuda \
      --test_size "${TEST_SIZE}" \
      --n_cpu "${N_CPU}" \
      --save_path "${FLAT_SAVE_DIR}" \
      --clean_img_path "${PREP_DIR}/clean" \
      --depth_img_path "${PREP_DIR}/depth" \
      --underwater_path "${RUOD_REF_ROOT}" \
      --fid_gt_path "${FID_PATH}" \
      --model_path "${UWNR_CKPT}"
  ) 2>&1 | tee "${LOG_DIR}/uwnr_ruod_ref_generate_${SPLIT}.log"
else
  echo
  echo "Step 4/5: Skip UWNR generation"
fi

if [[ "${RUN_RESTORE}" == "1" ]]; then
  echo
  echo "Step 5/5: Restore flat UWNR outputs to ImageNet synset directories"
  PAIR_MANIFEST="${PAIR_MANIFEST}" FLAT_SAVE_DIR="${FLAT_SAVE_DIR}" RESTORE_DIR="${RESTORE_DIR}" CLEAR_RESTORE_OUTPUT="${CLEAR_RESTORE_OUTPUT}" python - <<'PY'
from pathlib import Path
import json
import os
import shutil

manifest = Path(os.environ["PAIR_MANIFEST"])
flat_root = Path(os.environ["FLAT_SAVE_DIR"])
restore_root = Path(os.environ["RESTORE_DIR"])
clear_restore = os.environ.get("CLEAR_RESTORE_OUTPUT", "0") == "1"

records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if not records:
    raise SystemExit(f"No records in manifest: {manifest}")

if clear_restore and restore_root.exists():
    for path in restore_root.rglob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
restore_root.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
outputs = sorted(p for p in flat_root.rglob("*")
                 if p.is_file() and p.suffix.lower() in exts)
if len(outputs) < len(records):
    raise SystemExit(
        f"Not enough generated images. outputs={len(outputs)}, records={len(records)}, flat_root={flat_root}")
if len(outputs) > len(records):
    print(f"Warning: generated outputs ({len(outputs)}) exceed records ({len(records)}); using sorted first records only.")
    outputs = outputs[:len(records)]

written = skipped = 0
for record, output_path in zip(records, outputs):
    rel = Path(record["relative"])
    dst = (restore_root / rel).with_suffix(output_path.suffix.lower())
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        skipped += 1
        continue
    shutil.copy2(output_path, dst)
    written += 1

summary = {
    "manifest": str(manifest),
    "flat_root": str(flat_root),
    "restore_root": str(restore_root),
    "records": len(records),
    "outputs_used": len(outputs),
    "written": written,
    "skipped_existing": skipped,
}
(restore_root / "restore_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
else
  echo
  echo "Step 5/5: Skip output restoration"
fi

echo
echo "========================================="
echo "UWNR generation pipeline finished"
echo "========================================="
echo "Prepared pairs: ${PAIR_COUNT}"
echo "Flat generated images:"
find "${FLAT_SAVE_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l
echo "Restored generated images:"
find "${RESTORE_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l
echo "Flat output: ${FLAT_SAVE_DIR}"
echo "Restored output: ${RESTORE_DIR}"
