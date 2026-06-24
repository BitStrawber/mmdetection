#!/usr/bin/env bash
set -euo pipefail

# Smoke-test ImageNet -> underwater synthesis with official UWNR weights and
# RUOD real-underwater reference images. This script reuses the existing
# synthetic_imagenet/uwnr/source sampling result and only processes LIMIT images.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
RUOD_REF_SRC="${RUOD_REF_SRC:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
UWNR_DIR="${UWNR_DIR:-/home/fcp/xcx/exp_2/syn/UWNR}"
UWNR_CKPT="${UWNR_CKPT:-${UWNR_DIR}/checkpoints/uwnr_pretrained.pk}"
MEGADEPTH_DIR="${MEGADEPTH_DIR:-/home/fcp/xcx/exp_2/syn/MegaDepth}"
MEGADEPTH_CKPT="${MEGADEPTH_CKPT:-${MEGADEPTH_DIR}/checkpoints/best_generalization_net_G.pth}"

GPU="${GPU:-2}"
LIMIT="${LIMIT:-200}"
TEST_SIZE="${TEST_SIZE:-256}"
N_CPU="${N_CPU:-4}"

SOURCE_DIR="${SOURCE_DIR:-${SYN_ROOT}/uwnr/source/train}"
DEPTH_DIR="${DEPTH_DIR:-${SYN_ROOT}/uwnr_ruod_ref/megadepth_smoke/train}"
RUOD_REF_ROOT="${RUOD_REF_ROOT:-${SYN_ROOT}/uwnr_ruod_ref/ruod_reference_smoke}"
# Official UWNR dataloader expects underwater images under
# ${underwater_path}/qingxi, so --underwater_path points to RUOD_REF_ROOT.
RUOD_REF_DIR="${RUOD_REF_DIR:-}"
if [[ -z "${RUOD_REF_DIR}" || "${RUOD_REF_DIR}" == "${RUOD_REF_ROOT}/images" ]]; then
  RUOD_REF_DIR="${RUOD_REF_ROOT}/qingxi"
fi
PREP_DIR="${PREP_DIR:-${SYN_ROOT}/uwnr_ruod_ref/prepared_smoke/train}"
SAVE_DIR="${SAVE_DIR:-${SYN_ROOT}/uwnr_ruod_ref/generated_smoke_flat/train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PY_COMPAT_DIR="${PY_COMPAT_DIR:-${SYN_ROOT}/uwnr_ruod_ref/python_compat}"
RUOD_REF_COUNT="${RUOD_REF_COUNT:-${LIMIT}}"

RUN_RUOD_REF="${RUN_RUOD_REF:-1}"
RUN_DEPTH="${RUN_DEPTH:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_UWNR="${RUN_UWNR:-1}"

mkdir -p "${LOG_DIR}"

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: ${label} not found: ${path}" >&2
    exit 1
  fi
}

echo "========================================="
echo "UWNR + RUOD reference smoke pipeline"
echo "========================================="
echo "SYN_ROOT:       ${SYN_ROOT}"
echo "SOURCE_DIR:     ${SOURCE_DIR}"
echo "RUOD_REF_SRC:   ${RUOD_REF_SRC}"
echo "RUOD_REF_ROOT:  ${RUOD_REF_ROOT}"
echo "RUOD_REF_DIR:   ${RUOD_REF_DIR}"
echo "DEPTH_DIR:      ${DEPTH_DIR}"
echo "PREP_DIR:       ${PREP_DIR}"
echo "SAVE_DIR:       ${SAVE_DIR}"
echo "PY_COMPAT_DIR:  ${PY_COMPAT_DIR}"
echo "UWNR_DIR:       ${UWNR_DIR}"
echo "UWNR_CKPT:      ${UWNR_CKPT}"
echo "MEGADEPTH_DIR:  ${MEGADEPTH_DIR}"
echo "MEGADEPTH_CKPT: ${MEGADEPTH_CKPT}"
echo "GPU:            ${GPU}"
echo "LIMIT:          ${LIMIT}"
echo "TEST_SIZE:      ${TEST_SIZE}"
echo "N_CPU:          ${N_CPU}"
echo "RUOD_REF_COUNT: ${RUOD_REF_COUNT}"
echo "========================================="

check_path "${SOURCE_DIR}" "ImageNet UWNR source directory"
check_path "${RUOD_REF_SRC}" "RUOD reference source directory"
check_path "${UWNR_DIR}/test.py" "UWNR test.py"
check_path "${UWNR_CKPT}" "UWNR pretrained checkpoint"
check_path "${MEGADEPTH_DIR}" "MegaDepth repository"
check_path "${MEGADEPTH_CKPT}" "MegaDepth checkpoint"

if [[ "${RUN_RUOD_REF}" == "1" ]]; then
  echo
  echo "Step 1/4: Build flat RUOD reference symlink directory"
  mkdir -p "${RUOD_REF_DIR}"
  RUOD_REF_SRC="${RUOD_REF_SRC}" RUOD_REF_DIR="${RUOD_REF_DIR}" RUOD_REF_COUNT="${RUOD_REF_COUNT}" python - <<'PY'
from pathlib import Path
import os

src = Path(os.environ["RUOD_REF_SRC"])
dst = Path(os.environ["RUOD_REF_DIR"])
target_count = int(os.environ["RUOD_REF_COUNT"])
dst.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if not files:
    raise SystemExit(f"No reference images found under {src}")
if target_count <= 0:
    target_count = len(files)

# Keep the official UWNR reference loader length aligned with the smoke clean
# set length. Stale links from earlier runs would otherwise make test.py loop
# over too many reference images and index beyond the clean/depth pairs.
for old in dst.iterdir():
    if old.is_file() or old.is_symlink():
        old.unlink()

created = 0
for i in range(target_count):
    path = files[i % len(files)]
    target = dst / f"{i:08d}{path.suffix.lower()}"
    os.symlink(path, target)
    created += 1

total = sum(1 for p in dst.iterdir() if p.is_file() or p.is_symlink())
print(f"RUOD source images: {len(files)}")
print(f"requested reference images: {target_count}")
print(f"new symlinks: {created}")
print(f"reference images ready: {total}")
PY
else
  echo
  echo "Step 1/4: Skip RUOD reference build"
fi

if [[ "${RUN_DEPTH}" == "1" ]]; then
  echo
  echo "Step 2/4: Generate MegaDepth maps"
  python tools/generate_megadepth_maps.py \
    --image-dir "${SOURCE_DIR}" \
    --out-dir "${DEPTH_DIR}" \
    --megadepth-dir "${MEGADEPTH_DIR}" \
    --checkpoint "${MEGADEPTH_CKPT}" \
    --device "cuda:${GPU}" \
    --limit "${LIMIT}" \
    2>&1 | tee "${LOG_DIR}/uwnr_ruod_ref_megadepth_smoke_train.log"
else
  echo
  echo "Step 2/4: Skip MegaDepth generation"
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo
  echo "Step 3/4: Prepare flat clean/depth pairs for official UWNR test.py"
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

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if limit > 0:
    images = images[:limit]

records = []
missing_depth = 0
for index, image_path in enumerate(images):
    rel = image_path.relative_to(source_root)
    depth_path = depth_root / rel.with_suffix(".png")
    if not depth_path.exists():
        missing_depth += 1
        print(f"missing depth: {rel}")
        continue

    stem = f"{len(records):08d}"
    clean_link = clean_out / f"{stem}{image_path.suffix.lower()}"
    depth_link = depth_out / f"{stem}.png"

    if not clean_link.exists() and not clean_link.is_symlink():
        os.symlink(image_path, clean_link)
    if not depth_link.exists() and not depth_link.is_symlink():
        os.symlink(depth_path, depth_link)

    records.append({
        "index": len(records),
        "source_index": index,
        "relative": str(rel).replace("\\", "/"),
        "synset": rel.parts[0],
        "original_name": rel.name,
        "clean": str(clean_link),
        "depth": str(depth_link),
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
  echo "Step 3/4: Skip flat pair preparation"
fi

if [[ "${RUN_UWNR}" == "1" ]]; then
  echo
  echo "Step 4/4: Run official UWNR test.py with RUOD reference"
  mkdir -p "${SAVE_DIR}"
  mkdir -p "${PY_COMPAT_DIR}"
  cat > "${PY_COMPAT_DIR}/sitecustomize.py" <<'PY'
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
PY
  (
    cd "${UWNR_DIR}"
    PYTHONPATH="${PY_COMPAT_DIR}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
      --cuda \
      --test_size "${TEST_SIZE}" \
      --n_cpu "${N_CPU}" \
      --save_path "${SAVE_DIR}" \
      --clean_img_path "${PREP_DIR}/clean" \
      --depth_img_path "${PREP_DIR}/depth" \
      --underwater_path "${RUOD_REF_ROOT}" \
      --fid_gt_path "${RUOD_REF_DIR}" \
      --model_path "${UWNR_CKPT}"
  ) 2>&1 | tee "${LOG_DIR}/uwnr_ruod_ref_smoke_test.log"
else
  echo
  echo "Step 4/4: Skip UWNR generation"
fi

echo
echo "========================================="
echo "Smoke pipeline finished"
echo "========================================="
echo "Prepared pairs:"
find "${PREP_DIR}/clean" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l
echo "Generated images:"
find "${SAVE_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l
echo "Output: ${SAVE_DIR}"
