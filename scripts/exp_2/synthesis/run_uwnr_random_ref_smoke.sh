#!/usr/bin/env bash
set -euo pipefail

# Random UWNR smoke test.
#
# It samples N clean ImageNet images across different synsets and N underwater
# reference images, then runs the existing UWNR generation script with matching
# clean/reference counts. This avoids the official UWNR test loop trying to run
# 5000 reference images against only a few clean/depth pairs.
#
# Usage:
#   NUM=20 GPU=2 bash scripts/exp_2/synthesis/run_uwnr_random_ref_smoke.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU="${GPU:-2}"
TEST_SIZE="${TEST_SIZE:-256}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwnr/source/train}"
REF_ROOT="${REF_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwnr_lnrud_ref/random${NUM}}"

RANDOM_SOURCE_DIR="${RANDOM_SOURCE_DIR:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwnr/source_random${NUM}/train}"
RANDOM_REF_ROOT="${RANDOM_REF_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref_random${NUM}}"
RANDOM_REF_QINGXI="${RANDOM_REF_ROOT}/qingxi"

DEPTH_DIR="${DEPTH_DIR:-${WORK_ROOT}/megadepth/train}"
PREP_DIR="${PREP_DIR:-${WORK_ROOT}/prepared/train}"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${WORK_ROOT}/generated_flat/train}"
RESTORE_DIR="${RESTORE_DIR:-${WORK_ROOT}/generated/train}"

echo "========================================="
echo "UWNR random clean/reference smoke"
echo "========================================="
echo "NUM:                ${NUM}"
echo "SEED:               ${SEED}"
echo "GPU:                ${GPU}"
echo "TEST_SIZE:          ${TEST_SIZE}"
echo "SOURCE_ROOT:        ${SOURCE_ROOT}"
echo "REF_ROOT:           ${REF_ROOT}"
echo "RANDOM_SOURCE_DIR:  ${RANDOM_SOURCE_DIR}"
echo "RANDOM_REF_ROOT:    ${RANDOM_REF_ROOT}"
echo "WORK_ROOT:          ${WORK_ROOT}"
echo "========================================="

python - <<PY
from pathlib import Path
import json
import os
import random
import shutil

num = int("${NUM}")
seed = int("${SEED}")
source_root = Path("${SOURCE_ROOT}")
ref_root = Path("${REF_ROOT}")
source_out = Path("${RANDOM_SOURCE_DIR}")
ref_out = Path("${RANDOM_REF_QINGXI}")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clear_tree(path):
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

clear_tree(source_out)
source_out.mkdir(parents=True, exist_ok=True)
ref_out.mkdir(parents=True, exist_ok=True)
for p in list(ref_out.iterdir()):
    if p.is_file() or p.is_symlink():
        p.unlink()

by_cls = {}
for p in source_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        rel = p.relative_to(source_root)
        if len(rel.parts) >= 2:
            by_cls.setdefault(rel.parts[0], []).append(p)

classes = sorted(by_cls)
if len(classes) < num:
    raise RuntimeError(f"Not enough source classes: {len(classes)} < {num}")

rng = random.Random(seed)
picked_classes = rng.sample(classes, num)
picked_sources = []
for cls in picked_classes:
    src = rng.choice(sorted(by_cls[cls]))
    rel = src.relative_to(source_root)
    dst = source_out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    picked_sources.append({"class": cls, "source": str(src), "selected": str(dst)})

refs = sorted(p for p in ref_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if len(refs) < num:
    raise RuntimeError(f"Not enough reference images: {len(refs)} < {num}")

picked_refs = rng.sample(refs, num)
for i, src in enumerate(picked_refs, 1):
    dst = ref_out / f"{i:08d}{src.suffix.lower()}"
    os.symlink(src, dst)

manifest = Path("${WORK_ROOT}") / "random_selection_manifest.json"
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({
    "num": num,
    "seed": seed,
    "source_root": str(source_root),
    "reference_root": str(ref_root),
    "random_source_dir": str(source_out),
    "random_reference_root": str(ref_out.parent),
    "sources": picked_sources,
    "references": [{"source": str(p), "selected": str(ref_out / f"{i:08d}{p.suffix.lower()}")}
                   for i, p in enumerate(picked_refs, 1)],
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"picked source images: {len(picked_sources)}")
print(f"picked source classes: {len(set(x['class'] for x in picked_sources))}")
print(f"picked references: {len(picked_refs)}")
print(f"manifest: {manifest}")
PY

SOURCE_DIR="${RANDOM_SOURCE_DIR}" \
RUOD_REF_ROOT="${RANDOM_REF_ROOT}" \
FID_REF_DIR="${RANDOM_REF_ROOT}" \
DEPTH_DIR="${DEPTH_DIR}" \
PREP_DIR="${PREP_DIR}" \
FLAT_SAVE_DIR="${FLAT_SAVE_DIR}" \
RESTORE_DIR="${RESTORE_DIR}" \
LIMIT="${NUM}" \
GPU="${GPU}" \
TEST_SIZE="${TEST_SIZE}" \
RUN_DEPTH=1 \
RUN_PREPARE=1 \
RUN_RUOD_REF=0 \
RUN_UWNR=1 \
RUN_RESTORE=1 \
bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh

echo
echo "Done."
echo "Flat outputs:     ${FLAT_SAVE_DIR}"
echo "Restored outputs: ${RESTORE_DIR}"
echo "Manifest:         ${WORK_ROOT}/random_selection_manifest.json"
