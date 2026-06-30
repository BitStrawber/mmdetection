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
TRIPLET_DIR="${TRIPLET_DIR:-${WORK_ROOT}/triplets}"

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
echo "TRIPLET_DIR:        ${TRIPLET_DIR}"
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
echo "Build source/reference/generated triplets"
python - <<PY
from pathlib import Path
from PIL import Image, ImageDraw
import json

manifest_path = Path("${WORK_ROOT}") / "random_selection_manifest.json"
gen_dir = Path("${FLAT_SAVE_DIR}")
out_dir = Path("${TRIPLET_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
data = json.loads(manifest_path.read_text(encoding="utf-8"))
sources = data.get("sources", [])
refs = data.get("references", [])
generated = sorted(p for p in gen_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in exts)

def clear_outputs(path):
    for p in path.glob("*_uwnr_triplet.jpg"):
        p.unlink()

def load_image(path, size=(320, 320)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas

def with_label(img, text):
    label_h = 34
    out = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, label_h], fill=(245, 245, 245))
    draw.text((8, 9), text, fill=(0, 0, 0))
    out.paste(img, (0, label_h))
    return out

clear_outputs(out_dir)
n = min(len(sources), len(refs), len(generated))
if n == 0:
    raise RuntimeError(
        f"No valid triplet data: sources={len(sources)}, refs={len(refs)}, generated={len(generated)}")

for i in range(n):
    src_path = Path(sources[i]["selected"])
    ref_path = Path(refs[i]["selected"])
    gen_path = generated[i]

    src_img = with_label(load_image(src_path), f"source: {src_path.parent.name}")
    ref_img = with_label(load_image(ref_path), "reference")
    gen_img = with_label(load_image(gen_path), "uwnr generated")

    w, h = src_img.size
    triplet = Image.new("RGB", (w * 3, h), (255, 255, 255))
    triplet.paste(src_img, (0, 0))
    triplet.paste(ref_img, (w, 0))
    triplet.paste(gen_img, (w * 2, 0))
    triplet.save(out_dir / f"{i + 1:03d}_uwnr_triplet.jpg", quality=95)

print(f"sources: {len(sources)}")
print(f"refs: {len(refs)}")
print(f"generated: {len(generated)}")
print(f"triplets: {n}")
print(f"triplet_dir: {out_dir}")
PY

echo
echo "Done."
echo "Flat outputs:     ${FLAT_SAVE_DIR}"
echo "Restored outputs: ${RESTORE_DIR}"
echo "Triplets:         ${TRIPLET_DIR}"
echo "Manifest:         ${WORK_ROOT}/random_selection_manifest.json"
