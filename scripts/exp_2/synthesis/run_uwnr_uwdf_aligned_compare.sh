#!/usr/bin/env bash
set -euo pipefail

# Build an aligned smoke comparison for UWNR and UWDF.
#
# It creates one random set of ImageNet source images and one random set of
# underwater references, regenerates UWNR, runs two UWDF variants on the same
# selected inputs, then exports aligned comparison grids.
#
# UWDF variants:
#   1. text + ImageNet image: IP_ADAPTER_SCALE=0
#   2. text + reference + ImageNet image: IP_ADAPTER_SCALE=0.75 by default
#
# Outputs:
#   compare_4panel: source | reference | uwdf_text_img | uwdf_text_ref_img
#   compare_5panel: source | reference | uwnr | uwdf_text_img | uwdf_text_ref_img

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU="${GPU:-2}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwnr/source/train}"
REF_ROOT="${REF_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/aligned_uwnr_uwdf_compare/random${NUM}_seed${SEED}}"

SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR:-${WORK_ROOT}/source/train}"
SELECTED_REF_ROOT="${SELECTED_REF_ROOT:-${WORK_ROOT}/reference}"
SELECTED_REF_QINGXI="${SELECTED_REF_ROOT}/qingxi"

UWNR_DEPTH_DIR="${UWNR_DEPTH_DIR:-${WORK_ROOT}/uwnr/megadepth/train}"
UWNR_PREP_DIR="${UWNR_PREP_DIR:-${WORK_ROOT}/uwnr/prepared/train}"
UWNR_FLAT_DIR="${UWNR_FLAT_DIR:-${WORK_ROOT}/uwnr/generated_flat/train}"
UWNR_RESTORE_DIR="${UWNR_RESTORE_DIR:-${WORK_ROOT}/uwnr/generated/train}"
UWNR_TEST_SIZE="${UWNR_TEST_SIZE:-256}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
UWDF_HEIGHT="${UWDF_HEIGHT:-1024}"
UWDF_WIDTH="${UWDF_WIDTH:-1024}"
UWDF_STRENGTH="${UWDF_STRENGTH:-0.35}"
UWDF_GUIDANCE_SCALE="${UWDF_GUIDANCE_SCALE:-5.0}"
UWDF_REF_SCALE="${UWDF_REF_SCALE:-0.75}"
UWDF_STEPS="${UWDF_STEPS:-20}"
UWDF_TEXT_IMG_DIR="${UWDF_TEXT_IMG_DIR:-${WORK_ROOT}/uwdf_text_imagenet}"
UWDF_TEXT_REF_IMG_DIR="${UWDF_TEXT_REF_IMG_DIR:-${WORK_ROOT}/uwdf_text_ref_imagenet}"

COMPARE_4_DIR="${COMPARE_4_DIR:-${WORK_ROOT}/compare_4panel}"
COMPARE_5_DIR="${COMPARE_5_DIR:-${WORK_ROOT}/compare_5panel}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

echo "========================================="
echo "Aligned UWNR/UWDF comparison"
echo "========================================="
echo "NUM:                  ${NUM}"
echo "SEED:                 ${SEED}"
echo "GPU:                  ${GPU}"
echo "SOURCE_ROOT:          ${SOURCE_ROOT}"
echo "REF_ROOT:             ${REF_ROOT}"
echo "WORK_ROOT:            ${WORK_ROOT}"
echo "SELECTED_SOURCE_DIR:  ${SELECTED_SOURCE_DIR}"
echo "SELECTED_REF_ROOT:    ${SELECTED_REF_ROOT}"
echo "UWDF_DIR:             ${UWDF_DIR}"
echo "UWDF_SIZE:            ${UWDF_WIDTH}x${UWDF_HEIGHT}"
echo "UWDF_STRENGTH:        ${UWDF_STRENGTH}"
echo "UWDF_REF_SCALE:       ${UWDF_REF_SCALE}"
echo "========================================="

mkdir -p "${WORK_ROOT}" "${LOG_DIR}"

echo
echo "Step 1/5: Select aligned random source/reference"
SOURCE_ROOT="${SOURCE_ROOT}" \
REF_ROOT="${REF_ROOT}" \
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
SELECTED_REF_QINGXI="${SELECTED_REF_QINGXI}" \
WORK_ROOT="${WORK_ROOT}" \
NUM="${NUM}" \
SEED="${SEED}" \
python - <<'PY'
from pathlib import Path
import json
import os
import random
import shutil

source_root = Path(os.environ["SOURCE_ROOT"])
ref_root = Path(os.environ["REF_ROOT"])
source_out = Path(os.environ["SELECTED_SOURCE_DIR"])
ref_out = Path(os.environ["SELECTED_REF_QINGXI"])
work_root = Path(os.environ["WORK_ROOT"])
num = int(os.environ["NUM"])
seed = int(os.environ["SEED"])
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
clear_tree(ref_out)
source_out.mkdir(parents=True, exist_ok=True)
ref_out.mkdir(parents=True, exist_ok=True)

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
sources = []
for idx, cls in enumerate(picked_classes):
    src = rng.choice(sorted(by_cls[cls]))
    rel = src.relative_to(source_root)
    dst = source_out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    sources.append({
        "index": idx,
        "class": cls,
        "source": str(src),
        "selected": str(dst),
        "relative": str(rel).replace("\\", "/"),
    })

refs_all = sorted(p for p in ref_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if len(refs_all) < num:
    raise RuntimeError(f"Not enough reference images: {len(refs_all)} < {num}")

refs = []
for idx, src in enumerate(rng.sample(refs_all, num)):
    dst = ref_out / f"{idx:08d}{src.suffix.lower()}"
    os.symlink(src, dst)
    refs.append({"index": idx, "source": str(src), "selected": str(dst)})

manifest = {
    "num": num,
    "seed": seed,
    "source_root": str(source_root),
    "reference_root": str(ref_root),
    "selected_source_dir": str(source_out),
    "selected_reference_root": str(ref_out.parent),
    "sources": sources,
    "references": refs,
}
(work_root / "aligned_selection_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"selected sources: {len(sources)}")
print(f"selected source classes: {len(set(x['class'] for x in sources))}")
print(f"selected references: {len(refs)}")
print(f"manifest: {work_root / 'aligned_selection_manifest.json'}")
PY

echo
echo "Step 2/5: Regenerate aligned UWNR outputs"
SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
RUOD_REF_ROOT="${SELECTED_REF_ROOT}" \
FID_REF_DIR="${SELECTED_REF_ROOT}" \
DEPTH_DIR="${UWNR_DEPTH_DIR}" \
PREP_DIR="${UWNR_PREP_DIR}" \
FLAT_SAVE_DIR="${UWNR_FLAT_DIR}" \
RESTORE_DIR="${UWNR_RESTORE_DIR}" \
LIMIT="${NUM}" \
GPU="${GPU}" \
TEST_SIZE="${UWNR_TEST_SIZE}" \
RUN_DEPTH=1 \
RUN_PREPARE=1 \
RUN_RUOD_REF=0 \
RUN_UWNR=1 \
RUN_RESTORE=1 \
bash scripts/exp_2/synthesis/run_uwnr_ruod_ref_generate.sh \
  2>&1 | tee "${LOG_DIR}/uwnr.log"

echo
echo "Step 3/5: Run UWDF text + ImageNet only"
(
  cd "${UWDF_DIR}"
  GPU="${GPU}" \
  SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
  REFERENCE_DIR="${SELECTED_REF_QINGXI}" \
  OUT_DIR="${UWDF_TEXT_IMG_DIR}" \
  HEIGHT="${UWDF_HEIGHT}" \
  WIDTH="${UWDF_WIDTH}" \
  STRENGTH="${UWDF_STRENGTH}" \
  GUIDANCE_SCALE="${UWDF_GUIDANCE_SCALE}" \
  IP_ADAPTER_SCALE=0.0 \
  STEPS="${UWDF_STEPS}" \
  LIMIT="${NUM}" \
  SAVE_COMPARISON=0 \
  bash scripts/run_ipadapter_img2img_generate.sh
) 2>&1 | tee "${LOG_DIR}/uwdf_text_imagenet.log"

echo
echo "Step 4/5: Run UWDF text + reference + ImageNet"
(
  cd "${UWDF_DIR}"
  GPU="${GPU}" \
  SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
  REFERENCE_DIR="${SELECTED_REF_QINGXI}" \
  OUT_DIR="${UWDF_TEXT_REF_IMG_DIR}" \
  HEIGHT="${UWDF_HEIGHT}" \
  WIDTH="${UWDF_WIDTH}" \
  STRENGTH="${UWDF_STRENGTH}" \
  GUIDANCE_SCALE="${UWDF_GUIDANCE_SCALE}" \
  IP_ADAPTER_SCALE="${UWDF_REF_SCALE}" \
  STEPS="${UWDF_STEPS}" \
  LIMIT="${NUM}" \
  SAVE_COMPARISON=0 \
  bash scripts/run_ipadapter_img2img_generate.sh
) 2>&1 | tee "${LOG_DIR}/uwdf_text_ref_imagenet.log"

echo
echo "Step 5/5: Build aligned comparison panels"
WORK_ROOT="${WORK_ROOT}" \
UWNR_PREP_DIR="${UWNR_PREP_DIR}" \
UWNR_FLAT_DIR="${UWNR_FLAT_DIR}" \
UWDF_TEXT_IMG_DIR="${UWDF_TEXT_IMG_DIR}" \
UWDF_TEXT_REF_IMG_DIR="${UWDF_TEXT_REF_IMG_DIR}" \
COMPARE_4_DIR="${COMPARE_4_DIR}" \
COMPARE_5_DIR="${COMPARE_5_DIR}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import json
import os

work_root = Path(os.environ["WORK_ROOT"])
selection = json.loads((work_root / "aligned_selection_manifest.json").read_text(encoding="utf-8"))
pair_manifest = Path(os.environ["UWNR_PREP_DIR"]) / "pair_manifest.jsonl"
uwnr_flat = Path(os.environ["UWNR_FLAT_DIR"])
uwdf_text = Path(os.environ["UWDF_TEXT_IMG_DIR"]) / "generated"
uwdf_ref = Path(os.environ["UWDF_TEXT_REF_IMG_DIR"]) / "generated"
compare4 = Path(os.environ["COMPARE_4_DIR"])
compare5 = Path(os.environ["COMPARE_5_DIR"])
compare4.mkdir(parents=True, exist_ok=True)
compare5.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
pairs = [json.loads(line) for line in pair_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
refs = selection["references"]
uwnr_outputs = sorted(p for p in uwnr_flat.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def clear(path):
    for p in path.glob("*.jpg"):
        p.unlink()

def find_by_relative(root, rel):
    base = (root / rel).with_suffix("")
    candidates = []
    for suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        p = base.with_suffix(suffix)
        if p.exists():
            candidates.append(p)
    if candidates:
        return candidates[0]
    name = Path(rel).stem
    matches = sorted(root.rglob(f"{name}*"))
    matches = [p for p in matches if p.is_file() and p.suffix.lower() in exts]
    if not matches:
        raise FileNotFoundError(f"No output for {rel} under {root}")
    return matches[0]

def load(path, size=(300, 300)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas

def label(img, text):
    h = 34
    out = Image.new("RGB", (img.width, img.height + h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, h], fill=(245, 245, 245))
    draw.text((8, 9), text, fill=(0, 0, 0))
    out.paste(img, (0, h))
    return out

def concat(images):
    w, h = images[0].size
    panel = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for i, img in enumerate(images):
        panel.paste(img, (i * w, 0))
    return panel

clear(compare4)
clear(compare5)
rows = []
for i, record in enumerate(pairs):
    rel = Path(record["relative"])
    src_path = Path(record["source"])
    ref_path = Path(refs[i]["selected"])
    uwnr_path = uwnr_outputs[i]
    uwdf_text_path = find_by_relative(uwdf_text, rel)
    uwdf_ref_path = find_by_relative(uwdf_ref, rel)

    source = label(load(src_path), f"source: {record['synset']}")
    ref = label(load(ref_path), "reference")
    uwnr = label(load(uwnr_path), "uwnr")
    text = label(load(uwdf_text_path), "uwdf text+image")
    refimg = label(load(uwdf_ref_path), "uwdf text+ref+image")

    concat([source, ref, text, refimg]).save(compare4 / f"{i:03d}_uwdf_4panel.jpg", quality=95)
    concat([source, ref, uwnr, text, refimg]).save(compare5 / f"{i:03d}_uwnr_uwdf_5panel.jpg", quality=95)
    rows.append({
        "index": i,
        "relative": str(rel).replace("\\", "/"),
        "source": str(src_path),
        "reference": str(ref_path),
        "uwnr": str(uwnr_path),
        "uwdf_text_image": str(uwdf_text_path),
        "uwdf_text_ref_image": str(uwdf_ref_path),
    })

(work_root / "aligned_outputs_manifest.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"pairs: {len(rows)}")
print(f"compare_4panel: {compare4}")
print(f"compare_5panel: {compare5}")
PY

echo
echo "Done."
echo "Work root:      ${WORK_ROOT}"
echo "4-panel output: ${COMPARE_4_DIR}"
echo "5-panel output: ${COMPARE_5_DIR}"
