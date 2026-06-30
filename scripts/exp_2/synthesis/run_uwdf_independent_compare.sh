#!/usr/bin/env bash
set -euo pipefail

# Independent UWDF visual comparison.
#
# Run this only after activating the UWDF environment:
#   conda activate /media/SSD1/conda_envs/uwdf
#   NUM=20 GPU=2 bash scripts/exp_2/synthesis/run_uwdf_independent_compare.sh
#
# This does not share samples with UWNR. It runs two UWDF variants on UWDF's own
# ImageNet source split:
#   1. text + ImageNet image, with IP_ADAPTER_SCALE=0.0
#   2. text + reference + ImageNet image, with IP_ADAPTER_SCALE=0.75 by default
#
# Output panel:
#   source | reference | text+image | text+reference+image

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU="${GPU:-2}"
SPLIT="${SPLIT:-train}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/${SPLIT}}"
REF_ROOT="${REF_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_ipadapter/independent_${SPLIT}_random${NUM}_seed${SEED}}"
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR:-${WORK_ROOT}/source/${SPLIT}}"
SELECTED_REFERENCE_DIR="${SELECTED_REFERENCE_DIR:-${WORK_ROOT}/reference/qingxi}"

HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STRENGTH="${STRENGTH:-0.35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
REF_IP_ADAPTER_SCALE="${REF_IP_ADAPTER_SCALE:-0.75}"
STEPS="${STEPS:-20}"
PROMPT="${PROMPT:-a realistic underwater photograph}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, extra objects, fish, coral, diver, text, watermark, blurry, low quality, worst quality}"

TEXT_IMG_DIR="${TEXT_IMG_DIR:-${WORK_ROOT}/uwdf_text_imagenet}"
TEXT_REF_IMG_DIR="${TEXT_REF_IMG_DIR:-${WORK_ROOT}/uwdf_text_ref_imagenet}"
PANEL_DIR="${PANEL_DIR:-${WORK_ROOT}/compare_4panel}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

echo "========================================="
echo "Independent UWDF comparison"
echo "========================================="
echo "NUM:                  ${NUM}"
echo "SEED:                 ${SEED}"
echo "GPU:                  ${GPU}"
echo "SPLIT:                ${SPLIT}"
echo "UWDF_DIR:             ${UWDF_DIR}"
echo "SOURCE_ROOT:          ${SOURCE_ROOT}"
echo "REF_ROOT:             ${REF_ROOT}"
echo "SELECTED_SOURCE_DIR:  ${SELECTED_SOURCE_DIR}"
echo "SELECTED_REFERENCE:   ${SELECTED_REFERENCE_DIR}"
echo "WORK_ROOT:            ${WORK_ROOT}"
echo "SIZE:                 ${WIDTH}x${HEIGHT}"
echo "STRENGTH:             ${STRENGTH}"
echo "GUIDANCE_SCALE:       ${GUIDANCE_SCALE}"
echo "REF_IP_ADAPTER_SCALE: ${REF_IP_ADAPTER_SCALE}"
echo "PROMPT:               ${PROMPT}"
echo "========================================="

if [[ ! -d "${UWDF_DIR}" ]]; then
  echo "Error: UWDF_DIR not found: ${UWDF_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Error: SOURCE_ROOT not found: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${REF_ROOT}" ]]; then
  echo "Error: REF_ROOT not found: ${REF_ROOT}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}" "${LOG_DIR}"

echo
echo "Step 1/4: Select independent random source/reference for UWDF"
SOURCE_ROOT="${SOURCE_ROOT}" \
REF_ROOT="${REF_ROOT}" \
SELECTED_SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
SELECTED_REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
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
ref_out = Path(os.environ["SELECTED_REFERENCE_DIR"])
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
        cls = rel.parts[0] if len(rel.parts) >= 2 else "_flat"
        by_cls.setdefault(cls, []).append(p)

classes = sorted(by_cls)
if not classes:
    raise RuntimeError(f"No source images found under {source_root}")

rng = random.Random(seed)
if len(classes) >= num:
    picked_classes = rng.sample(classes, num)
    picked_sources = [rng.choice(sorted(by_cls[cls])) for cls in picked_classes]
else:
    all_sources = sorted(p for paths in by_cls.values() for p in paths)
    picked_sources = rng.sample(all_sources, min(num, len(all_sources)))

sources = []
for idx, src in enumerate(picked_sources):
    rel = src.relative_to(source_root)
    dst = source_out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    sources.append({
        "index": idx,
        "class": rel.parts[0] if len(rel.parts) >= 2 else "_flat",
        "source": str(src),
        "selected": str(dst),
        "relative": str(rel).replace("\\", "/"),
    })

refs_all = sorted(p for p in ref_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
if not refs_all:
    raise RuntimeError(f"No reference images found under {ref_root}")
if len(refs_all) < len(sources):
    picked_refs = [rng.choice(refs_all) for _ in sources]
else:
    picked_refs = rng.sample(refs_all, len(sources))

refs = []
for idx, src in enumerate(picked_refs):
    dst = ref_out / f"{idx:08d}{src.suffix.lower()}"
    os.symlink(src, dst)
    refs.append({"index": idx, "source": str(src), "selected": str(dst)})

manifest = {
    "num": len(sources),
    "requested_num": num,
    "seed": seed,
    "source_root": str(source_root),
    "reference_root": str(ref_root),
    "selected_source_dir": str(source_out),
    "selected_reference_dir": str(ref_out),
    "sources": sources,
    "references": refs,
}
(work_root / "uwdf_selection_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"selected sources: {len(sources)}")
print(f"selected classes: {len(set(x['class'] for x in sources))}")
print(f"selected references: {len(refs)}")
print(f"manifest: {work_root / 'uwdf_selection_manifest.json'}")
PY

echo
echo "Step 2/4: UWDF text + ImageNet"
(
  cd "${UWDF_DIR}"
  GPU="${GPU}" \
  SPLIT="${SPLIT}" \
  SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
  REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
  OUT_DIR="${TEXT_IMG_DIR}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  STRENGTH="${STRENGTH}" \
  GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
  IP_ADAPTER_SCALE=0.0 \
  STEPS="${STEPS}" \
  LIMIT="${NUM}" \
  SEED="${SEED}" \
  PROMPT="${PROMPT}" \
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
  SAVE_COMPARISON=0 \
  bash scripts/run_ipadapter_img2img_generate.sh
) 2>&1 | tee "${LOG_DIR}/uwdf_text_imagenet.log"

echo
echo "Step 3/4: UWDF text + reference + ImageNet"
(
  cd "${UWDF_DIR}"
  GPU="${GPU}" \
  SPLIT="${SPLIT}" \
  SOURCE_DIR="${SELECTED_SOURCE_DIR}" \
  REFERENCE_DIR="${SELECTED_REFERENCE_DIR}" \
  OUT_DIR="${TEXT_REF_IMG_DIR}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  STRENGTH="${STRENGTH}" \
  GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
  IP_ADAPTER_SCALE="${REF_IP_ADAPTER_SCALE}" \
  STEPS="${STEPS}" \
  LIMIT="${NUM}" \
  SEED="${SEED}" \
  PROMPT="${PROMPT}" \
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
  SAVE_COMPARISON=0 \
  bash scripts/run_ipadapter_img2img_generate.sh
) 2>&1 | tee "${LOG_DIR}/uwdf_text_ref_imagenet.log"

echo
echo "Step 4/4: Build UWDF four-panel comparisons"
TEXT_IMG_DIR="${TEXT_IMG_DIR}" \
TEXT_REF_IMG_DIR="${TEXT_REF_IMG_DIR}" \
PANEL_DIR="${PANEL_DIR}" \
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import json
import os

text_dir = Path(os.environ["TEXT_IMG_DIR"])
ref_dir = Path(os.environ["TEXT_REF_IMG_DIR"])
panel_dir = Path(os.environ["PANEL_DIR"])
panel_dir.mkdir(parents=True, exist_ok=True)

text_manifest = text_dir / "manifest.jsonl"
ref_manifest = ref_dir / "manifest.jsonl"
if not text_manifest.exists():
    raise RuntimeError(f"Missing manifest: {text_manifest}")
if not ref_manifest.exists():
    raise RuntimeError(f"Missing manifest: {ref_manifest}")

def read_manifest(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

text_records = read_manifest(text_manifest)
ref_records = read_manifest(ref_manifest)
ref_by_source = {r["source"]: r for r in ref_records}

for p in panel_dir.glob("*_uwdf_4panel.jpg"):
    p.unlink()

def load_tile(path, size=(320, 320)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, (255, 255, 255))
    tile.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return tile

def label(tile, text):
    label_h = 34
    out = Image.new("RGB", (tile.width, tile.height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, tile.width, label_h], fill=(245, 245, 245))
    draw.text((8, 9), text, fill=(0, 0, 0))
    out.paste(tile, (0, label_h))
    return out

written = 0
missing = []
for i, text_record in enumerate(text_records):
    ref_record = ref_by_source.get(text_record["source"])
    if ref_record is None:
        missing.append({"source": text_record["source"], "reason": "missing_ref_variant_record"})
        continue
    paths = {
        "source": Path(text_record["source"]),
        "reference": Path(ref_record["reference"]),
        "text_image": Path(text_record["output"]),
        "text_ref_image": Path(ref_record["output"]),
    }
    bad = [name for name, path in paths.items() if not path.exists()]
    if bad:
        missing.append({"source": text_record["source"], "reason": f"missing_files:{','.join(bad)}"})
        continue

    tiles = [
        label(load_tile(paths["source"]), "source"),
        label(load_tile(paths["reference"]), "reference"),
        label(load_tile(paths["text_image"]), "uwdf text+image"),
        label(load_tile(paths["text_ref_image"]), "uwdf text+ref+image"),
    ]
    w, h = tiles[0].size
    panel = Image.new("RGB", (w * len(tiles), h), (255, 255, 255))
    for j, tile in enumerate(tiles):
        panel.paste(tile, (j * w, 0))
    panel.save(panel_dir / f"{i + 1:03d}_uwdf_4panel.jpg", quality=95)
    written += 1

summary = {
    "text_manifest": str(text_manifest),
    "ref_manifest": str(ref_manifest),
    "panel_dir": str(panel_dir),
    "text_records": len(text_records),
    "ref_records": len(ref_records),
    "written": written,
    "missing": missing[:50],
}
(panel_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "Done."
echo "UWDF text+image:     ${TEXT_IMG_DIR}/generated"
echo "UWDF text+ref+image: ${TEXT_REF_IMG_DIR}/generated"
echo "UWDF four panels:    ${PANEL_DIR}"
