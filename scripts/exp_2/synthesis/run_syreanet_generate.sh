#!/usr/bin/env bash
set -euo pipefail

# Generate synthetic images with SyreaNet.
#
# SyreaNet does not require a real underwater reference directory at inference
# time. This script prepares a flat resized ImageNet subset, runs the official
# SyreaNet test.py, then restores generated images to ImageNet-style class
# folders.
#
# Examples:
#   # Smoke test, 200 train images on physical GPU 2
#   SPLIT=train LIMIT=200 GPU=2 bash scripts/exp_2/synthesis/run_syreanet_generate.sh
#
#   # Full train shard 0/4 on physical GPU 2
#   SPLIT=train LIMIT=0 GPU=2 NUM_SHARDS=4 SHARD_INDEX=0 bash scripts/exp_2/synthesis/run_syreanet_generate.sh

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-200}"
GPU="${GPU:-2}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
IMG_SIZE="${IMG_SIZE:-512}"
IMG_EXT="${IMG_EXT:-jpg}"

SYREANET_DIR="${SYREANET_DIR:-/home/fcp/xcx/exp_2/syn/SyreaNet}"
SYREANET_CKPT="${SYREANET_CKPT:-${SYREANET_DIR}/checkpoints/pretrained.pth}"
SYREANET_BASE_CONFIG="${SYREANET_BASE_CONFIG:-${SYREANET_DIR}/configs/syreanet_test.yaml}"

DEFAULT_SOURCE_DIR="${SYN_ROOT}/syreanet/source/${SPLIT}"
FALLBACK_SOURCE_DIR="${SYN_ROOT}/uwnr/source/${SPLIT}"
SOURCE_DIR="${SOURCE_DIR:-${DEFAULT_SOURCE_DIR}}"
if [[ ! -d "${SOURCE_DIR}" && -d "${FALLBACK_SOURCE_DIR}" ]]; then
  SOURCE_DIR="${FALLBACK_SOURCE_DIR}"
fi

SHARD_TAG=""
if [[ "${NUM_SHARDS}" != "1" ]]; then
  SHARD_TAG="_shard${SHARD_INDEX}of${NUM_SHARDS}"
fi

PREP_DIR="${PREP_DIR:-${SYN_ROOT}/syreanet/prepared/${SPLIT}${SHARD_TAG}}"
FLAT_INPUT_DIR="${PREP_DIR}/input"
MANIFEST="${PREP_DIR}/manifest.jsonl"
RUN_CONFIG="${PREP_DIR}/syreanet_${SPLIT}${SHARD_TAG}.yaml"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${SYN_ROOT}/syreanet/generated_flat/${SPLIT}${SHARD_TAG}}"
FINAL_SAVE_DIR="${FINAL_SAVE_DIR:-${SYN_ROOT}/syreanet/generated/${SPLIT}}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SYREANET="${RUN_SYREANET:-1}"
RUN_RESTORE="${RUN_RESTORE:-1}"

echo "========================================="
echo "SyreaNet synthetic ImageNet generation"
echo "========================================="
echo "SYN_ROOT:            ${SYN_ROOT}"
echo "SOURCE_DIR:          ${SOURCE_DIR}"
echo "SPLIT:               ${SPLIT}"
echo "LIMIT:               ${LIMIT}"
echo "GPU:                 ${GPU}"
echo "NUM_SHARDS:          ${NUM_SHARDS}"
echo "SHARD_INDEX:         ${SHARD_INDEX}"
echo "IMG_SIZE:            ${IMG_SIZE}"
echo "PREP_DIR:            ${PREP_DIR}"
echo "FLAT_INPUT_DIR:      ${FLAT_INPUT_DIR}"
echo "FLAT_SAVE_DIR:       ${FLAT_SAVE_DIR}"
echo "FINAL_SAVE_DIR:      ${FINAL_SAVE_DIR}"
echo "SYREANET_DIR:        ${SYREANET_DIR}"
echo "SYREANET_CKPT:       ${SYREANET_CKPT}"
echo "SYREANET_BASE_CONFIG:${SYREANET_BASE_CONFIG}"
echo "========================================="
echo

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Error: source directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SYREANET_CKPT}" ]]; then
  echo "Error: SyreaNet checkpoint does not exist: ${SYREANET_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${SYREANET_BASE_CONFIG}" ]]; then
  echo "Error: SyreaNet base config does not exist: ${SYREANET_BASE_CONFIG}" >&2
  exit 1
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo "Step 1/3: Prepare flat resized input directory"
  mkdir -p "${FLAT_INPUT_DIR}"
  python - "${SOURCE_DIR}" "${FLAT_INPUT_DIR}" "${MANIFEST}" "${LIMIT}" "${NUM_SHARDS}" "${SHARD_INDEX}" "${IMG_SIZE}" "${IMG_EXT}" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

source_dir = Path(sys.argv[1])
flat_input_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
limit = int(sys.argv[4])
num_shards = int(sys.argv[5])
shard_index = int(sys.argv[6])
img_size = int(sys.argv[7])
img_ext = sys.argv[8].lower().lstrip(".")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
all_images = sorted(
    p for p in source_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in {e.lower() for e in exts}
)
if limit > 0:
    all_images = all_images[:limit]
images = all_images[shard_index::num_shards]

flat_input_dir.mkdir(parents=True, exist_ok=True)
manifest_path.parent.mkdir(parents=True, exist_ok=True)

records = []
written = 0
skipped = 0
failed = []

for idx, src in enumerate(tqdm(images, desc="prepare SyreaNet input", unit="image")):
    rel = src.relative_to(source_dir)
    synset = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    flat_name = f"{idx:08d}.{img_ext}"
    dst = flat_input_dir / flat_name
    try:
        if dst.exists():
            skipped += 1
        else:
            with Image.open(src) as im:
                im = im.convert("RGB")
                if img_size > 0:
                    im = ImageOps.contain(im, (img_size, img_size), Image.Resampling.BICUBIC)
                im.save(dst, quality=95)
            written += 1
        records.append({
            "flat_name": flat_name,
            "source": str(src),
            "source_rel": str(rel),
            "synset": synset,
            "original_name": src.name,
        })
    except Exception as exc:
        failed.append({"source": str(src), "error": repr(exc)})

with manifest_path.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

summary = {
    "source_dir": str(source_dir),
    "flat_input_dir": str(flat_input_dir),
    "manifest": str(manifest_path),
    "limit": limit,
    "num_shards": num_shards,
    "shard_index": shard_index,
    "total_before_limit": len(all_images) if limit <= 0 else None,
    "prepared": len(records),
    "written": written,
    "skipped_existing": skipped,
    "failed": len(failed),
    "failures": failed[:20],
}
(manifest_path.parent / "prepare_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

  python - "${SYREANET_BASE_CONFIG}" "${RUN_CONFIG}" "${FLAT_INPUT_DIR}" "${IMG_EXT}" <<'PY'
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])
out = Path(sys.argv[2])
data_path = sys.argv[3]
img_fmt = sys.argv[4]

text = base.read_text(encoding="utf-8")
text = re.sub(r'data_path:\s*["\']?.*?["\']?\s*$', f'      data_path: "{data_path}"', text, count=1, flags=re.M)
text = re.sub(r'img_fmt:\s*["\']?.*?["\']?\s*$', f'      img_fmt: "{img_fmt}"', text, count=1, flags=re.M)
text = re.sub(r'cuda:\s*["\']?.*?["\']?\s*$', 'cuda: "0"', text, count=1, flags=re.M)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
print(f"config: {out}")
PY
fi

if [[ "${RUN_SYREANET}" == "1" ]]; then
  echo
  echo "Step 2/3: Run official SyreaNet test.py"
  mkdir -p "${FLAT_SAVE_DIR}"
  (
    cd "${SYREANET_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
      --config "${RUN_CONFIG}" \
      --load-path "${SYREANET_CKPT}" \
      --output-dir "${FLAT_SAVE_DIR}"
  )
fi

if [[ "${RUN_RESTORE}" == "1" ]]; then
  echo
  echo "Step 3/3: Restore flat outputs to ImageNet-style class folders"
  python - "${MANIFEST}" "${FLAT_SAVE_DIR}" "${FINAL_SAVE_DIR}" <<'PY'
import json
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

manifest = Path(sys.argv[1])
flat_save_dir = Path(sys.argv[2])
final_save_dir = Path(sys.argv[3])
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

outputs_by_stem = {}
for p in flat_save_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in image_exts:
        outputs_by_stem.setdefault(p.stem, p)

written = 0
missing = []
with manifest.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

for rec in tqdm(records, desc="restore SyreaNet outputs", unit="image"):
    stem = Path(rec["flat_name"]).stem
    out = outputs_by_stem.get(stem)
    if out is None:
        missing.append(rec["flat_name"])
        continue
    dst_dir = final_save_dir / rec["synset"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(rec["original_name"]).stem
    dst = dst_dir / f"{source_stem}{out.suffix.lower()}"
    if not dst.exists():
        shutil.copy2(out, dst)
        written += 1

summary = {
    "manifest": str(manifest),
    "flat_save_dir": str(flat_save_dir),
    "final_save_dir": str(final_save_dir),
    "records": len(records),
    "flat_outputs": len(outputs_by_stem),
    "written": written,
    "missing": len(missing),
    "missing_samples": missing[:20],
}
(final_save_dir / f"restore_summary{Path(manifest.parent).name}.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
fi

echo
echo "Done."
echo "Flat outputs:  ${FLAT_SAVE_DIR}"
echo "Final outputs: ${FINAL_SAVE_DIR}"
