#!/usr/bin/env bash
set -euo pipefail

# Generate underwater-style ImageNet images with SyreaNet's physical synthesis
# module. This is different from SyreaNet test.py enhancement: it uses
# ImageNet RGB images plus MegaDepth pseudo-depth maps as inputs to
# synthesize/synthesize.py.
#
# Smoke:
#   SPLIT=train LIMIT=200 GPU=2 bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh
#
# Full single shard:
#   SPLIT=train LIMIT=0 GPU=2 bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh
#
# One shard of a multi-shard run:
#   SPLIT=train LIMIT=0 GPU=2 NUM_SHARDS=4 SHARD_INDEX=0 bash scripts/exp_2/synthesis/run_syreanet_synthesis_generate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-200}"
GPU="${GPU:-2}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

SOURCE_DIR="${SOURCE_DIR:-${SYN_ROOT}/syreanet/source/${SPLIT}}"
SYREANET_DIR="${SYREANET_DIR:-/home/fcp/xcx/exp_2/syn/SyreaNet}"
MEGADEPTH_DIR="${MEGADEPTH_DIR:-/home/fcp/xcx/exp_2/syn/MegaDepth}"
MEGADEPTH_CKPT="${MEGADEPTH_CKPT:-${MEGADEPTH_DIR}/checkpoints/best_generalization_net_G.pth}"

PREP_SIZE="${PREP_SIZE:-512}"
IMG_EXT="${IMG_EXT:-png}"
DEPTH_EXT="${DEPTH_EXT:-${IMG_EXT}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

SHARD_TAG=""
if [[ "${NUM_SHARDS}" != "1" ]]; then
  SHARD_TAG="_shard${SHARD_INDEX}of${NUM_SHARDS}"
fi

DEPTH_DIR="${DEPTH_DIR:-${SYN_ROOT}/syreanet_synthesis/depth/${SPLIT}${SHARD_TAG}}"
PREP_DIR="${PREP_DIR:-${SYN_ROOT}/syreanet_synthesis/prepared/${SPLIT}${SHARD_TAG}}"
FLAT_IMAGE_DIR="${PREP_DIR}/image"
FLAT_DEPTH_DIR="${PREP_DIR}/depth"
MANIFEST="${PREP_DIR}/manifest.jsonl"
FLAT_SAVE_DIR="${FLAT_SAVE_DIR:-${SYN_ROOT}/syreanet_synthesis/generated_flat/${SPLIT}${SHARD_TAG}}"
RESTORE_DIR="${RESTORE_DIR:-${SYN_ROOT}/syreanet_synthesis/generated/${SPLIT}}"

RUN_DEPTH="${RUN_DEPTH:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SYREANET="${RUN_SYREANET:-1}"
RUN_RESTORE="${RUN_RESTORE:-1}"
CLEAR_PREPARE="${CLEAR_PREPARE:-1}"
CLEAR_FLAT_OUTPUT="${CLEAR_FLAT_OUTPUT:-1}"

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
echo "SyreaNet synthesis generation"
echo "========================================="
echo "SYN_ROOT:       ${SYN_ROOT}"
echo "SPLIT:          ${SPLIT}"
echo "LIMIT:          ${LIMIT}"
echo "GPU:            ${GPU}"
echo "NUM_SHARDS:     ${NUM_SHARDS}"
echo "SHARD_INDEX:    ${SHARD_INDEX}"
echo "SOURCE_DIR:     ${SOURCE_DIR}"
echo "DEPTH_DIR:      ${DEPTH_DIR}"
echo "PREP_DIR:       ${PREP_DIR}"
echo "PREP_SIZE:      ${PREP_SIZE}"
echo "FLAT_SAVE_DIR:  ${FLAT_SAVE_DIR}"
echo "RESTORE_DIR:    ${RESTORE_DIR}"
echo "SYREANET_DIR:   ${SYREANET_DIR}"
echo "MEGADEPTH_DIR:  ${MEGADEPTH_DIR}"
echo "MEGADEPTH_CKPT: ${MEGADEPTH_CKPT}"
echo "========================================="
echo

check_path "${SOURCE_DIR}" "ImageNet sampled source"
check_path "${SYREANET_DIR}/synthesize/synthesize.py" "SyreaNet synthesis script"
check_path "${MEGADEPTH_DIR}" "MegaDepth directory"
check_path "${MEGADEPTH_CKPT}" "MegaDepth checkpoint"

if [[ "${RUN_DEPTH}" == "1" ]]; then
  echo "Step 1/4: Generate MegaDepth maps"
  python tools/generate_megadepth_maps.py \
    --image-dir "${SOURCE_DIR}" \
    --out-dir "${DEPTH_DIR}" \
    --megadepth-dir "${MEGADEPTH_DIR}" \
    --checkpoint "${MEGADEPTH_CKPT}" \
    --device "cuda:${GPU}" \
    --limit "${LIMIT}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}"
else
  echo "Step 1/4: Skip MegaDepth generation"
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo
  echo "Step 2/4: Prepare flat image/depth pairs"
  SOURCE_DIR="${SOURCE_DIR}" DEPTH_DIR="${DEPTH_DIR}" PREP_DIR="${PREP_DIR}" LIMIT="${LIMIT}" NUM_SHARDS="${NUM_SHARDS}" SHARD_INDEX="${SHARD_INDEX}" IMG_EXT="${IMG_EXT}" DEPTH_EXT="${DEPTH_EXT}" PREP_SIZE="${PREP_SIZE}" CLEAR_PREPARE="${CLEAR_PREPARE}" python - <<'PY'
from pathlib import Path
import json
import os
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

source_root = Path(os.environ["SOURCE_DIR"])
depth_root = Path(os.environ["DEPTH_DIR"])
prep_root = Path(os.environ["PREP_DIR"])
limit = int(os.environ["LIMIT"])
num_shards = int(os.environ["NUM_SHARDS"])
shard_index = int(os.environ["SHARD_INDEX"])
img_ext = os.environ["IMG_EXT"].lower().lstrip(".")
depth_ext = os.environ["DEPTH_EXT"].lower().lstrip(".")
prep_size = int(os.environ["PREP_SIZE"])
clear_prepare = os.environ.get("CLEAR_PREPARE", "1") == "1"

image_out = prep_root / "image"
depth_out = prep_root / "depth"
manifest = prep_root / "manifest.jsonl"
for directory in (image_out, depth_out):
    directory.mkdir(parents=True, exist_ok=True)
    if clear_prepare:
        for old in directory.iterdir():
            if old.is_file() or old.is_symlink():
                old.unlink()

image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
print(f"scanning images: {source_root}", flush=True)
images = []
for p in tqdm(source_root.rglob("*"), desc=f"scan {source_root.name}", unit="entry"):
    if p.is_file() and p.suffix.lower() in image_suffixes:
        images.append(p)
images.sort()
print(f"found images under {source_root}: {len(images)}", flush=True)
if limit > 0:
    images = images[:limit]
if num_shards > 1:
    images = images[shard_index::num_shards]

records = []
missing_depth = []
for source_index, image_path in enumerate(tqdm(images, desc="prepare SyreaNet synthesis", unit="image")):
    rel = image_path.relative_to(source_root)
    depth_path = depth_root / rel.with_suffix(".png")
    if not depth_path.exists():
        missing_depth.append(str(rel).replace("\\", "/"))
        continue

    index = len(records)
    stem = f"{index:08d}"
    image_link = image_out / f"{stem}.{img_ext}"
    depth_link = depth_out / f"{stem}.{depth_ext}"
    if image_link.exists() or image_link.is_symlink():
        image_link.unlink()
    if depth_link.exists() or depth_link.is_symlink():
        depth_link.unlink()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if prep_size > 0:
            image = ImageOps.fit(
                image,
                (prep_size, prep_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        image.save(image_link)

    with Image.open(depth_path) as depth:
        depth = depth.convert("L")
        if prep_size > 0:
            depth = ImageOps.fit(
                depth,
                (prep_size, prep_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        # SyreaNet looks up the depth file by the same basename and suffix as
        # the input image. Save a real image instead of a symlink so image/depth
        # dimensions cannot diverge or be interpreted with swapped axes.
        depth.save(depth_link)

    records.append({
        "index": index,
        "source_index": source_index,
        "relative": str(rel).replace("\\", "/"),
        "synset": rel.parts[0] if len(rel.parts) > 1 else "unknown",
        "original_name": rel.name,
        "source": str(image_path),
        "depth": str(depth_path),
        "flat_image": image_link.name,
        "flat_depth": depth_link.name,
    })

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

summary = {
    "source_root": str(source_root),
    "depth_root": str(depth_root),
    "prep_root": str(prep_root),
    "candidate_images": len(images),
    "prepared_pairs": len(records),
    "missing_depth": len(missing_depth),
    "missing_depth_samples": missing_depth[:20],
    "prep_size": prep_size,
    "image_ext": img_ext,
    "depth_ext": depth_ext,
    "num_shards": num_shards,
    "shard_index": shard_index,
    "manifest": str(manifest),
}
(prep_root / "prepare_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
if not records:
    raise SystemExit("No SyreaNet synthesis image/depth pairs prepared.")
PY
else
  echo
  echo "Step 2/4: Skip flat pair preparation"
fi

PAIR_COUNT="$(wc -l "${MANIFEST}" 2>/dev/null | awk '{print $1}')"
if [[ -z "${PAIR_COUNT}" || "${PAIR_COUNT}" == "0" ]]; then
  echo "Error: no prepared pairs found in ${MANIFEST}" >&2
  exit 1
fi
echo "Prepared pair count: ${PAIR_COUNT}"

if [[ "${RUN_SYREANET}" == "1" ]]; then
  echo
  echo "Step 3/4: Run SyreaNet synthesize.py"
  mkdir -p "${FLAT_SAVE_DIR}"
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
  (
    cd "${SYREANET_DIR}/synthesize"
    CUDA_VISIBLE_DEVICES="${GPU}" python synthesize.py \
      --image-dir "${FLAT_IMAGE_DIR}" \
      --depth-dir "${FLAT_DEPTH_DIR}" \
      --out-dir "${FLAT_SAVE_DIR}"
  ) 2>&1 | tee "${LOG_DIR}/syreanet_synthesis_${SPLIT}${SHARD_TAG}.log"
else
  echo
  echo "Step 3/4: Skip SyreaNet synthesis"
fi

if [[ "${RUN_RESTORE}" == "1" ]]; then
  echo
  echo "Step 4/4: Restore flat outputs to ImageNet synset directories"
  MANIFEST="${MANIFEST}" FLAT_SAVE_DIR="${FLAT_SAVE_DIR}" RESTORE_DIR="${RESTORE_DIR}" SHARD_TAG="${SHARD_TAG}" python - <<'PY'
from pathlib import Path
import json
import os
import shutil

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

manifest = Path(os.environ["MANIFEST"])
flat_root = Path(os.environ["FLAT_SAVE_DIR"])
restore_root = Path(os.environ["RESTORE_DIR"])
shard_tag = os.environ.get("SHARD_TAG", "")
image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
outputs_by_stem = {}
for path in flat_root.rglob("*"):
    if path.is_file() and path.suffix.lower() in image_suffixes:
        outputs_by_stem.setdefault(path.stem, path)

restore_root.mkdir(parents=True, exist_ok=True)
written = 0
skipped_existing = 0
missing = []
for record in tqdm(records, desc="restore SyreaNet synthesis", unit="image"):
    stem = Path(record["flat_image"]).stem
    generated = outputs_by_stem.get(stem)
    if generated is None:
        missing.append(record["flat_image"])
        continue
    dst_dir = restore_root / record["synset"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{Path(record['original_name']).stem}{generated.suffix.lower()}"
    if dst.exists():
        skipped_existing += 1
        continue
    shutil.copy2(generated, dst)
    written += 1

summary = {
    "manifest": str(manifest),
    "flat_root": str(flat_root),
    "restore_root": str(restore_root),
    "records": len(records),
    "flat_outputs": len(outputs_by_stem),
    "written": written,
    "skipped_existing": skipped_existing,
    "missing": len(missing),
    "missing_samples": missing[:20],
}
summary_path = restore_root / f"restore_summary{shard_tag}.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
else
  echo
  echo "Step 4/4: Skip output restoration"
fi

echo
echo "Done."
echo "Flat outputs:  ${FLAT_SAVE_DIR}"
echo "Final outputs: ${RESTORE_DIR}"
