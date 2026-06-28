#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GEN_DIR="${GEN_DIR:-/media/SSD1/XCX/exp_2/synthesis_work/stable_diffusion_diffusers/vae_text_smoke_10_simple_prompt}"
SRC_ROOT="${SRC_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/uwdf/source/train}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/sd_vae_text_simple_prompt_compare}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/sd_vae_text_10_simple_prompt.log}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-syn:datasets/exp2_synthesis_visual/}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Export Stable Diffusion comparison package"
echo "========================================="
echo "GEN_DIR:      ${GEN_DIR}"
echo "SRC_ROOT:     ${SRC_ROOT}"
echo "OUT_ROOT:     ${OUT_ROOT}"
echo "LOG_FILE:     ${LOG_FILE}"
echo "ARCHIVE_PATH: ${ARCHIVE_PATH}"
echo "UPLOAD:       ${UPLOAD}"
echo "RCLONE_DEST:  ${RCLONE_DEST}"
echo "OVERWRITE:    ${OVERWRITE}"
echo "========================================="

if [[ ! -d "${GEN_DIR}" ]]; then
  echo "Error: GEN_DIR not found: ${GEN_DIR}" >&2
  exit 1
fi

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Error: SRC_ROOT not found: ${SRC_ROOT}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi

mkdir -p \
  "${OUT_ROOT}/original" \
  "${OUT_ROOT}/generated" \
  "${OUT_ROOT}/compare" \
  "${OUT_ROOT}/logs"

GEN_DIR="${GEN_DIR}" \
SRC_ROOT="${SRC_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

gen_dir = Path(os.environ["GEN_DIR"])
src_root = Path(os.environ["SRC_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])

suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in suffixes


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if is_image(p))


def make_compare(original: Path, generated: Path, out: Path) -> bool:
    try:
        with Image.open(original) as a, Image.open(generated) as b:
            left = a.convert("RGB")
            right = b.convert("RGB")
    except Exception:
        return False

    size = (384, 384)
    left.thumbnail(size, Image.Resampling.BICUBIC)
    right.thumbnail(size, Image.Resampling.BICUBIC)

    pad = 12
    label_h = 34
    canvas = Image.new(
        "RGB",
        (size[0] * 2 + pad * 3, size[1] + label_h + pad * 2),
        "white",
    )

    lx = pad + (size[0] - left.width) // 2
    rx = pad * 2 + size[0] + (size[0] - right.width) // 2
    y = label_h + pad + (size[1] - max(left.height, right.height)) // 2

    canvas.paste(left, (lx, y))
    canvas.paste(right, (rx, y))

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), "original", fill=(0, 0, 0))
    draw.text((pad * 2 + size[0], 8), "stable diffusion generated", fill=(0, 0, 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return True


def generated_to_source_stem(path: Path) -> str:
    return path.stem.split("_underwater_")[0]


def find_original(source_stem: str) -> Path | None:
    synset = source_stem.split("_", 1)[0]
    candidate_dirs = [src_root / synset, src_root]
    candidate_suffixes = [".JPEG", ".jpeg", ".jpg", ".png", ".bmp", ".webp"]

    for candidate_dir in candidate_dirs:
        if not candidate_dir.is_dir():
            continue
        for suffix in candidate_suffixes:
            candidate = candidate_dir / f"{source_stem}{suffix}"
            if candidate.exists():
                return candidate

    # Slow fallback for unusual layouts only. The normal ImageNet-style layout
    # should match above without scanning the whole source tree.
    for suffix in candidate_suffixes:
        hits = list(src_root.rglob(f"{source_stem}{suffix}"))
        if hits:
            return hits[0]
    return None


generated_images = image_files(gen_dir)

records = []
for idx, gen in enumerate(tqdm(generated_images, desc="export SD comparisons", unit="image")):
    source_stem = generated_to_source_stem(gen)
    ori = find_original(source_stem)
    if ori is None:
        records.append({
            "index": idx,
            "generated": str(gen),
            "source_stem": source_stem,
            "original": None,
            "status": "missing_original",
        })
        continue

    name = f"{idx:03d}_{source_stem}"
    ori_dst = out_root / "original" / f"{name}{ori.suffix.lower()}"
    gen_dst = out_root / "generated" / f"{name}{gen.suffix.lower()}"
    cmp_dst = out_root / "compare" / f"{name}_compare.jpg"

    shutil.copy2(ori, ori_dst)
    shutil.copy2(gen, gen_dst)
    compared = make_compare(ori_dst, gen_dst, cmp_dst)

    records.append({
        "index": idx,
        "source_stem": source_stem,
        "original": str(ori),
        "generated": str(gen),
        "original_export": str(ori_dst),
        "generated_export": str(gen_dst),
        "compare": str(cmp_dst) if compared else None,
        "status": "ok" if compared else "compare_failed",
    })

summary = {
    "generated_dir": str(gen_dir),
    "source_root": str(src_root),
    "out_root": str(out_root),
    "total_generated": len(generated_images),
    "matched": sum(1 for r in records if r["status"] == "ok"),
    "missing_original": sum(1 for r in records if r["status"] == "missing_original"),
    "compare_failed": sum(1 for r in records if r["status"] == "compare_failed"),
    "records": records,
}

summary_path = out_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"summary: {summary_path}")
PY

if [[ -f "${LOG_FILE}" ]]; then
  cp -a "${LOG_FILE}" "${OUT_ROOT}/logs/"
else
  echo "Warning: LOG_FILE not found, skip log copy: ${LOG_FILE}" >&2
fi

echo
echo "Create archive"
rm -f "${ARCHIVE_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  echo
  echo "Upload archive"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone not found. Set UPLOAD=0 to skip upload." >&2
    exit 1
  fi
  rclone copy -P "${ARCHIVE_PATH}" "${RCLONE_DEST}"
else
  echo "Skip upload because UPLOAD=${UPLOAD}"
fi

echo
echo "Done."
echo "Output dir: ${OUT_ROOT}"
echo "Archive:    ${ARCHIVE_PATH}"
echo "Remote:     ${RCLONE_DEST}"
