#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/synthesis_visual_pairs}"
MAX_PER_METHOD="${MAX_PER_METHOD:-20}"
METHOD_TIMEOUT_SEC="${METHOD_TIMEOUT_SEC:-60}"
UPLOAD="${UPLOAD:-1}"
RCLONE_DEST="${RCLONE_DEST:-syn:datasets/exp2_synthesis_visual/}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUT_ROOT}.tar.gz}"
OVERWRITE="${OVERWRITE:-1}"

echo "========================================="
echo "Export synthesis visual comparison pairs"
echo "========================================="
echo "SYN_ROOT:       ${SYN_ROOT}"
echo "WORK_ROOT:      ${WORK_ROOT}"
echo "OUT_ROOT:       ${OUT_ROOT}"
echo "MAX_PER_METHOD: ${MAX_PER_METHOD}"
echo "METHOD_TIMEOUT_SEC: ${METHOD_TIMEOUT_SEC}"
echo "ARCHIVE_PATH:   ${ARCHIVE_PATH}"
echo "UPLOAD:         ${UPLOAD}"
echo "RCLONE_DEST:    ${RCLONE_DEST}"
echo "OVERWRITE:      ${OVERWRITE}"
echo "========================================="

if [[ "${OVERWRITE}" == "1" && -d "${OUT_ROOT}" ]]; then
  rm -rf "${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}"

SYN_ROOT="${SYN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
MAX_PER_METHOD="${MAX_PER_METHOD}" \
METHOD_TIMEOUT_SEC="${METHOD_TIMEOUT_SEC}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

syn_root = Path(os.environ["SYN_ROOT"])
work_root = Path(os.environ["WORK_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])
max_per_method = int(os.environ["MAX_PER_METHOD"])
method_timeout_sec = float(os.environ["METHOD_TIMEOUT_SEC"])


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if is_image(p))


def find_by_stem(root: Path, stem: str) -> Path | None:
    if not root.is_dir():
        return None
    for suffix in (".JPEG", ".jpeg", ".jpg", ".png", ".bmp", ".webp"):
        candidates = list(root.rglob(f"{stem}{suffix}"))
        if candidates:
            return candidates[0]
    for p in root.rglob("*"):
        if is_image(p) and p.stem == stem:
            return p
    return None


def build_stem_index(source_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for path in tqdm(image_files(source_root), desc=f"index {source_root.name}", unit="image"):
            index.setdefault(path.stem, path)
    return index


def build_generated_index(generated_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in tqdm(image_files(generated_root), desc=f"index generated {generated_root.name}", unit="image"):
        stems = {path.stem}
        if path.stem.endswith("_fake_B"):
            stems.add(path.stem[:-7])
        if "_underwater_" in path.stem:
            stems.add(path.stem.split("_underwater_")[0])
        for stem in stems:
            index.setdefault(stem, path)
    return index


def timed_out(start_time: float) -> bool:
    return method_timeout_sec > 0 and (time.monotonic() - start_time) > method_timeout_sec


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_compare(original: Path, generated: Path, dst: Path, title: str) -> bool:
    if Image is None:
        return False
    try:
        with Image.open(original) as left_img, Image.open(generated) as right_img:
            left = left_img.convert("RGB")
            right = right_img.convert("RGB")
    except Exception:
        return False

    size = (384, 384)
    left.thumbnail(size, Image.Resampling.BICUBIC)
    right.thumbnail(size, Image.Resampling.BICUBIC)

    label_h = 34
    pad = 12
    w = size[0] * 2 + pad * 3
    h = size[1] + label_h + pad * 2
    canvas = Image.new("RGB", (w, h), "white")

    lx = pad + (size[0] - left.width) // 2
    rx = pad * 2 + size[0] + (size[0] - right.width) // 2
    y = label_h + pad + (size[1] - max(left.height, right.height)) // 2
    canvas.paste(left, (lx, y))
    canvas.paste(right, (rx, y))

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), f"{title} | original", fill=(0, 0, 0))
    draw.text((pad * 2 + size[0], 8), "generated", fill=(0, 0, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return True


def export_pair(method: str, original: Path, generated: Path, index: int, rel_hint: str = "") -> bool:
    if not original.exists() or not generated.exists():
        return False
    method_root = out_root / method
    stem = generated.stem
    if stem.endswith("_fake_B"):
        stem = stem[:-7]
    safe_name = f"{index:03d}_{stem}"
    if rel_hint:
        safe_name = f"{index:03d}_{Path(rel_hint).stem}"

    original_dst = method_root / "original" / f"{safe_name}{original.suffix.lower()}"
    generated_dst = method_root / "generated" / f"{safe_name}{generated.suffix.lower()}"
    compare_dst = method_root / "compare" / f"{safe_name}_compare.jpg"

    copy_image(original, original_dst)
    copy_image(generated, generated_dst)
    make_compare(original_dst, generated_dst, compare_dst, method)
    return True


def export_manifest_pairs(method: str, manifest: Path, generated_root: Path) -> dict:
    written = 0
    missing = 0
    start_time = time.monotonic()
    if not manifest.is_file() or not generated_root.is_dir():
        return {
            "method": method,
            "status": "skipped",
            "reason": "manifest or generated root missing",
            "manifest": str(manifest),
            "generated_root": str(generated_root),
        }

    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated_index = build_generated_index(generated_root)
    status = "ok"
    reason = ""
    for rec in tqdm(records, desc=f"export {method}", unit="pair"):
        if written >= max_per_method:
            break
        if timed_out(start_time):
            status = "timeout"
            reason = f"method exceeded {method_timeout_sec:g}s; kept partial results"
            break
        original = Path(rec.get("source") or rec.get("destination") or "")
        if not original.exists() and rec.get("destination"):
            original = Path(rec["destination"])
        stem = rec.get("flat_stem") or Path(rec.get("flat_name", "")).stem
        synset = rec.get("synset", "")
        original_name = Path(rec.get("original_name", stem)).stem

        generated = None
        if synset:
            for cand in (generated_root / synset).glob(f"{original_name}.*"):
                if is_image(cand):
                    generated = cand
                    break
        if generated is None:
            generated = generated_index.get(stem) or generated_index.get(original_name)

        if generated is None or not original.exists():
            missing += 1
            continue
        if export_pair(method, original, generated, written, rec.get("relative", "")):
            written += 1

    return {
        "method": method,
        "status": status,
        "reason": reason,
        "written": written,
        "missing": missing,
        "manifest": str(manifest),
        "generated_root": str(generated_root),
    }


def export_tree_pairs(method: str, source_roots: list[Path], generated_roots: list[Path]) -> dict:
    written = 0
    missing = 0
    used_generated_root = None
    generated_images = []
    start_time = time.monotonic()
    status = "skipped"
    reason = ""
    for generated_root in generated_roots:
        if timed_out(start_time):
            status = "timeout"
            reason = f"method exceeded {method_timeout_sec:g}s before finding generated images"
            break
        generated_images = image_files(generated_root)
        if not generated_images:
            continue
        used_generated_root = generated_root
        status = "ok"
        source_stem_index = build_stem_index(source_roots)
        for generated in tqdm(generated_images, desc=f"export {method}", unit="pair"):
            if written >= max_per_method:
                break
            if timed_out(start_time):
                status = "timeout"
                reason = f"method exceeded {method_timeout_sec:g}s; kept partial results"
                break
            rel_original = None
            for source_root in source_roots:
                if not source_root.is_dir():
                    continue
                try:
                    rel = generated.relative_to(generated_root)
                except ValueError:
                    rel = None
                original = None
                if rel is not None:
                    candidate = (source_root / rel).with_suffix(generated.suffix)
                    if candidate.exists():
                        original = candidate
                    else:
                        original = source_stem_index.get(generated.stem)
                else:
                    original = source_stem_index.get(generated.stem)
                if original is not None and original.exists():
                    rel_original = original
                    break
            if rel_original is None:
                missing += 1
                continue
            if export_pair(method, rel_original, generated, written):
                written += 1
        break

    return {
        "method": method,
        "status": status,
        "reason": reason,
        "written": written,
        "missing": missing,
        "generated_root": str(used_generated_root) if used_generated_root else "",
        "source_roots": [str(p) for p in source_roots],
    }


def export_sd_single() -> dict:
    method = "stable_diffusion_vae_text"
    generated_root = work_root / "stable_diffusion_diffusers/vae_text_smoke"
    generated_images = image_files(generated_root)
    if not generated_images:
        return {
            "method": method,
            "status": "skipped",
            "reason": "no generated images",
            "generated_root": str(generated_root),
        }
    source_roots = [
        syn_root / "uwdf/source/train",
        work_root / "sources/stable_diffusion_img2img/train",
    ]
    written = 0
    missing = 0
    start_time = time.monotonic()
    source_index = build_stem_index(source_roots)
    status = "ok"
    reason = ""
    for generated in generated_images:
        if written >= max_per_method:
            break
        if timed_out(start_time):
            status = "timeout"
            reason = f"method exceeded {method_timeout_sec:g}s; kept partial results"
            break
        stem = generated.stem.split("_underwater_")[0]
        original = source_index.get(stem)
        if original is None:
            missing += 1
            continue
        if export_pair(method, original, generated, written):
            written += 1
    return {
        "method": method,
        "status": status,
        "reason": reason,
        "written": written,
        "missing": missing,
        "generated_root": str(generated_root),
    }


summaries = []

# CUT tiny validation output.
summaries.append(export_manifest_pairs(
    "cut",
    work_root / "cut/datasets/imagenet_ruod_cut_tiny_from_full_ssd/manifests/testA_manifest.jsonl",
    work_root / "cut/generated_tiny_from_full/val",
))

# SyreaNet enhancement route.
summaries.append(export_tree_pairs(
    "syreanet",
    [
        syn_root / "syreanet/source/train",
        work_root / "sources/syreanet/train",
    ],
    [
        syn_root / "syreanet/generated/train",
        work_root / "syreanet/generated/train",
        work_root / "syreanet/generated/val",
    ],
))

# SyreaNet physical synthesis route.
summaries.append(export_tree_pairs(
    "syreanet_synthesis",
    [
        syn_root / "syreanet_synthesis/source/train",
        work_root / "sources/syreanet_synthesis/train",
    ],
    [
        syn_root / "syreanet_synthesis/generated/train",
        work_root / "syreanet_synthesis/generated/train",
        work_root / "syreanet_synthesis/generated/val",
    ],
))

# UWNR route.
summaries.append(export_tree_pairs(
    "uwnr",
    [
        syn_root / "uwnr/source/train",
        work_root / "sources/uwnr/train",
    ],
    [
        syn_root / "uwnr_ruod_ref/generated/train",
        syn_root / "uwnr_ruod_ref/generated_smoke/train",
        syn_root / "uwnr_ruod_ref/generated_smoke",
        work_root / "uwnr/generated/train",
    ],
))

# Stable Diffusion VAE latent + text route.
summaries.append(export_sd_single())

summary_path = out_root / "visual_pairs_summary.json"
summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summaries, indent=2, ensure_ascii=False))
print(f"summary: {summary_path}")
PY

echo
echo "Step 1b/3: Collect generation logs"
LOG_EXPORT_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_EXPORT_DIR}"

copy_logs() {
  local method="$1"
  shift
  local dst="${LOG_EXPORT_DIR}/${method}"
  mkdir -p "${dst}"
  local copied=0
  for item in "$@"; do
    if [[ -f "${item}" ]]; then
      cp -a "${item}" "${dst}/"
      copied=$((copied + 1))
    elif [[ -d "${item}" ]]; then
      find "${item}" -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.json' \) -exec cp -a {} "${dst}/" \;
    fi
  done
  echo "  ${method}: copied logs to ${dst}"
}

copy_logs "cut" \
  "${REPO_ROOT}/logs/synthesis_smoke/cut_full_tiny_check" \
  "${REPO_ROOT}/logs/cut_full_tiny_check_launcher.log"

copy_logs "watergan" \
  "${REPO_ROOT}/logs/synthesis_smoke/watergan_tiny_check" \
  "${REPO_ROOT}/logs/watergan_tiny_check_launcher.log" \
  "${REPO_ROOT}/logs/watergan_tiny_reuse_check.log" \
  "${REPO_ROOT}/logs/watergan_train_tiny.log"

copy_logs "syreanet" \
  "${REPO_ROOT}/logs/synthesis_smoke/syreanet" \
  "${REPO_ROOT}/logs/syreanet_train_smoke.log"

copy_logs "syreanet_synthesis" \
  "${REPO_ROOT}/logs/synthesis_smoke/syreanet_synthesis" \
  "${REPO_ROOT}/logs/syreanet_synthesis_smoke.log"

copy_logs "uwnr" \
  "${REPO_ROOT}/logs/synthesis_smoke/uwnr" \
  "${REPO_ROOT}/logs/uwnr_ruod_ref_smoke.log"

copy_logs "stable_diffusion_vae_text" \
  "${REPO_ROOT}/logs/sd_vae_text_single_check.log"

OUT_ROOT="${OUT_ROOT}" python - <<'PY'
import json
import os
import re
from datetime import datetime
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
log_root = out_root / "logs"

time_patterns = [
    re.compile(r"Time Taken:\s*([0-9.]+)\s*sec", re.I),
    re.compile(r"Elapsed time:\s*([0-9hms:.]+)", re.I),
    re.compile(r"Total time:\s*([0-9:]+)\s*\(([0-9.]+)\s*s\s*/\s*it\)", re.I),
]

items = []
for method_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
    for log_path in sorted(method_dir.glob("*")):
        if not log_path.is_file():
            continue
        stat = log_path.stat()
        text = ""
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
        hits = []
        for pattern in time_patterns:
            for match in pattern.finditer(text):
                hits.append(match.group(0))
        items.append({
            "method": method_dir.name,
            "log": str(log_path),
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "time_mentions": hits[:20],
        })

summary_path = log_root / "log_time_summary.json"
summary_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

txt_path = log_root / "log_time_summary.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(f"[{item['method']}] {Path(item['log']).name}\n")
        f.write(f"  size_bytes: {item['size_bytes']}\n")
        f.write(f"  mtime: {item['mtime']}\n")
        if item["time_mentions"]:
            for hit in item["time_mentions"]:
                f.write(f"  time: {hit}\n")
        else:
            f.write("  time: no explicit duration found; use log timestamps/mtime or progress lines\n")
        f.write("\n")

print(f"log summary: {summary_path}")
print(f"log summary txt: {txt_path}")
PY

echo
echo "Step 2/3: Create archive"
rm -f "${ARCHIVE_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

echo
echo "Step 3/3: Upload"
if [[ "${UPLOAD}" == "1" ]]; then
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
