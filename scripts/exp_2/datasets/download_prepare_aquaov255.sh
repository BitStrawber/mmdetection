#!/usr/bin/env bash
# Download, extract, and apply the official AquaOV255 data corrections.
set -Eeuo pipefail

HF_REPO="${HF_REPO:-kkk2026/UOVSBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/HDD2/XCX/exp_2/AquaOV255}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$OUTPUT_ROOT/_downloads}"
DATA_ROOT="${DATA_ROOT:-$OUTPUT_ROOT/raw}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v hf >/dev/null 2>&1 || die "Hugging Face CLI 'hf' is required. Activate the hf_transfer environment first."
command -v unzip >/dev/null 2>&1 || die "unzip is required."
"$PYTHON_BIN" -c 'import numpy; from PIL import Image' >/dev/null 2>&1 || \
  die "PYTHON_BIN must provide numpy and Pillow; for example set PYTHON_BIN=/media/SSD1/conda_envs/detector/bin/python"

mkdir -p "$DOWNLOAD_ROOT" "$DATA_ROOT"

echo "============================================================"
echo "AquaOV255 download and official preparation"
echo "repository=$HF_REPO"
echo "download_root=$DOWNLOAD_ROOT"
echo "data_root=$DATA_ROOT"
echo "============================================================"

hf download "$HF_REPO" \
  --repo-type dataset \
  --include 'archives/AquaOV255-*.zip' \
  --local-dir "$DOWNLOAD_ROOT"

mapfile -t archives < <(find "$DOWNLOAD_ROOT/archives" -maxdepth 1 -type f -name 'AquaOV255-*.zip' -print | sort)
[ "${#archives[@]}" -eq 7 ] || die "expected 7 AquaOV255 archives, found ${#archives[@]} under $DOWNLOAD_ROOT/archives"

for archive in "${archives[@]}"; do
  echo "Extracting $(basename "$archive")"
  unzip -n -q "$archive" -d "$DATA_ROOT"
done

# Archives may contain one top-level AquaOV255 directory or directly contain
# images/ and masks/. Resolve the actual root without relying on archive layout.
if [ -d "$DATA_ROOT/images" ] && [ -d "$DATA_ROOT/masks" ]; then
  AQUA_ROOT="$DATA_ROOT"
else
  AQUA_ROOT="$(find "$DATA_ROOT" -type d -name images -print 2>/dev/null | while IFS= read -r image_dir; do
    candidate="$(dirname "$image_dir")"
    [ -d "$candidate/masks" ] && { printf '%s\n' "$candidate"; break; }
  done)"
fi

[ -n "${AQUA_ROOT:-}" ] && [ -d "$AQUA_ROOT/images" ] && [ -d "$AQUA_ROOT/masks" ] || die "could not locate AquaOV255 images/ and masks/ after extraction"

# Official Earth2Ocean repository corrections:
# remove the invalid Catfish pair and remap 254 -> 206 in three Lanternfish masks.
rm -f -- "$AQUA_ROOT/images/Catfish_112.png" "$AQUA_ROOT/masks/Catfish_112.jpg"

"$PYTHON_BIN" - "$AQUA_ROOT" <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
for name in ("Lanternfish_003.png", "Lanternfish_001.png", "Lanternfish_002.png"):
    path = root / "masks" / name
    if not path.is_file():
        raise SystemExit(f"missing official correction target: {path}")
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise SystemExit(f"expected a single-channel mask: {path}; shape={mask.shape}")
    fixed = mask.astype(np.uint16, copy=True)
    fixed[fixed == 254] = 206
    Image.fromarray(fixed, mode="I;16").save(path)
    print(f"corrected={path}")
PY

MANIFEST="$OUTPUT_ROOT/aquaov255_source_manifest.tsv"
"$PYTHON_BIN" - "$AQUA_ROOT" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
images = sorted(path for path in (root / "images").iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes)
masks = sorted(path for path in (root / "masks").iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes)
mask_by_stem = {path.stem: path for path in masks}

with manifest.open("w", encoding="utf-8") as handle:
    handle.write("image_name\timage_bytes\tmask_name\tmask_bytes\tpair_status\n")
    for image in images:
        mask = mask_by_stem.get(image.stem)
        handle.write(
            f"{image.name}\t{image.stat().st_size}\t{mask.name if mask else ''}\t"
            f"{mask.stat().st_size if mask else ''}\t{'PAIRED' if mask else 'MISSING_MASK'}\n"
        )

image_stems = {path.stem for path in images}
mask_stems = set(mask_by_stem)
print(f"aqua_root={root}")
print(f"images={len(images)}")
print(f"masks={len(masks)}")
print(f"paired_images={len(image_stems & mask_stems)}")
print(f"images_without_mask={len(image_stems - mask_stems)}")
print(f"masks_without_image={len(mask_stems - image_stems)}")
print(f"manifest={manifest}")
PY

printf 'field\tvalue\n' > "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'hf_repo\t%s\n' "$HF_REPO" >> "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'download_root\t%s\n' "$DOWNLOAD_ROOT" >> "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'data_root\t%s\n' "$DATA_ROOT" >> "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'aqua_root\t%s\n' "$AQUA_ROOT" >> "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'official_fix_removed_pair\tCatfish_112\n' >> "$OUTPUT_ROOT/preparation_provenance.tsv"
printf 'official_fix_mask_remap\tLanternfish_001,002,003: 254->206\n' >> "$OUTPUT_ROOT/preparation_provenance.tsv"

if [ "$KEEP_ARCHIVES" != 1 ]; then
  rm -f -- "${archives[@]}"
fi

echo "AQUA_ROOT=$AQUA_ROOT"
echo "RESULT=PASS_AQUAOV255_PREPARED"
