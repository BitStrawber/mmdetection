#!/usr/bin/env bash
set -euo pipefail

NUM="${NUM:-10}"
SEED="${SEED:-20260721}"

SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
SYN_ROOT="${SYN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"

UWNR_SOURCE_ROOT="${UWNR_SOURCE_ROOT:-$SOURCE_ROOT/uwnr/source/train}"
UWNR_GEN_ROOT="${UWNR_GEN_ROOT:-$SYN_ROOT/uwnr_ruod_ref/generated/train}"
SYREANET_SOURCE_ROOT="${SYREANET_SOURCE_ROOT:-$SOURCE_ROOT/syreanet/source/train}"
SYREANET_GEN_ROOT="${SYREANET_GEN_ROOT:-$SYN_ROOT/syreanet_synthesis/generated/train}"
CUT_SOURCE_ROOT="${CUT_SOURCE_ROOT:-$SOURCE_ROOT/cut/source/train}"
CUT_GEN_ROOT="${CUT_GEN_ROOT:-$SYN_ROOT/cut/generated/train}"
WATERGAN_SOURCE_ROOT="${WATERGAN_SOURCE_ROOT:-$SOURCE_ROOT/watergan/source/train}"
WATERGAN_GEN_ROOT="${WATERGAN_GEN_ROOT:-$SYN_ROOT/watergan/generated/train}"
UWDF_SOURCE_ROOT="${UWDF_SOURCE_ROOT:-$SOURCE_ROOT/uwdf/source/train}"
UWDF_GEN_ROOT="${UWDF_GEN_ROOT:-$WORK_ROOT/uwdf_controlnet_ipadapter/train}"

EXPORT_ROOT="${EXPORT_ROOT:-/media/HDD2/XCX/exp_2/synthesis_random10_manual_review}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${EXPORT_ROOT}.tar.gz}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/synthesis_random10_manual_review}"

MAKE_PANELS="${MAKE_PANELS:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"
PACKAGE_EXPORT="${PACKAGE_EXPORT:-1}"
UPLOAD="${UPLOAD:-1}"

section() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

count_images() {
  find -L "$1" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
       -o -iname '*.bmp' -o -iname '*.webp' \) \
    2>/dev/null | wc -l
}

show_pair_root() {
  local method="$1"
  local source="$2"
  local generated="$3"

  echo
  echo "[$method]"
  echo "source:    $source"
  echo "generated: $generated"

  [ -d "$source" ] || {
    echo "Error: source directory missing: $source" >&2
    return 1
  }
  [ -d "$generated" ] || {
    echo "Error: generated directory missing: $generated" >&2
    return 1
  }

  echo "source images:    $(count_images "$source")"
  echo "generated images: $(count_images "$generated")"
}

section "Full synthesis random review export"
echo "NUM:            $NUM"
echo "SEED:           $SEED"
echo "EXPORT_ROOT:    $EXPORT_ROOT"
echo "ARCHIVE_PATH:   $ARCHIVE_PATH"
echo "RCLONE_DEST:    $RCLONE_DEST"
echo "MAKE_PANELS:    $MAKE_PANELS"
echo "CHECK_ONLY:     $CHECK_ONLY"
echo "RESET_OUTPUTS:  $RESET_OUTPUTS"
echo "PACKAGE_EXPORT: $PACKAGE_EXPORT"
echo "UPLOAD:         $UPLOAD"

section "Input path and count check"
show_pair_root uwnr "$UWNR_SOURCE_ROOT" "$UWNR_GEN_ROOT"
show_pair_root syreanet "$SYREANET_SOURCE_ROOT" "$SYREANET_GEN_ROOT"
show_pair_root cut_5epoch "$CUT_SOURCE_ROOT" "$CUT_GEN_ROOT"
show_pair_root watergan "$WATERGAN_SOURCE_ROOT" "$WATERGAN_GEN_ROOT"
show_pair_root uwdf "$UWDF_SOURCE_ROOT" "$UWDF_GEN_ROOT"

echo
echo "rclone remotes:"
rclone listremotes || true

if [ "$CHECK_ONLY" = "1" ]; then
  section "Check only completed"
  exit 0
fi

if [ "$RESET_OUTPUTS" = "1" ] && [ -e "$EXPORT_ROOT" ]; then
  case "$EXPORT_ROOT" in
    /media/HDD2/XCX/exp_2/*|/media/HDD1/XCX/exp_2/*|/media/SSD1/XCX/exp_2/*)
      rm -rf -- "$EXPORT_ROOT"
      ;;
    *)
      echo "Error: refuse to reset unexpected path: $EXPORT_ROOT" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$EXPORT_ROOT"
export NUM SEED EXPORT_ROOT MAKE_PANELS
export UWNR_SOURCE_ROOT UWNR_GEN_ROOT
export SYREANET_SOURCE_ROOT SYREANET_GEN_ROOT
export CUT_SOURCE_ROOT CUT_GEN_ROOT
export WATERGAN_SOURCE_ROOT WATERGAN_GEN_ROOT
export UWDF_SOURCE_ROOT UWDF_GEN_ROOT

section "Select strict source/generated pairs"
python - <<'PY'
from __future__ import print_function

import json
import os
import random
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

NUM = int(os.environ["NUM"])
SEED = os.environ["SEED"]
EXPORT_ROOT = Path(os.environ["EXPORT_ROOT"])
MAKE_PANELS = os.environ.get("MAKE_PANELS", "1") == "1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

METHODS = [
    ("uwnr", Path(os.environ["UWNR_SOURCE_ROOT"]), Path(os.environ["UWNR_GEN_ROOT"])),
    ("syreanet", Path(os.environ["SYREANET_SOURCE_ROOT"]), Path(os.environ["SYREANET_GEN_ROOT"])),
    ("cut_5epoch", Path(os.environ["CUT_SOURCE_ROOT"]), Path(os.environ["CUT_GEN_ROOT"])),
    ("watergan", Path(os.environ["WATERGAN_SOURCE_ROOT"]), Path(os.environ["WATERGAN_GEN_ROOT"])),
    ("uwdf", Path(os.environ["UWDF_SOURCE_ROOT"]), Path(os.environ["UWDF_GEN_ROOT"])),
]

if MAKE_PANELS and Image is None:
    raise SystemExit("Error: Pillow is required when MAKE_PANELS=1")


def images(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def relative_key(path, root):
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/")


def build_source_indexes(root):
    by_relative = {}
    by_stem = {}
    duplicate_stems = set()

    for path in images(root):
        by_relative[relative_key(path, root)] = path
        stem = path.stem
        if stem in by_stem:
            duplicate_stems.add(stem)
        else:
            by_stem[stem] = path

    for stem in duplicate_stems:
        by_stem.pop(stem, None)
    return by_relative, by_stem


def generated_keys(path, root):
    parts = list(path.relative_to(root).with_suffix("").parts)
    keys = ["/".join(parts)]
    while parts and parts[0].lower() in {
        "generated", "generated_images", "images", "train", "output", "outputs"
    }:
        parts = parts[1:]
        if parts:
            keys.append("/".join(parts))
    return keys


def generated_stems(path):
    stems = [path.stem]
    if "_underwater_" in path.stem:
        stems.append(path.stem.split("_underwater_", 1)[0])
    return stems


def match_pairs(source_root, generated_root):
    by_relative, by_stem = build_source_indexes(source_root)
    pairs = []

    for generated in images(generated_root):
        source = None
        for key in generated_keys(generated, generated_root):
            source = by_relative.get(key)
            if source is not None:
                break
        if source is None:
            for stem in generated_stems(generated):
                source = by_stem.get(stem)
                if source is not None:
                    break
        if source is not None:
            pairs.append((source, generated))
    return pairs


def safe_name(path):
    return "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in path.name
    )


def fit(image, size=384):
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_panel(source, generated, destination):
    tile = 384
    header = 40
    left = fit(Image.open(str(source)), tile)
    right = fit(Image.open(str(generated)), tile)
    panel = Image.new("RGB", (tile * 2, tile + header), "white")
    panel.paste(left, (0, header))
    panel.paste(right, (tile, header))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, tile * 2, header), fill=(30, 30, 30))
    draw.text((12, 12), "original", fill="white")
    draw.text((tile + 12, 12), "generated", fill="white")
    panel.save(str(destination), quality=95)


summary = {"num": NUM, "seed": SEED, "methods": {}}
manifest_root = EXPORT_ROOT / "manifests"
manifest_root.mkdir(parents=True, exist_ok=True)

for method, source_root, generated_root in METHODS:
    print("[{}] index and match pairs".format(method), flush=True)
    pairs = match_pairs(source_root, generated_root)
    if len(pairs) < NUM:
        raise SystemExit(
            "Error: {} has only {} strict pairs; need {}".format(method, len(pairs), NUM)
        )

    rng = random.Random("{}:{}".format(SEED, method))
    selected = rng.sample(pairs, NUM)
    method_root = EXPORT_ROOT / method
    source_out = method_root / "original"
    generated_out = method_root / "generated"
    panel_out = method_root / "pair_panel"
    source_out.mkdir(parents=True, exist_ok=True)
    generated_out.mkdir(parents=True, exist_ok=True)
    if MAKE_PANELS:
        panel_out.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_root / "{}.jsonl".format(method)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for index, (source, generated) in enumerate(selected):
            prefix = "{:02d}_{}".format(index, source.stem)
            source_export = source_out / "{}{}".format(prefix, source.suffix.lower())
            generated_export = generated_out / "{}{}".format(prefix, generated.suffix.lower())
            shutil.copy2(str(source), str(source_export))
            shutil.copy2(str(generated), str(generated_export))

            panel_export = None
            if MAKE_PANELS:
                panel_export = panel_out / "{}_pair.jpg".format(prefix)
                make_panel(source, generated, panel_export)

            record = {
                "index": index,
                "method": method,
                "original": str(source),
                "generated": str(generated),
                "original_export": str(source_export),
                "generated_export": str(generated_export),
                "panel_export": str(panel_export) if panel_export else None,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["methods"][method] = {
        "matched_pairs": len(pairs),
        "selected_pairs": len(selected),
        "source_root": str(source_root),
        "generated_root": str(generated_root),
        "manifest": str(manifest_path),
    }
    print("[OK] {}: matched={}, selected={}".format(method, len(pairs), NUM))

(EXPORT_ROOT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

section "Validate export"
for method in uwnr syreanet cut_5epoch watergan uwdf; do
  original_count="$(count_images "$EXPORT_ROOT/$method/original")"
  generated_count="$(count_images "$EXPORT_ROOT/$method/generated")"
  panel_count=0
  if [ "$MAKE_PANELS" = "1" ]; then
    panel_count="$(count_images "$EXPORT_ROOT/$method/pair_panel")"
  fi
  printf '%-12s original=%d/%d generated=%d/%d panels=%d\n' \
    "$method" "$original_count" "$NUM" "$generated_count" "$NUM" "$panel_count"
  [ "$original_count" -eq "$NUM" ] && [ "$generated_count" -eq "$NUM" ] || {
    echo "Error: exported count validation failed for $method" >&2
    exit 1
  }
done

if [ "$PACKAGE_EXPORT" = "1" ]; then
  section "Create archive"
  tar -C "$(dirname "$EXPORT_ROOT")" -czf "$ARCHIVE_PATH" "$(basename "$EXPORT_ROOT")"
  sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
  ls -lh "$ARCHIVE_PATH" "${ARCHIVE_PATH}.sha256"
fi

if [ "$UPLOAD" = "1" ]; then
  section "Upload to Google Drive"
  rclone copy -P "$EXPORT_ROOT" "$RCLONE_DEST/files/"
  if [ "$PACKAGE_EXPORT" = "1" ]; then
    rclone copy -P "$ARCHIVE_PATH" "$RCLONE_DEST/"
    rclone copy -P "${ARCHIVE_PATH}.sha256" "$RCLONE_DEST/"
  fi
fi

section "Done"
echo "Local:   $EXPORT_ROOT"
echo "Archive: $ARCHIVE_PATH"
echo "Remote:  $RCLONE_DEST"
