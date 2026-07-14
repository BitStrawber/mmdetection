#!/usr/bin/env bash
set -euo pipefail

# Build a compact review pack:
#   - 10 random ImageNet source images
#   - 10 random real underwater images
#   - for each synthetic method, 10 source/generated pairs plus side-by-side panels
#
# Defaults match the current exp_2 server layout. Override any path through env vars.

NUM="${NUM:-10}"
SEED="${SEED:-20260714}"

EXPORT_ROOT="${EXPORT_ROOT:-/media/HDD1/XCX/exp_2/method_random10_review_pack}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/media/HDD1/XCX/exp_2/method_random10_review_pack.tar.gz}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/method_random10_review_pack}"

IMAGENET_ROOT="${IMAGENET_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/cut/source/train}"
RUOD_ROOT="${RUOD_ROOT:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"

UWNR_SOURCE_ROOT="${UWNR_SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/cut/source/train}"
UWNR_GEN_ROOT="${UWNR_GEN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/uwnr_ruod_ref/generated/train}"

CUT_SOURCE_ROOT="${CUT_SOURCE_ROOT:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export/source}"
CUT_GEN_ROOT="${CUT_GEN_ROOT:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export/generated/imagenet_ruod_cut_full_bs2_5epoch_gpu5}"

WATERGAN_RESULT_ROOT="${WATERGAN_RESULT_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/results/imagenet_ruod_watergan_train_balanced50_ssd_gpu4}"

SYREANET_SOURCE_ROOT="${SYREANET_SOURCE_ROOT:-$IMAGENET_ROOT}"
SYREANET_GEN_ROOT="${SYREANET_GEN_ROOT:-}"

UWDF_SOURCE_ROOT="${UWDF_SOURCE_ROOT:-$IMAGENET_ROOT}"
UWDF_GEN_ROOT="${UWDF_GEN_ROOT:-}"

UPLOAD="${UPLOAD:-1}"
PACKAGE_EXPORT="${PACKAGE_EXPORT:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

section() {
  echo "========================================="
  echo "$1"
  echo "========================================="
}

count_images_recursive() {
  local root="$1"
  if [ -e "$root" ]; then
    find -L "$root" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) 2>/dev/null | wc -l
  else
    echo 0
  fi
}

count_images_flat_prefix() {
  local root="$1"
  local prefix="$2"
  if [ -e "$root" ]; then
    find -L "$root" -maxdepth 1 -type f -name "${prefix}*.png" 2>/dev/null | wc -l
  else
    echo 0
  fi
}

show_path() {
  local label="$1"
  local path="$2"
  echo
  echo "[$label]"
  echo "path: $path"
  if [ -e "$path" ]; then
    echo "exists: yes"
    echo -n "images: "
    count_images_recursive "$path"
    echo "first 5:"
    find -L "$path" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) 2>/dev/null | sort | head -n 5 || true
  else
    echo "exists: no"
  fi
}

section "Method random10 review pack config"
echo "NUM:                  $NUM"
echo "SEED:                 $SEED"
echo "EXPORT_ROOT:          $EXPORT_ROOT"
echo "ARCHIVE_PATH:         $ARCHIVE_PATH"
echo "RCLONE_DEST:          $RCLONE_DEST"
echo "UPLOAD:               $UPLOAD"
echo "PACKAGE_EXPORT:       $PACKAGE_EXPORT"
echo "CHECK_ONLY:           $CHECK_ONLY"
echo "RESET_OUTPUTS:        $RESET_OUTPUTS"

section "Input path check"
show_path "imagenet" "$IMAGENET_ROOT"
show_path "real underwater" "$RUOD_ROOT"
show_path "uwnr source" "$UWNR_SOURCE_ROOT"
show_path "uwnr generated" "$UWNR_GEN_ROOT"
show_path "cut 5epoch source" "$CUT_SOURCE_ROOT"
show_path "cut 5epoch generated" "$CUT_GEN_ROOT"
show_path "watergan result root" "$WATERGAN_RESULT_ROOT"
if [ -n "$SYREANET_GEN_ROOT" ]; then
  show_path "syreanet source" "$SYREANET_SOURCE_ROOT"
  show_path "syreanet generated" "$SYREANET_GEN_ROOT"
fi
if [ -n "$UWDF_GEN_ROOT" ]; then
  show_path "uwdf source" "$UWDF_SOURCE_ROOT"
  show_path "uwdf generated" "$UWDF_GEN_ROOT"
fi

echo
echo "[watergan paired files]"
echo -n "air images:  "
count_images_flat_prefix "$WATERGAN_RESULT_ROOT" "air_"
echo -n "fake images: "
count_images_flat_prefix "$WATERGAN_RESULT_ROOT" "fake_"

echo
echo "[rclone remotes]"
rclone listremotes || true

if [ "$CHECK_ONLY" = "1" ]; then
  section "Check only done"
  exit 0
fi

if [ "$RESET_OUTPUTS" = "1" ]; then
  case "$EXPORT_ROOT" in
    /media/HDD1/XCX/exp_2/*|/media/SSD1/XCX/exp_2/*)
      echo
      echo "Reset export root: $EXPORT_ROOT"
      rm -rf "$EXPORT_ROOT"
      ;;
    *)
      echo "Refuse to reset unexpected EXPORT_ROOT: $EXPORT_ROOT" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$EXPORT_ROOT"
export NUM SEED EXPORT_ROOT
export IMAGENET_ROOT RUOD_ROOT
export UWNR_SOURCE_ROOT UWNR_GEN_ROOT
export CUT_SOURCE_ROOT CUT_GEN_ROOT
export WATERGAN_RESULT_ROOT
export SYREANET_SOURCE_ROOT SYREANET_GEN_ROOT
export UWDF_SOURCE_ROOT UWDF_GEN_ROOT

section "Select, pair, and export"
python - <<'PY'
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

seed = int(os.environ.get("SEED", "20260714"))
num = int(os.environ.get("NUM", "10"))
random.seed(seed)

export_root = Path(os.environ["EXPORT_ROOT"])
manifest_root = export_root / "manifests"
manifest_root.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    )

def list_images_flat(root, prefix=None):
    root = Path(root)
    if not root.exists():
        return []
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if prefix:
        files = [p for p in files if p.name.startswith(prefix)]
    return files

def safe_name(path):
    path = Path(path)
    s = str(path).replace("\\", "/")
    parts = s.split("/")
    base = "_".join(parts[-3:]) if len(parts) >= 3 else path.name
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base)

def copy_random_images(name, root, out_subdir):
    out_dir = export_root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(root)
    if len(images) < num:
        print("[WARN] {}: only {} images found under {}, need {}".format(name, len(images), root, num))

    chosen = random.sample(images, min(num, len(images)))

    manifest = manifest_root / "{}.jsonl".format(name)
    with manifest.open("w", encoding="utf-8") as f:
        for i, src in enumerate(chosen):
            dst = out_dir / "{:03d}_{}".format(i, safe_name(src))
            shutil.copy2(str(src), str(dst))
            f.write(json.dumps({
                "index": i,
                "name": name,
                "src": str(src),
                "dst": str(dst),
            }, ensure_ascii=False) + "\n")

    print("[OK] {}: copied {} images -> {}".format(name, len(chosen), out_dir))

def resize_canvas(img, size=360):
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

def make_pair(src_path, gen_path, out_path, tile=360, label_h=36):
    src_img = resize_canvas(Image.open(src_path), tile)
    gen_img = resize_canvas(Image.open(gen_path), tile)

    panel = Image.new("RGB", (tile * 2, tile + label_h), (255, 255, 255))
    panel.paste(src_img, (0, label_h))
    panel.paste(gen_img, (tile, label_h))

    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, tile * 2, label_h], fill=(32, 32, 32))
    draw.text((12, 10), "source", fill=(255, 255, 255))
    draw.text((tile + 12, 10), "generated", fill=(255, 255, 255))

    panel.save(str(out_path), quality=95)

def export_pairs(method, pairs):
    if len(pairs) < num:
        print("[WARN] {}: only {} matched pairs, need {}".format(method, len(pairs), num))

    chosen = random.sample(pairs, min(num, len(pairs)))

    method_root = export_root / "synthetic" / method
    source_out = method_root / "source"
    gen_out = method_root / "generated"
    pair_out = method_root / "pair"

    for d in [source_out, gen_out, pair_out]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = manifest_root / "synthetic_{}.jsonl".format(method)
    with manifest.open("w", encoding="utf-8") as f:
        for i, item in enumerate(chosen):
            src, gen = item
            src_dst = source_out / "{:03d}_{}".format(i, safe_name(src))
            gen_dst = gen_out / "{:03d}_{}".format(i, safe_name(gen))
            pair_dst = pair_out / "{:03d}_{}_source_generated.jpg".format(i, method)

            shutil.copy2(str(src), str(src_dst))
            shutil.copy2(str(gen), str(gen_dst))
            make_pair(src, gen, pair_dst)

            f.write(json.dumps({
                "index": i,
                "method": method,
                "source": str(src),
                "generated": str(gen),
                "source_export": str(src_dst),
                "generated_export": str(gen_dst),
                "pair_export": str(pair_dst),
            }, ensure_ascii=False) + "\n")

    print("[OK] {}: exported {} pairs -> {}".format(method, len(chosen), method_root))

def match_by_relative(source_root, gen_root):
    source_root = Path(source_root)
    gen_root = Path(gen_root)

    src_map = {}
    for src in list_images(source_root):
        rel = src.relative_to(source_root).with_suffix("")
        src_map[str(rel).replace("\\", "/")] = src

    pairs = []
    for gen in list_images(gen_root):
        rel = gen.relative_to(gen_root).with_suffix("")
        key = str(rel).replace("\\", "/")
        src = src_map.get(key)
        if src:
            pairs.append((src, gen))

    return pairs

def match_cut_random20(source_root, gen_root):
    source_images = list_images(source_root)
    gen_images = list_images(gen_root)

    src_map = {}
    for src in source_images:
        key = src.stem.split("_", 1)[0]
        src_map[key] = src

    pairs = []
    for gen in gen_images:
        key = gen.stem.split("_", 1)[0]
        if key in src_map:
            pairs.append((src_map[key], gen))

    return pairs

def match_generic(source_root, gen_root):
    pairs = match_by_relative(source_root, gen_root)
    if pairs:
        return pairs
    pairs = match_cut_random20(source_root, gen_root)
    if pairs:
        return pairs
    source_images = list_images(source_root)
    gen_images = list_images(gen_root)
    return list(zip(source_images[:len(gen_images)], gen_images))

def match_watergan(result_root):
    result_root = Path(result_root)

    air_map = {}
    for air in list_images_flat(result_root, prefix="air_"):
        key = air.stem[len("air_"):]
        air_map[key] = air

    pairs = []
    for fake in list_images_flat(result_root, prefix="fake_"):
        key = fake.stem[len("fake_"):]
        air = air_map.get(key)
        if air:
            pairs.append((air, fake))

    return pairs

copy_random_images("imagenet", os.environ["IMAGENET_ROOT"], "imagenet/images")
copy_random_images("real_underwater", os.environ["RUOD_ROOT"], "real_underwater/images")

uwnr_pairs = match_by_relative(os.environ["UWNR_SOURCE_ROOT"], os.environ["UWNR_GEN_ROOT"])
print("[INFO] uwnr matched pairs: {}".format(len(uwnr_pairs)))
export_pairs("uwnr", uwnr_pairs)

cut_pairs = match_cut_random20(os.environ["CUT_SOURCE_ROOT"], os.environ["CUT_GEN_ROOT"])
print("[INFO] cut_5epoch matched pairs: {}".format(len(cut_pairs)))
export_pairs("cut_5epoch", cut_pairs)

watergan_pairs = match_watergan(os.environ["WATERGAN_RESULT_ROOT"])
print("[INFO] watergan matched pairs: {}".format(len(watergan_pairs)))
export_pairs("watergan", watergan_pairs)

synthetic_methods = ["uwnr", "cut_5epoch", "watergan"]

optional_methods = [
    ("syreanet", os.environ.get("SYREANET_SOURCE_ROOT", os.environ["IMAGENET_ROOT"]), os.environ.get("SYREANET_GEN_ROOT", "")),
    ("uwdf", os.environ.get("UWDF_SOURCE_ROOT", os.environ["IMAGENET_ROOT"]), os.environ.get("UWDF_GEN_ROOT", "")),
]
for method, source_root, gen_root in optional_methods:
    if gen_root:
        pairs = match_generic(source_root, gen_root)
        print("[INFO] {} matched pairs: {}".format(method, len(pairs)))
        export_pairs(method, pairs)
        synthetic_methods.append(method)
    else:
        print("[INFO] {} skipped: set {}_GEN_ROOT to include it".format(method, method.upper()))

summary = {
    "seed": seed,
    "num": num,
    "export_root": str(export_root),
    "standalone": ["imagenet", "real_underwater"],
    "synthetic_methods": synthetic_methods,
}
(export_root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("[OK] summary: {}".format(export_root / "summary.json"))
PY

section "Exported count"
echo -n "all exported images: "
find "$EXPORT_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l

echo
echo "synthetic pair counts:"
for d in "$EXPORT_ROOT"/synthetic/*/pair; do
  [ -d "$d" ] || continue
  echo "$(basename "$(dirname "$d")"): $(find "$d" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
done

echo
echo "manifest files:"
find "$EXPORT_ROOT/manifests" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' | sort

if [ "$PACKAGE_EXPORT" = "1" ]; then
  section "Create archive"
  tar -C "$(dirname "$EXPORT_ROOT")" -czf "$ARCHIVE_PATH" "$(basename "$EXPORT_ROOT")"
  ls -lh "$ARCHIVE_PATH"
fi

if [ "$UPLOAD" = "1" ]; then
  section "Upload"
  if [ "$PACKAGE_EXPORT" = "1" ] && [ -f "$ARCHIVE_PATH" ]; then
    rclone copy -P "$ARCHIVE_PATH" "$RCLONE_DEST"
  fi
  rclone copy -P "$EXPORT_ROOT" "$RCLONE_DEST/$(basename "$EXPORT_ROOT")/"
fi

section "Done"
echo "Local export: $EXPORT_ROOT"
echo "Archive:      $ARCHIVE_PATH"
echo "Remote:       $RCLONE_DEST"
