#!/usr/bin/env bash
set -euo pipefail

# Randomly sample 10 images from each synthesis method, ImageNet, and easy RUOD,
# then upload the compact sample folder to a Hugging Face dataset repository.
#
# Default upload target:
#   https://huggingface.co/datasets/BitStrawber/DATA
#
# The script uploads a folder, not only an archive, so the samples can be browsed
# directly on Hugging Face.

NUM="${NUM:-10}"
SEED="${SEED:-20260714}"

EXPORT_ROOT="${EXPORT_ROOT:-/media/HDD1/XCX/exp_2/random10_methods_hf_pack}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/media/HDD1/XCX/exp_2/random10_methods_hf_pack.tar.gz}"
HF_REPO_ID="${HF_REPO_ID:-BitStrawber/DATA}"
HF_REMOTE_DIR="${HF_REMOTE_DIR:-exp_2/random10_methods_hf_pack}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Add exp_2 random10 synthesis/ImageNet/easy-RUOD samples}"

IMAGENET_ROOT="${IMAGENET_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/cut/source/train}"
EASY_RUOD_IMG_DIR="${EASY_RUOD_IMG_DIR:-/media/HDD0/XCX/exp_2/RUOD/coco/train}"
EASY_RUOD_ANN="${EASY_RUOD_ANN:-/media/HDD0/XCX/exp_2/RUOD/coco/annotations/easy_merged.json}"

UWNR_GEN_ROOT="${UWNR_GEN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/uwnr_ruod_ref/generated/train}"
CUT_GEN_ROOT="${CUT_GEN_ROOT:-/media/HDD1/XCX/exp_2/cut_four_weights_random20_export/generated/imagenet_ruod_cut_full_bs2_5epoch_gpu5}"
WATERGAN_RESULT_ROOT="${WATERGAN_RESULT_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/watergan/results/imagenet_ruod_watergan_train_balanced50_ssd_gpu4}"
SYREANET_GEN_ROOT="${SYREANET_GEN_ROOT:-/media/HDD1/XCX/exp_2/synthetic_imagenet/syreanet_synthesis/generated/train}"
UWDF_GEN_ROOT="${UWDF_GEN_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_controlnet_ipadapter/train}"

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

section "Random10 methods Hugging Face export config"
echo "NUM:               $NUM"
echo "SEED:              $SEED"
echo "EXPORT_ROOT:       $EXPORT_ROOT"
echo "ARCHIVE_PATH:      $ARCHIVE_PATH"
echo "HF_REPO_ID:        $HF_REPO_ID"
echo "HF_REMOTE_DIR:     $HF_REMOTE_DIR"
echo "UPLOAD:            $UPLOAD"
echo "PACKAGE_EXPORT:    $PACKAGE_EXPORT"
echo "CHECK_ONLY:        $CHECK_ONLY"
echo "RESET_OUTPUTS:     $RESET_OUTPUTS"

section "Input path check"
show_path "imagenet" "$IMAGENET_ROOT"
show_path "easy_ruod image dir" "$EASY_RUOD_IMG_DIR"
echo "easy_ruod annotation: $EASY_RUOD_ANN"
if [ -f "$EASY_RUOD_ANN" ]; then
  echo "easy_ruod annotation exists: yes"
else
  echo "easy_ruod annotation exists: no"
fi
show_path "uwnr generated" "$UWNR_GEN_ROOT"
show_path "cut 5epoch generated" "$CUT_GEN_ROOT"
show_path "watergan result root" "$WATERGAN_RESULT_ROOT"
show_path "syreanet generated" "$SYREANET_GEN_ROOT"
show_path "uwdf generated" "$UWDF_GEN_ROOT"

echo
if command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli: $(command -v huggingface-cli)"
else
  echo "huggingface-cli: NOT FOUND"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  section "Check only done"
  exit 0
fi

if [ "$RESET_OUTPUTS" = "1" ]; then
  case "$EXPORT_ROOT" in
    /media/HDD1/XCX/exp_2/*|/media/SSD1/XCX/exp_2/*)
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
export IMAGENET_ROOT EASY_RUOD_IMG_DIR EASY_RUOD_ANN
export UWNR_GEN_ROOT CUT_GEN_ROOT WATERGAN_RESULT_ROOT SYREANET_GEN_ROOT UWDF_GEN_ROOT
export HF_REPO_ID HF_REMOTE_DIR

section "Sample and export"
python - <<'PY'
import json
import os
import random
import shutil
from pathlib import Path

seed = int(os.environ.get("SEED", "20260714"))
num = int(os.environ.get("NUM", "10"))
random.seed(seed)

export_root = Path(os.environ["EXPORT_ROOT"])
manifest_dir = export_root / "manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in image_exts)


def safe_name(path):
    path = Path(path)
    s = str(path).replace("\\", "/")
    parts = s.split("/")
    base = "_".join(parts[-3:]) if len(parts) >= 3 else path.name
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base)


def copy_sample(name, images, out_subdir):
    out_dir = export_root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(images) < num:
        print("[WARN] {}: only {} images available, need {}".format(name, len(images), num))
    chosen = random.sample(images, min(num, len(images)))
    manifest = manifest_dir / "{}.jsonl".format(name)
    with manifest.open("w", encoding="utf-8") as f:
        for i, src in enumerate(chosen):
            dst = out_dir / "{:03d}_{}".format(i, safe_name(src))
            shutil.copy2(str(src), str(dst))
            f.write(json.dumps({
                "index": i,
                "group": name,
                "source_path": str(src),
                "export_path": str(dst),
            }, ensure_ascii=False) + "\n")
    print("[OK] {}: exported {} images -> {}".format(name, len(chosen), out_dir))
    return len(chosen)


def list_easy_ruod(img_dir, ann_path):
    img_dir = Path(img_dir)
    ann_path = Path(ann_path)
    if not ann_path.is_file():
        print("[WARN] easy_ruod annotation missing: {}; fallback to all images under {}".format(ann_path, img_dir))
        return list_images(img_dir)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = []
    for rec in coco.get("images", []):
        file_name = rec.get("file_name", "")
        if not file_name:
            continue
        candidates = [img_dir / file_name, img_dir / Path(file_name).name]
        for cand in candidates:
            if cand.is_file():
                images.append(cand)
                break
    images = sorted(set(images))
    return images


def list_watergan_fake(result_root):
    result_root = Path(result_root)
    if not result_root.exists():
        return []
    return sorted(
        p for p in result_root.iterdir()
        if p.is_file() and p.name.startswith("fake_") and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

summary = {
    "seed": seed,
    "num": num,
    "hf_repo_id": os.environ.get("HF_REPO_ID", ""),
    "hf_remote_dir": os.environ.get("HF_REMOTE_DIR", ""),
    "groups": {},
}

standalone = [
    ("imagenet", list_images(os.environ["IMAGENET_ROOT"]), "imagenet"),
    ("easy_ruod", list_easy_ruod(os.environ["EASY_RUOD_IMG_DIR"], os.environ["EASY_RUOD_ANN"]), "easy_ruod"),
]

synthetic = [
    ("uwnr", list_images(os.environ["UWNR_GEN_ROOT"]), "synthetic/uwnr"),
    ("cut_5epoch", list_images(os.environ["CUT_GEN_ROOT"]), "synthetic/cut_5epoch"),
    ("watergan", list_watergan_fake(os.environ["WATERGAN_RESULT_ROOT"]), "synthetic/watergan"),
    ("syreanet", list_images(os.environ["SYREANET_GEN_ROOT"]), "synthetic/syreanet"),
    ("uwdf", list_images(os.environ["UWDF_GEN_ROOT"]), "synthetic/uwdf"),
]

for name, images, subdir in standalone + synthetic:
    summary["groups"][name] = {
        "available": len(images),
        "exported": copy_sample(name, images, subdir),
    }

readme = export_root / "README.md"
readme.write_text("""# Exp 2 Random10 Visual Samples

This folder contains randomly selected visual samples for quick inspection.

- `imagenet/`: 10 ImageNet source images.
- `easy_ruod/`: 10 images sampled from RUOD `easy_merged.json`.
- `synthetic/uwnr/`: 10 UWNR generated images.
- `synthetic/cut_5epoch/`: 10 CUT 5epoch generated images.
- `synthetic/watergan/`: 10 WaterGAN generated images (`fake_*.png`).
- `synthetic/syreanet/`: 10 SyreaNet generated images.
- `synthetic/uwdf/`: 10 UWDF generated images.

Selection is deterministic for the configured seed. See `summary.json` and `manifests/*.jsonl` for original paths.
""", encoding="utf-8")

(export_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("[OK] summary: {}".format(export_root / "summary.json"))
PY

section "Exported count"
echo -n "all exported images: "
find "$EXPORT_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l

echo
find "$EXPORT_ROOT" -maxdepth 3 -type d | sort

echo
if [ -d "$EXPORT_ROOT/manifests" ]; then
  find "$EXPORT_ROOT/manifests" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' | sort
fi

if [ "$PACKAGE_EXPORT" = "1" ]; then
  section "Create archive"
  tar -C "$(dirname "$EXPORT_ROOT")" -czf "$ARCHIVE_PATH" "$(basename "$EXPORT_ROOT")"
  ls -lh "$ARCHIVE_PATH"
fi

if [ "$UPLOAD" = "1" ]; then
  section "Upload to Hugging Face"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Error: huggingface-cli not found. Install huggingface_hub or activate an env that has it." >&2
    exit 1
  fi
  huggingface-cli upload \
    --repo-type dataset \
    --commit-message "$HF_COMMIT_MESSAGE" \
    "$HF_REPO_ID" \
    "$EXPORT_ROOT" \
    "$HF_REMOTE_DIR"
fi

section "Done"
echo "Local export:      $EXPORT_ROOT"
echo "Archive:           $ARCHIVE_PATH"
echo "Hugging Face repo:  https://huggingface.co/datasets/$HF_REPO_ID"
echo "Remote directory:  $HF_REMOTE_DIR"