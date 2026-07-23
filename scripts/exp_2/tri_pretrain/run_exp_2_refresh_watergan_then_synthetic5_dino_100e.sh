#!/bin/bash
set -Eeuo pipefail

# Refresh the corrected WaterGAN split from Hugging Face, rebuild the merged
# five-method ImageFolder with hard links, then train DINO R50 -> ViT-S 100e.
#
# The archive is downloaded before the installed dataset is touched. The new
# WaterGAN tree and merged ImageFolder are validated before atomic replacement.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

HF_REPO_ID="${HF_REPO_ID:-BitStrawber/transfer}"
HF_REVISION="${HF_REVISION:-main}"
HF_ARCHIVE_PATH="${HF_ARCHIVE_PATH:-archives/watergan_train_val.tar}"
HF_CHECKSUM_PATH="${HF_CHECKSUM_PATH:-archives/SHA256SUMS.txt}"
EXPECTED_WATERGAN_SHA256="${EXPECTED_WATERGAN_SHA256:-92b1258ed3efbd51d5078fdc99e6ea15022512ac044c96cdd19f094fcfc16b79}"

SYNTH_ROOT="${SYNTH_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$SYNTH_ROOT/.hf_transfer_watergan}"
STAGE_ROOT="${STAGE_ROOT:-$SYNTH_ROOT/.watergan_refresh_stage}"
WATERGAN_ROOT="${WATERGAN_ROOT:-$SYNTH_ROOT/watergan}"
WATERGAN_GENERATED="$WATERGAN_ROOT/generated"
MERGED_ROOT="${MERGED_ROOT:-$SYNTH_ROOT/merged_5methods}"

UWN_ROOT="${UWN_ROOT:-$SYNTH_ROOT/uwnr_ruod_ref/generated}"
SYREANET_ROOT="${SYREANET_ROOT:-$SYNTH_ROOT/syreanet_synthesis/generated}"
CUT_ROOT="${CUT_ROOT:-$SYNTH_ROOT/cut/generated}"
UWDF_ROOT="${UWDF_ROOT:-$SYNTH_ROOT/uwdf/generated}"

EXPECTED_METHOD_TRAIN="${EXPECTED_METHOD_TRAIN:-250000}"
EXPECTED_METHOD_VAL="${EXPECTED_METHOD_VAL:-10000}"
EXPECTED_MERGED_TRAIN="${EXPECTED_MERGED_TRAIN:-1250000}"
EXPECTED_MERGED_VAL="${EXPECTED_MERGED_VAL:-50000}"
EXPECTED_CLASSES="${EXPECTED_CLASSES:-1000}"

DOWNLOAD_WATERGAN="${DOWNLOAD_WATERGAN:-1}"
VERIFY_ARCHIVE="${VERIFY_ARCHIVE:-1}"
REFRESH_WATERGAN="${REFRESH_WATERGAN:-1}"
REBUILD_MERGED="${REBUILD_MERGED:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-1}"
KEEP_BACKUPS="${KEEP_BACKUPS:-0}"
LINK_WORKERS="${LINK_WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
MIN_FREE_GB="${MIN_FREE_GB:-260}"

PRETRAIN_RUNNER="${PRETRAIN_RUNNER:-scripts/exp_2/tri_pretrain/run_exp_2_synthetic5_merged_dino_r50_then_vits_100e.sh}"
R50_NAME="${R50_NAME:-synthetic5_fixedwatergan_dino_resnet50_100e}"
VITS_NAME="${VITS_NAME:-synthetic5_fixedwatergan_dino_vits_100e}"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_DIR/synthetic5_fixedwatergan_refresh_and_dino100e.log}"

mkdir -p "$DOWNLOAD_ROOT" "$LOG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

die() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

assert_safe_child() {
    local path="$1"
    case "$path" in
        "$SYNTH_ROOT"/*) ;;
        *) die "Refuse to modify path outside SYNTH_ROOT: $path" ;;
    esac
}

check_free_space() {
    local available_kb required_kb
    available_kb=$(df -Pk "$SYNTH_ROOT" | awk 'NR==2 {print $4}')
    required_kb=$((MIN_FREE_GB * 1024 * 1024))
    echo "available_space_gb=$((available_kb / 1024 / 1024)) required_space_gb=$MIN_FREE_GB"
    if [ "$available_kb" -lt "$required_kb" ]; then
        die "Insufficient free space under $SYNTH_ROOT"
    fi
}

validate_split() {
    local label="$1"
    local root="$2"
    local expected_images="$3"

    python - "$label" "$root" "$expected_images" "$EXPECTED_CLASSES" <<'PY'
import os
import sys
from pathlib import Path

label, root, expected_images, expected_classes = sys.argv[1:]
root = Path(root)
expected_images = int(expected_images)
expected_classes = int(expected_classes)
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

if not root.is_dir():
    raise SystemExit(f'{label}: missing directory: {root}')

class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
images = 0
zero_size = 0
empty_classes = []
for class_dir in class_dirs:
    class_images = 0
    with os.scandir(class_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            if Path(entry.name).suffix.lower() not in extensions:
                continue
            images += 1
            class_images += 1
            if entry.stat(follow_symlinks=True).st_size == 0:
                zero_size += 1
    if class_images == 0:
        empty_classes.append(class_dir.name)

print(
    f'{label}: root={root} classes={len(class_dirs)} images={images} '
    f'zero_size={zero_size} empty_classes={len(empty_classes)}')
if len(class_dirs) != expected_classes:
    raise SystemExit(
        f'{label}: expected {expected_classes} classes, found {len(class_dirs)}')
if images != expected_images:
    raise SystemExit(
        f'{label}: expected {expected_images} images, found {images}')
if zero_size:
    raise SystemExit(f'{label}: found {zero_size} zero-size files')
if empty_classes:
    raise SystemExit(f'{label}: empty classes: {empty_classes[:10]}')
PY
}

find_generated_root() {
    local extract_root="$1"
    local candidate
    for candidate in \
        "$extract_root/generated" \
        "$extract_root/watergan/generated" \
        "$extract_root/synthetic_imagenet/watergan/generated" \
        "$extract_root"; do
        if [ -d "$candidate/train" ] && [ -d "$candidate/val" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    while IFS= read -r candidate; do
        if [ -d "$candidate/val" ]; then
            printf '%s\n' "$(dirname "$candidate")"
            return 0
        fi
    done < <(find "$extract_root" -type d -name train -print)
    return 1
}

download_watergan() {
    command -v hf >/dev/null 2>&1 || die "hf CLI not found"
    hf auth whoami >/dev/null 2>&1 || die "Hugging Face login required: hf auth login"

    echo "[$(timestamp)] Download corrected WaterGAN archive"
    echo "repo=$HF_REPO_ID revision=$HF_REVISION"
    echo "archive=$HF_ARCHIVE_PATH"
    HF_HOME="${HF_HOME:-$SYNTH_ROOT/.huggingface_cache}" \
    HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    hf download \
        "$HF_REPO_ID" \
        "$HF_ARCHIVE_PATH" \
        "$HF_CHECKSUM_PATH" \
        --repo-type dataset \
        --revision "$HF_REVISION" \
        --local-dir "$DOWNLOAD_ROOT"
}

verify_archive() {
    local archive="$DOWNLOAD_ROOT/$HF_ARCHIVE_PATH"
    [ -s "$archive" ] || die "Downloaded archive missing or empty: $archive"

    echo "[$(timestamp)] Verify archive SHA-256"
    local actual
    actual=$(sha256sum "$archive" | awk '{print $1}')
    echo "expected_sha256=$EXPECTED_WATERGAN_SHA256"
    echo "actual_sha256=$actual"
    [ "$actual" = "$EXPECTED_WATERGAN_SHA256" ] \
        || die "WaterGAN archive checksum mismatch"

    echo "[$(timestamp)] Verify tar structure"
    tar -tf "$archive" >/dev/null
}

refresh_watergan() {
    local archive="$DOWNLOAD_ROOT/$HF_ARCHIVE_PATH"
    local extract_root="$STAGE_ROOT/extracted"
    local ready_root="$STAGE_ROOT/generated.ready"
    local backup="$WATERGAN_ROOT/generated.backup.$(date '+%Y%m%d_%H%M%S')"

    assert_safe_child "$STAGE_ROOT"
    assert_safe_child "$WATERGAN_ROOT"
    rm -rf -- "$STAGE_ROOT"
    mkdir -p "$extract_root" "$WATERGAN_ROOT"

    echo "[$(timestamp)] Extract corrected WaterGAN archive"
    if command -v pv >/dev/null 2>&1; then
        pv "$archive" | tar -xf - -C "$extract_root"
    else
        tar -xf "$archive" -C "$extract_root"
    fi

    local discovered
    discovered=$(find_generated_root "$extract_root") \
        || die "Could not find train/val below extracted archive"
    echo "discovered_generated_root=$discovered"

    validate_split watergan.train "$discovered/train" "$EXPECTED_METHOD_TRAIN"
    validate_split watergan.val "$discovered/val" "$EXPECTED_METHOD_VAL"

    if [ "$discovered" = "$extract_root" ]; then
        mv "$extract_root" "$ready_root"
    else
        mv "$discovered" "$ready_root"
    fi

    if [ -e "$WATERGAN_GENERATED" ]; then
        mv "$WATERGAN_GENERATED" "$backup"
        echo "old_watergan_backup=$backup"
    fi
    mv "$ready_root" "$WATERGAN_GENERATED"

    validate_split installed.watergan.train "$WATERGAN_GENERATED/train" "$EXPECTED_METHOD_TRAIN"
    validate_split installed.watergan.val "$WATERGAN_GENERATED/val" "$EXPECTED_METHOD_VAL"

    if [ -d "$backup" ] && [ "$KEEP_BACKUPS" != "1" ]; then
        rm -rf -- "$backup"
        echo "removed_old_watergan_backup=$backup"
    fi
    rm -rf -- "$STAGE_ROOT"
}

validate_all_methods() {
    validate_split uwnr.train "$UWN_ROOT/train" "$EXPECTED_METHOD_TRAIN"
    validate_split uwnr.val "$UWN_ROOT/val" "$EXPECTED_METHOD_VAL"
    validate_split syreanet.train "$SYREANET_ROOT/train" "$EXPECTED_METHOD_TRAIN"
    validate_split syreanet.val "$SYREANET_ROOT/val" "$EXPECTED_METHOD_VAL"
    validate_split cut.train "$CUT_ROOT/train" "$EXPECTED_METHOD_TRAIN"
    validate_split cut.val "$CUT_ROOT/val" "$EXPECTED_METHOD_VAL"
    validate_split uwdf.train "$UWDF_ROOT/train" "$EXPECTED_METHOD_TRAIN"
    validate_split uwdf.val "$UWDF_ROOT/val" "$EXPECTED_METHOD_VAL"
    validate_split watergan.train "$WATERGAN_GENERATED/train" "$EXPECTED_METHOD_TRAIN"
    validate_split watergan.val "$WATERGAN_GENERATED/val" "$EXPECTED_METHOD_VAL"
}

rebuild_merged() {
    local next_root="$SYNTH_ROOT/.merged_5methods.next.$$"
    local backup="$SYNTH_ROOT/.merged_5methods.backup.$(date '+%Y%m%d_%H%M%S')"

    assert_safe_child "$next_root"
    assert_safe_child "$MERGED_ROOT"
    mkdir -p "$next_root/imagefolder"

    echo "[$(timestamp)] Build merged five-method ImageFolder with hard links"
    python - \
        "$next_root/imagefolder" \
        "$LINK_WORKERS" \
        "$PROGRESS_EVERY" \
        "uwnr=$UWN_ROOT" \
        "syreanet=$SYREANET_ROOT" \
        "cut=$CUT_ROOT" \
        "uwdf=$UWDF_ROOT" \
        "watergan=$WATERGAN_GENERATED" <<'PY'
import concurrent.futures
import os
import sys
import threading
from pathlib import Path

out_root = Path(sys.argv[1])
workers = int(sys.argv[2])
progress_every = int(sys.argv[3])
methods = []
for spec in sys.argv[4:]:
    name, path = spec.split('=', 1)
    methods.append((name, Path(path)))

extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
counter = 0
counter_lock = threading.Lock()

def link_one(item):
    global counter
    method, split, source = item
    class_name = source.parent.name
    target_dir = out_root / split / class_name
    target = target_dir / f'{method}__{source.name}'
    try:
        os.link(source, target)
    except FileExistsError:
        source_stat = source.stat()
        target_stat = target.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
                target_stat.st_dev, target_stat.st_ino):
            raise
    with counter_lock:
        counter += 1
        if progress_every > 0 and counter % progress_every == 0:
            print(f'[progress] linked={counter}', flush=True)

for split in ('train', 'val'):
    classes = set()
    for _, method_root in methods:
        split_root = method_root / split
        classes.update(p.name for p in split_root.iterdir() if p.is_dir())
    for class_name in sorted(classes):
        (out_root / split / class_name).mkdir(parents=True, exist_ok=True)

def iter_items():
    for method, method_root in methods:
        for split in ('train', 'val'):
            split_root = method_root / split
            for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
                with os.scandir(class_dir) as entries:
                    for entry in entries:
                        if not entry.is_file(follow_symlinks=True):
                            continue
                        source = Path(entry.path)
                        if source.suffix.lower() in extensions:
                            yield method, split, source

with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    for _ in executor.map(link_one, iter_items(), chunksize=256):
        pass

print(f'[complete] linked={counter}', flush=True)
PY

    validate_split merged.train "$next_root/imagefolder/train" "$EXPECTED_MERGED_TRAIN"
    validate_split merged.val "$next_root/imagefolder/val" "$EXPECTED_MERGED_VAL"

    if [ -e "$MERGED_ROOT" ]; then
        mv "$MERGED_ROOT" "$backup"
        echo "old_merged_backup=$backup"
    fi
    mv "$next_root" "$MERGED_ROOT"

    validate_split installed.merged.train "$MERGED_ROOT/imagefolder/train" "$EXPECTED_MERGED_TRAIN"
    validate_split installed.merged.val "$MERGED_ROOT/imagefolder/val" "$EXPECTED_MERGED_VAL"

    if [ -d "$backup" ] && [ "$KEEP_BACKUPS" != "1" ]; then
        rm -rf -- "$backup"
        echo "removed_old_merged_backup=$backup"
    fi
}

run_pretraining() {
    [ -f "$PRETRAIN_RUNNER" ] || die "Pretraining runner not found: $PRETRAIN_RUNNER"
    echo "[$(timestamp)] Start DINO ResNet-50 100e -> ViT-S 100e"
    env \
        DATA_ROOT="$MERGED_ROOT" \
        R50_NAME="$R50_NAME" \
        VITS_NAME="$VITS_NAME" \
        EXPECTED_TRAIN_IMAGES="$EXPECTED_MERGED_TRAIN" \
        EXPECTED_VAL_IMAGES="$EXPECTED_MERGED_VAL" \
        EXPECTED_CLASSES="$EXPECTED_CLASSES" \
        bash "$PRETRAIN_RUNNER"
}

echo "================================================================"
echo "Corrected WaterGAN refresh and DINO pretraining"
echo "HF repo:             $HF_REPO_ID"
echo "HF archive:          $HF_ARCHIVE_PATH"
echo "Synthetic root:      $SYNTH_ROOT"
echo "WaterGAN generated:  $WATERGAN_GENERATED"
echo "Merged ImageFolder:  $MERGED_ROOT/imagefolder"
echo "R50 output:          work_dirs/tri_pretrain/$R50_NAME"
echo "ViT-S output:        work_dirs/tri_pretrain/$VITS_NAME"
echo "Pipeline log:        $PIPELINE_LOG"
echo "================================================================"

check_free_space

if [ "$DOWNLOAD_WATERGAN" = "1" ]; then
    download_watergan
fi
if [ "$VERIFY_ARCHIVE" = "1" ]; then
    verify_archive
fi
if [ "$REFRESH_WATERGAN" = "1" ]; then
    refresh_watergan
fi

validate_all_methods

if [ "$REBUILD_MERGED" = "1" ]; then
    rebuild_merged
else
    validate_split existing.merged.train "$MERGED_ROOT/imagefolder/train" "$EXPECTED_MERGED_TRAIN"
    validate_split existing.merged.val "$MERGED_ROOT/imagefolder/val" "$EXPECTED_MERGED_VAL"
fi

if [ "$VALIDATE_ONLY" = "1" ]; then
    echo "[$(timestamp)] VALIDATE_ONLY=1, stop before pretraining"
    exit 0
fi

if [ "$RUN_PRETRAIN" = "1" ]; then
    run_pretraining
else
    echo "[$(timestamp)] RUN_PRETRAIN=0, data refresh completed"
fi

if [ "$KEEP_ARCHIVE" != "1" ]; then
    archive="$DOWNLOAD_ROOT/$HF_ARCHIVE_PATH"
    assert_safe_child "$archive"
    rm -f -- "$archive"
    echo "removed_archive=$archive"
fi

echo "[$(timestamp)] Pipeline completed"
