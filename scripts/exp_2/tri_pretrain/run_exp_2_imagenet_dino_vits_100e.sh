#!/bin/bash
set -euo pipefail

# Materialize ImageNet-1K train class tar files to SSD ImageFolder format,
# then run the same facebookresearch/DINO ViT-Small/16 100e recipe used by
# J14 RealUW pretraining.
#
# Expected source layout:
#   /media/HDD0/XCX/IMAGENET/n01440764.tar
#   /media/HDD0/XCX/IMAGENET/n01443537.tar
#   ...
#
# Output layout:
#   /media/SSD1/XCX/exp_2/IMAGENET1K/imagefolder/train/n01440764/*.JPEG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

IMAGENET_TAR_ROOT="${IMAGENET_TAR_ROOT:-/media/HDD0/XCX/IMAGENET}"
IMAGENET_SSL_ROOT="${IMAGENET_SSL_ROOT:-/media/SSD1/XCX/exp_2/IMAGENET1K}"
IMAGENET_TRAIN_DIR="$IMAGENET_SSL_ROOT/imagefolder/train"
LOG_DIR="${LOG_DIR:-logs/tri_pretrain}"

EXTRACT_IMAGENET="${EXTRACT_IMAGENET:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
SKIP_EXTRACTED="${SKIP_EXTRACTED:-1}"
VERIFY_IMAGENET="${VERIFY_IMAGENET:-1}"
EXPECTED_CLASSES="${EXPECTED_CLASSES:-1000}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-1281167}"

mkdir -p "$LOG_DIR" "$IMAGENET_TRAIN_DIR"

extract_imagenet() {
    python -u - <<'PY'
from pathlib import Path
import json
import os
import tarfile
import time

src_root = Path(os.environ.get('IMAGENET_TAR_ROOT', '/media/HDD0/XCX/IMAGENET'))
out_root = Path(os.environ['IMAGENET_TRAIN_DIR'])
log_dir = Path(os.environ.get('LOG_DIR', 'logs/tri_pretrain'))
skip_extracted = os.environ.get('SKIP_EXTRACTED', '1') == '1'
expected_classes = int(os.environ.get('EXPECTED_CLASSES', '1000'))

summary_path = log_dir / 'imagenet1k_extract_to_ssd_summary.json'
log_dir.mkdir(parents=True, exist_ok=True)
out_root.mkdir(parents=True, exist_ok=True)

tar_paths = sorted(src_root.glob('n*.tar'))
print('src_root:', src_root)
print('out_root:', out_root)
print('tar_files:', len(tar_paths))
print('skip_extracted:', skip_extracted)
if len(tar_paths) != expected_classes:
    print(f'WARNING: expected {expected_classes} class tar files, got {len(tar_paths)}')

def count_images(path: Path) -> int:
    return sum(
        1 for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )

start = time.time()
results = []
errors = []

for index, tar_path in enumerate(tar_paths, start=1):
    cls_name = tar_path.stem
    cls_dir = out_root / cls_name
    done_flag = cls_dir / '.extract_done'

    if skip_extracted and done_flag.exists():
        img_count = count_images(cls_dir)
        result = {
            'class': cls_name,
            'status': 'skipped_exists',
            'images': img_count,
            'error': '',
        }
        results.append(result)
        print(f'[{index:04d}/{len(tar_paths):04d}] skip {cls_name}: {img_count} images')
        continue

    cls_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, 'r') as tf:
            members = []
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name:
                    continue
                member.name = name
                members.append(member)
            tf.extractall(cls_dir, members=members)

        img_count = count_images(cls_dir)
        done_flag.write_text(str(img_count) + '\n')
        result = {
            'class': cls_name,
            'status': 'ok',
            'images': img_count,
            'error': '',
        }
        print(f'[{index:04d}/{len(tar_paths):04d}] extract {cls_name}: {img_count} images')
    except Exception as exc:
        result = {
            'class': cls_name,
            'status': 'error',
            'images': 0,
            'error': f'{type(exc).__name__}: {exc}',
        }
        errors.append(result)
        print(f'[{index:04d}/{len(tar_paths):04d}] ERROR {cls_name}: {result["error"]}')
    results.append(result)

summary = {
    'src_root': str(src_root),
    'out_root': str(out_root),
    'tar_files': len(tar_paths),
    'ok': sum(1 for item in results if item['status'] == 'ok'),
    'skipped_exists': sum(1 for item in results if item['status'] == 'skipped_exists'),
    'errors': len(errors),
    'total_images': sum(item['images'] for item in results),
    'elapsed_sec': round(time.time() - start, 2),
    'error_samples': errors[:20],
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('summary:', summary_path)

if errors:
    raise SystemExit('ImageNet extraction has errors; stop before pretraining.')
PY
}

verify_imagenet() {
    local class_count
    local image_count
    class_count=$(find "$IMAGENET_TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    image_count=$(find "$IMAGENET_TRAIN_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

    echo "ImageNet train class dirs: $class_count"
    echo "ImageNet train images: $image_count"
    du -sh "$IMAGENET_SSL_ROOT" || true

    if [ "$class_count" -ne "$EXPECTED_CLASSES" ]; then
        echo "Error: expected $EXPECTED_CLASSES class dirs, got $class_count"
        exit 1
    fi
    if [ "$image_count" -ne "$EXPECTED_IMAGES" ]; then
        echo "Warning: expected $EXPECTED_IMAGES images, got $image_count"
        echo "Continue because some ImageNet mirrors can differ slightly."
    fi
}

if [ "$EXTRACT_IMAGENET" = "1" ]; then
    export IMAGENET_TAR_ROOT IMAGENET_TRAIN_DIR LOG_DIR SKIP_EXTRACTED EXPECTED_CLASSES
    extract_imagenet 2>&1 | tee "$LOG_DIR/imagenet1k_extract_to_ssd.log"
else
    echo "EXTRACT_IMAGENET=$EXTRACT_IMAGENET, skip ImageNet extraction."
fi

if [ "$VERIFY_IMAGENET" = "1" ]; then
    verify_imagenet 2>&1 | tee "$LOG_DIR/imagenet1k_verify.log"
fi

if [ "$RUN_PRETRAIN" != "1" ]; then
    echo "RUN_PRETRAIN=$RUN_PRETRAIN, stop after ImageNet materialization."
    exit 0
fi

export EXP_ID="${EXP_ID:-j14}"
export TASK_CONFIG="${TASK_CONFIG:-configs/exp_2/tri_pretrain/s1_imagenet_dino_vits_100e.sh}"
export REALUW_SSL_ROOT="$IMAGENET_SSL_ROOT"
export BUILD_REALUW_SSL=0
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-29692}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
export GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-3000}"
export GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
export GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
export GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"

bash "$SCRIPT_DIR/run_exp_2_tri_pretrain_s1.sh"
