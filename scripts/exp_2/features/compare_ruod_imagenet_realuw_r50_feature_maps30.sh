#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

IMAGENET_CONFIG="${IMAGENET_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_ruod.py}"
REALUW_CONFIG="${REALUW_CONFIG:-configs/exp_2/cascade-rcnn_r50_dino-realuw-ssd_fpn_2x_ruod_j7.py}"
IMAGENET_WORK_DIR="${IMAGENET_WORK_DIR:-work_dirs/tri_pretrain/official_dino_rn50_100e_cascade}"
REALUW_WORK_DIR="${REALUW_WORK_DIR:-work_dirs/tri_pretrain/realuw_dino_rn50_ssd100e_cascade}"
IMAGENET_CHECKPOINT="${IMAGENET_CHECKPOINT:-}"
REALUW_CHECKPOINT="${REALUW_CHECKPOINT:-}"

DEVICE="${DEVICE:-cuda:0}"
SAMPLES="${SAMPLES:-30}"
SEED="${SEED:-2026}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.30}"
MAX_BOXES="${MAX_BOXES:-20}"
TILE_WIDTH="${TILE_WIDTH:-640}"
TILE_HEIGHT="${TILE_HEIGHT:-480}"
LABEL_HEIGHT="${LABEL_HEIGHT:-42}"
LOW_PERCENTILE="${LOW_PERCENTILE:-1.0}"
HIGH_PERCENTILE="${HIGH_PERCENTILE:-99.0}"
OVERWRITE="${OVERWRITE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/media/HDD2/XCX/exp_2/ruod_r50_feature_compare30_${STAMP}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUT_ROOT}/archives}"
UPLOAD="${UPLOAD:-0}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/feature_maps/ruod_r50_pretrain_comparison30}"

find_best_checkpoint() {
    local work_dir="$1"
    local checkpoint=""

    checkpoint=$(
        find "${work_dir}" \
            -maxdepth 1 \
            -type f \
            -name 'best_coco_bbox_mAP*.pth' \
            -printf '%T@|%p\n' \
            2>/dev/null |
        sort -t '|' -k1,1nr |
        head -n 1 |
        cut -d '|' -f 2-
    )
    if [[ -z "${checkpoint}" && -s "${work_dir}/latest.pth" ]]; then
        checkpoint="${work_dir}/latest.pth"
    fi
    printf '%s\n' "${checkpoint}"
}

resolve_ruod_root() {
    local candidate=""
    if [[ -n "${RUOD_ROOT:-}" ]]; then
        printf '%s\n' "${RUOD_ROOT}"
        return
    fi
    for candidate in \
        /media/SSD1/XCX/exp_2/RUOD/coco \
        /media/HDD0/XCX/exp_2/RUOD/coco \
        /media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco
    do
        if [[ -d "${candidate}/val" && \
              -s "${candidate}/annotations/instances_val.json" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
}

require_file() {
    [[ -s "$1" ]] || {
        echo "Error: required file is missing or empty: $1" >&2
        exit 1
    }
}

if [[ -z "${IMAGENET_CHECKPOINT}" ]]; then
    IMAGENET_CHECKPOINT="$(find_best_checkpoint "${IMAGENET_WORK_DIR}")"
fi
if [[ -z "${REALUW_CHECKPOINT}" ]]; then
    REALUW_CHECKPOINT="$(find_best_checkpoint "${REALUW_WORK_DIR}")"
fi

RUOD_ROOT="$(resolve_ruod_root)"
RUOD_IMAGE_DIR="${RUOD_IMAGE_DIR:-${RUOD_ROOT}/val}"
RUOD_ANNOTATION="${RUOD_ANNOTATION:-${RUOD_ROOT}/annotations/instances_val.json}"

echo "============================================================"
echo "RUOD R50 detector feature-map comparison (same 30 images)"
echo "============================================================"
echo "ImageNet config:      ${IMAGENET_CONFIG}"
echo "ImageNet checkpoint:  ${IMAGENET_CHECKPOINT}"
echo "RealUW config:        ${REALUW_CONFIG}"
echo "RealUW checkpoint:    ${REALUW_CHECKPOINT}"
echo "RUOD image dir:       ${RUOD_IMAGE_DIR}"
echo "RUOD annotation:      ${RUOD_ANNOTATION}"
echo "Samples / seed:       ${SAMPLES} / ${SEED}"
echo "Device:               ${DEVICE}"
echo "Output:               ${OUT_ROOT}"
echo "Normalization:        shared pairwise p${LOW_PERCENTILE}-p${HIGH_PERCENTILE}"
echo "Prediction boxes:     score>=${SCORE_THRESHOLD}, max=${MAX_BOXES}"
echo "Upload:               ${UPLOAD}"
echo "Google Drive:         ${RCLONE_DEST}"
echo "============================================================"

require_file "${IMAGENET_CONFIG}"
require_file "${REALUW_CONFIG}"
require_file "${IMAGENET_CHECKPOINT}"
require_file "${REALUW_CHECKPOINT}"
require_file "${RUOD_ANNOTATION}"
[[ -d "${RUOD_IMAGE_DIR}" ]] || {
    echo "Error: RUOD image directory not found: ${RUOD_IMAGE_DIR}" >&2
    exit 1
}

available_images=$(
    find "${RUOD_IMAGE_DIR}" \
        -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) |
    wc -l
)
echo "RUOD validation images found: ${available_images}"
[[ "${available_images}" -ge "${SAMPLES}" ]] || {
    echo "Error: not enough RUOD validation images" >&2
    exit 1
}

if [[ "${CHECK_ONLY}" == 1 ]]; then
    echo "CHECK_ONLY validation: OK"
    exit 0
fi

render_args=(
    --imagenet-config "${IMAGENET_CONFIG}"
    --imagenet-checkpoint "${IMAGENET_CHECKPOINT}"
    --realuw-config "${REALUW_CONFIG}"
    --realuw-checkpoint "${REALUW_CHECKPOINT}"
    --image-dir "${RUOD_IMAGE_DIR}"
    --annotation-file "${RUOD_ANNOTATION}"
    --out-dir "${OUT_ROOT}"
    --device "${DEVICE}"
    --samples "${SAMPLES}"
    --seed "${SEED}"
    --score-threshold "${SCORE_THRESHOLD}"
    --max-boxes "${MAX_BOXES}"
    --tile-width "${TILE_WIDTH}"
    --tile-height "${TILE_HEIGHT}"
    --label-height "${LABEL_HEIGHT}"
    --low-percentile "${LOW_PERCENTILE}"
    --high-percentile "${HIGH_PERCENTILE}"
)
[[ "${OVERWRITE}" == 1 ]] && render_args+=(--overwrite)

python tools/exp_2/compare_ruod_r50_detector_feature_maps.py \
    "${render_args[@]}"

echo
echo "============================================================"
echo "Validate output counts"
echo "============================================================"
for model in imagenet_r50_ruod realuw_r50_ruod; do
    for variant in no_box with_box; do
        root="${OUT_ROOT}/${model}/${variant}"
        originals=$(find "${root}/originals" -maxdepth 1 -type f -name '*.png' | wc -l)
        maps=$(find "${root}/feature_maps" -type f -name '*.png' | wc -l)
        panels=$(find "${root}/five_panels" -maxdepth 1 -type f -name '*.png' | wc -l)
        printf '%-22s %-8s originals=%d maps=%d panels=%d\n' \
            "${model}" "${variant}" "${originals}" "${maps}" "${panels}"
        [[ "${originals}" -eq "${SAMPLES}" && \
           "${maps}" -eq "$((SAMPLES * 4))" && \
           "${panels}" -eq "${SAMPLES}" ]] || {
            echo "Error: invalid output count for ${model}/${variant}" >&2
            exit 1
        }
    done
done
for variant in no_box with_box; do
    panels=$(
        find "${OUT_ROOT}/comparison_panels/${variant}" \
            -maxdepth 1 -type f -name '*.png' |
        wc -l
    )
    printf 'comparison_panels      %-8s panels=%d\n' "${variant}" "${panels}"
    [[ "${panels}" -eq "${SAMPLES}" ]] || {
        echo "Error: invalid comparison panel count" >&2
        exit 1
    }
done

mkdir -p "${ARCHIVE_DIR}"
export OUT_ROOT ARCHIVE_DIR
python - <<'PY'
import hashlib
import os
import zipfile
from pathlib import Path

root = Path(os.environ['OUT_ROOT']).resolve()
archive_dir = Path(os.environ['ARCHIVE_DIR']).resolve()
groups = {
    'imagenet_r50_ruod_feature_maps30.zip': [
        root / 'imagenet_r50_ruod',
    ],
    'realuw_r50_ruod_feature_maps30.zip': [
        root / 'realuw_r50_ruod',
    ],
    'ruod_r50_two_model_comparison30.zip': [
        root / 'comparison_panels',
        root / 'selected_originals',
        root / 'manifest.tsv',
        root / 'shared_normalization.tsv',
        root / 'summary.json',
        root / 'README.txt',
    ],
}
checksums = []
for archive_name, sources in groups.items():
    archive_path = archive_dir / archive_name
    with zipfile.ZipFile(
            str(archive_path), 'w', compression=zipfile.ZIP_DEFLATED,
            compresslevel=6) as archive:
        for source in sources:
            paths = source.rglob('*') if source.is_dir() else [source]
            for path in sorted(paths):
                if path.is_file():
                    archive.write(str(path), str(path.relative_to(root)))
    with zipfile.ZipFile(str(archive_path), 'r') as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(
                'ZIP integrity failure in {}: {}'.format(archive_path, bad))
    digest = hashlib.sha256()
    with archive_path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    checksums.append('{}  {}'.format(digest.hexdigest(), archive_name))
    print('{}: {} bytes'.format(archive_path, archive_path.stat().st_size))

(archive_dir / 'SHA256SUMS.txt').write_text(
    '\n'.join(checksums) + '\n', encoding='utf-8')
PY

ls -lh "${ARCHIVE_DIR}"/*.zip "${ARCHIVE_DIR}/SHA256SUMS.txt"

if [[ "${UPLOAD}" == 1 ]]; then
    command -v rclone >/dev/null 2>&1 || {
        echo "Error: rclone is not installed" >&2
        exit 1
    }
    rclone listremotes | grep -Fxq 'fcp:' || {
        echo "Error: rclone remote fcp: is not configured" >&2
        exit 1
    }
    rclone copy \
        --progress \
        --transfers 3 \
        --checkers 8 \
        "${ARCHIVE_DIR}" \
        "${RCLONE_DEST}"
    echo "Google Drive contents:"
    rclone lsl "${RCLONE_DEST}" | tail -n 20
fi

echo
echo "============================================================"
echo "RUOD R50 feature-map comparison completed"
echo "============================================================"
echo "Output:   ${OUT_ROOT}"
echo "Archives: ${ARCHIVE_DIR}"
if [[ "${UPLOAD}" == 1 ]]; then
    echo "Remote:   ${RCLONE_DEST}"
fi
