#!/usr/bin/env bash
set -euo pipefail

# Build two review archives from the random10 feature-map extraction:
#   1. torchvision ImageNet-supervised ResNet-50
#   2. J2 RUOD-supervised Cascade R-CNN ResNet-50 backbone
#
# Each archive contains the 10 original images, 10 five-panel figures, and a
# subset manifest. Set UPLOAD=1 to copy both archives and SHA256SUMS.txt to the
# configured rclone destination.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-work_dirs/exp_2/feature_maps/torchvision_resnet50_imagenet_train_vs_j2_ruod_random10_fixed_preprocess}"
EXPORT_ROOT="${EXPORT_ROOT:-${SOURCE_ROOT}_five_panel_export}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${EXPORT_ROOT}/archives}"
RCLONE_DEST="${RCLONE_DEST:-fcp:exp_2/feature_maps/resnet50_random10_five_panels}"
UPLOAD="${UPLOAD:-0}"
TILE_SIZE="${TILE_SIZE:-320}"
LABEL_HEIGHT="${LABEL_HEIGHT:-36}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-10}"

SOURCE_ROOT="${SOURCE_ROOT%/}"
EXPORT_ROOT="${EXPORT_ROOT%/}"
ARCHIVE_DIR="${ARCHIVE_DIR%/}"
RCLONE_DEST="${RCLONE_DEST%/}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Error: SOURCE_ROOT does not exist: ${SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_ROOT}/manifest.tsv" ]]; then
  echo "Error: source manifest does not exist: ${SOURCE_ROOT}/manifest.tsv" >&2
  exit 1
fi
if [[ "${SOURCE_ROOT}" == "${EXPORT_ROOT}" ]]; then
  echo "Error: EXPORT_ROOT must differ from SOURCE_ROOT." >&2
  exit 1
fi
if [[ "${TILE_SIZE}" -le 0 || "${LABEL_HEIGHT}" -le 0 ]]; then
  echo "Error: TILE_SIZE and LABEL_HEIGHT must be positive integers." >&2
  exit 1
fi

cat <<EOF
========================================================================
Random10 ResNet-50 feature-map five-panel export
========================================================================
SOURCE_ROOT:      ${SOURCE_ROOT}
EXPORT_ROOT:      ${EXPORT_ROOT}
ARCHIVE_DIR:      ${ARCHIVE_DIR}
EXPECTED_SAMPLES: ${EXPECTED_SAMPLES}
TILE_SIZE:        ${TILE_SIZE}
UPLOAD:           ${UPLOAD}
RCLONE_DEST:      ${RCLONE_DEST}
========================================================================
EOF

SOURCE_ROOT="${SOURCE_ROOT}" \
EXPORT_ROOT="${EXPORT_ROOT}" \
TILE_SIZE="${TILE_SIZE}" \
LABEL_HEIGHT="${LABEL_HEIGHT}" \
EXPECTED_SAMPLES="${EXPECTED_SAMPLES}" \
python - <<'PY'
import csv
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

source_root = Path(os.environ["SOURCE_ROOT"]).resolve()
export_root = Path(os.environ["EXPORT_ROOT"]).resolve()
tile_size = int(os.environ["TILE_SIZE"])
label_height = int(os.environ["LABEL_HEIGHT"])
expected_samples = int(os.environ["EXPECTED_SAMPLES"])

if export_root == source_root or source_root in export_root.parents:
    raise SystemExit(
        "EXPORT_ROOT must not be SOURCE_ROOT or a parent of SOURCE_ROOT")
if export_root == Path(export_root.anchor):
    raise SystemExit("Refusing to use a filesystem root as EXPORT_ROOT")

subsets = {
    "imagenet": {
        "source_dir": source_root / "imagenet_torchvision_resnet50",
        "export_name": "imagenet_torchvision_resnet50",
        "heatmaps": [
            ("layer1_heatmap.png", "Layer 1"),
            ("layer2_heatmap.png", "Layer 2"),
            ("layer3_heatmap.png", "Layer 3"),
            ("layer4_heatmap.png", "Layer 4"),
        ],
    },
    "ruod": {
        "source_dir": source_root / "ruod_supervised_cascade_resnet50",
        "export_name": "ruod_supervised_cascade_resnet50_j2",
        "heatmaps": [
            ("backbone_stage1_heatmap.png", "Stage 1"),
            ("backbone_stage2_heatmap.png", "Stage 2"),
            ("backbone_stage3_heatmap.png", "Stage 3"),
            ("backbone_stage4_heatmap.png", "Stage 4"),
        ],
    },
}

with (source_root / "manifest.tsv").open(
        "r", encoding="utf-8", newline="") as handle:
    records = list(csv.DictReader(handle, delimiter="\t"))


def load_tile(path: Path, label: str) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    tile.paste(
        image,
        ((tile_size - image.width) // 2, (tile_size - image.height) // 2),
    )
    labeled = Image.new(
        "RGB", (tile_size, tile_size + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(labeled)
    draw.rectangle(
        (0, 0, tile_size, label_height), fill=(236, 236, 236))
    draw.text((10, 10), label, fill=(0, 0, 0))
    labeled.paste(tile, (0, label_height))
    return labeled


def join_horizontal(tiles):
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    offset = 0
    for tile in tiles:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    return panel


def find_input(sample_dir: Path) -> Path:
    candidates = sorted(
        path for path in sample_dir.glob("input.*") if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one input image in {sample_dir}, "
            f"found {len(candidates)}")
    return candidates[0]


summary_rows = []
for subset, spec in subsets.items():
    subset_records = [row for row in records if row["subset"] == subset]
    if len(subset_records) != expected_samples:
        raise RuntimeError(
            f"{subset}: expected {expected_samples} manifest records, "
            f"found {len(subset_records)}")

    source_dir = spec["source_dir"].resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    subset_root = export_root / spec["export_name"]
    if subset_root.exists():
        shutil.rmtree(subset_root)
    originals_dir = subset_root / "originals"
    panels_dir = subset_root / "five_panels"
    originals_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for row in sorted(subset_records, key=lambda item: int(item["index"])):
        index = int(row["index"])
        sample_dir = Path(row["output_dir"])
        if not sample_dir.is_absolute():
            sample_dir = (Path.cwd() / sample_dir).resolve()
        if sample_dir.parent != source_dir:
            raise RuntimeError(
                f"{subset}: sample directory is outside the expected source "
                f"directory: {sample_dir}")

        input_path = find_input(sample_dir)
        heatmap_paths = []
        for filename, _ in spec["heatmaps"]:
            heatmap_path = sample_dir / filename
            if not heatmap_path.is_file():
                raise FileNotFoundError(
                    f"{subset}: missing heatmap: {heatmap_path}")
            heatmap_paths.append(heatmap_path)

        sample_name = sample_dir.name
        original_name = f"{sample_name}{input_path.suffix.lower()}"
        panel_name = f"{sample_name}_five_panel.jpg"
        original_path = originals_dir / original_name
        panel_path = panels_dir / panel_name
        shutil.copy2(input_path, original_path)

        tiles = [load_tile(input_path, "Original")]
        tiles.extend(
            load_tile(path, label)
            for path, (_, label) in zip(heatmap_paths, spec["heatmaps"])
        )
        panel = join_horizontal(tiles)
        panel.save(panel_path, quality=95, subsampling=0)

        output_rows.append({
            "subset": subset,
            "index": index,
            "source_path": row["source_path"],
            "original_file": str(
                original_path.relative_to(subset_root)),
            "five_panel_file": str(
                panel_path.relative_to(subset_root)),
            "layers": row["layers"],
        })

    manifest_path = subset_root / "manifest.tsv"
    fieldnames = [
        "subset",
        "index",
        "source_path",
        "original_file",
        "five_panel_file",
        "layers",
    ]
    with manifest_path.open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    readme = "\n".join([
        f"subset: {subset}",
        f"samples: {len(output_rows)}",
        "five-panel order: Original | Stage 1 | Stage 2 | Stage 3 | Stage 4",
        "originals/: copied source images",
        "five_panels/: one five-panel JPEG per source image",
        "manifest.tsv: source-to-export mapping",
        "",
    ])
    (subset_root / "README.txt").write_text(readme, encoding="utf-8")
    summary_rows.append((spec["export_name"], len(output_rows)))

print("Generated export subsets:")
for name, count in summary_rows:
    print(f"  {name}: originals={count}, five_panels={count}")
PY

mkdir -p "${ARCHIVE_DIR}"

ARCHIVE_DIR="${ARCHIVE_DIR}" \
EXPORT_ROOT="${EXPORT_ROOT}" \
python - <<'PY'
import hashlib
import os
import zipfile
from pathlib import Path

export_root = Path(os.environ["EXPORT_ROOT"]).resolve()
archive_dir = Path(os.environ["ARCHIVE_DIR"]).resolve()
archive_dir.mkdir(parents=True, exist_ok=True)

subset_names = [
    "imagenet_torchvision_resnet50",
    "ruod_supervised_cascade_resnet50_j2",
]
archives = []
for subset_name in subset_names:
    subset_root = export_root / subset_name
    archive_path = archive_dir / f"{subset_name}_random10_five_panels.zip"
    with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(subset_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(export_root))
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(
                f"ZIP integrity check failed for {archive_path}: {bad_file}")
    archives.append(archive_path)

checksum_lines = []
for archive_path in archives:
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checksum_lines.append(f"{digest.hexdigest()}  {archive_path.name}")

(archive_dir / "SHA256SUMS.txt").write_text(
    "\n".join(checksum_lines) + "\n", encoding="utf-8")

print("Created and verified archives:")
for archive_path in archives:
    print(f"  {archive_path} ({archive_path.stat().st_size} bytes)")
print(f"  {archive_dir / 'SHA256SUMS.txt'}")
PY

echo
echo "Export file counts:"
for subset in \
  imagenet_torchvision_resnet50 \
  ruod_supervised_cascade_resnet50_j2
do
  originals="$(find "${EXPORT_ROOT}/${subset}/originals" -maxdepth 1 -type f | wc -l)"
  panels="$(find "${EXPORT_ROOT}/${subset}/five_panels" -maxdepth 1 -type f -name '*_five_panel.jpg' | wc -l)"
  echo "  ${subset}: originals=${originals}, five_panels=${panels}"
  if [[ "${originals}" -ne "${EXPECTED_SAMPLES}" || "${panels}" -ne "${EXPECTED_SAMPLES}" ]]; then
    echo "Error: unexpected export counts for ${subset}" >&2
    exit 1
  fi
done

echo
echo "Archives:"
ls -lh "${ARCHIVE_DIR}"/*.zip "${ARCHIVE_DIR}/SHA256SUMS.txt"

if [[ "${UPLOAD}" == "1" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone not found. Set UPLOAD=0 to skip upload." >&2
    exit 1
  fi
  echo
  echo "Uploading archives to ${RCLONE_DEST}/"
  rclone copy -P "${ARCHIVE_DIR}" "${RCLONE_DEST}/"
  echo
  echo "Remote files:"
  rclone lsl "${RCLONE_DEST}/"
else
  echo
  echo "Upload skipped because UPLOAD=${UPLOAD}."
  echo "After inspection, rerun with UPLOAD=1 to upload the verified archives."
fi

echo
echo "Done."
echo "Export root: ${EXPORT_ROOT}"
echo "Archive dir: ${ARCHIVE_DIR}"
