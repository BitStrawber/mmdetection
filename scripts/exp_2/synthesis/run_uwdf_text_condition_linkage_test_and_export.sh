#!/usr/bin/env bash
set -u

# Run an 8-way UWDF text/condition linkage test and export comparison grids.
#
# Experiments:
#   e1_base_text_only           base prompt,   no style, no depth
#   e2_linked_text_only         linked prompt, no style, no depth
#   e3_base_text_style          base prompt,   style,    no depth
#   e4_linked_text_style        linked prompt, style,    no depth
#   e5_base_text_depth          base prompt,   no style, depth
#   e6_linked_text_depth        linked prompt, no style, depth
#   e7_base_text_style_depth    base prompt,   style,    depth
#   e8_linked_text_style_depth  linked prompt, style,    depth
#
# The script intentionally keeps going when one experiment fails so that partial
# results can still be inspected and exported.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

UWDF_DIR="${UWDF_DIR:-/home/fcp/xcx/exp_2/syn/uwdf}"
SOURCE_ROOT="${SOURCE_ROOT:-/media/SSD1/XCX/exp_2/synthetic_imagenet/uwdf/source/train}"
DEPTH_ROOT="${DEPTH_ROOT:-/media/SSD1/XCX/exp_2/depthanything_v2_maps/uwdf/train}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/media/SSD1/XCX/exp_2/UWNR_ref_underwater/lnrud_like_ref/qingxi}"
WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work/uwdf_text_condition_linkage_test}"
EXP_ROOT="${EXP_ROOT:-${WORK_ROOT}/experiments}"
SAMPLE_ROOT="${SAMPLE_ROOT:-${WORK_ROOT}/samples}"
OUT_ROOT="${OUT_ROOT:-/media/HDD1/XCX/exp_2/uwdf_text_condition_linkage_test_grid_export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/media/HDD1/XCX/exp_2/uwdf_text_condition_linkage_test_grid_export.tar.gz}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/uwdf_text_condition_linkage_test}"
RCLONE_DEST="${RCLONE_DEST:-fcp:datasets/exp2_synthesis_visual/}"

NUM="${NUM:-20}"
SEED="${SEED:-2026}"
GPU_IDS="${GPU_IDS:-2,4,5,6,7}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.75}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-8.0}"
IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.35}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.85}"
TILE_SIZE="${TILE_SIZE:-512}"
UPLOAD="${UPLOAD:-1}"
OVERWRITE="${OVERWRITE:-1}"
RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE:-1}"
RESIZE_MODE="${RESIZE_MODE:-pad}"

BASE_PROMPT="${BASE_PROMPT:-a realistic underwater photograph}"
LINKED_PROMPT="${LINKED_PROMPT:-a realistic underwater photograph, with underwater visual appearance guided by the reference image and spatial structure guided by the depth map}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, unrealistic image, artificial colors, object deformation, changed object identity, extra objects, text, watermark, low quality, worst quality}"

GEN_SCRIPT="${GEN_SCRIPT:-${UWDF_DIR}/scripts/run_ipadapter_controlnet_depth_generate.sh}"
LIGHTFIELD_SCRIPT="${LIGHTFIELD_SCRIPT:-${UWDF_DIR}/scripts/make_reference_lightfield.py}"

mkdir -p "${EXP_ROOT}" "${SAMPLE_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "Error: GPU_IDS is empty" >&2
  exit 1
fi

echo "========================================="
echo "UWDF text-condition linkage test"
echo "========================================="
echo "UWDF_DIR:          ${UWDF_DIR}"
echo "GEN_SCRIPT:        ${GEN_SCRIPT}"
echo "SOURCE_ROOT:       ${SOURCE_ROOT}"
echo "DEPTH_ROOT:        ${DEPTH_ROOT}"
echo "REFERENCE_ROOT:    ${REFERENCE_ROOT}"
echo "WORK_ROOT:         ${WORK_ROOT}"
echo "EXP_ROOT:          ${EXP_ROOT}"
echo "OUT_ROOT:          ${OUT_ROOT}"
echo "ARCHIVE_PATH:      ${ARCHIVE_PATH}"
echo "NUM:               ${NUM}"
echo "SEED:              ${SEED}"
echo "GPU_IDS:           ${GPU_IDS}"
echo "SIZE:              ${WIDTH}x${HEIGHT}"
echo "STEPS:             ${STEPS}"
echo "STRENGTH:          ${STRENGTH}"
echo "GUIDANCE_SCALE:    ${GUIDANCE_SCALE}"
echo "IP_ADAPTER_SCALE:  ${IP_ADAPTER_SCALE}"
echo "CONTROLNET_SCALE:  ${CONTROLNET_SCALE}"
echo "BASE_PROMPT:       ${BASE_PROMPT}"
echo "LINKED_PROMPT:     ${LINKED_PROMPT}"
echo "UPLOAD:            ${UPLOAD}"
echo "RCLONE_DEST:       ${RCLONE_DEST}"
echo "========================================="

for p in "${GEN_SCRIPT}" "${SOURCE_ROOT}" "${DEPTH_ROOT}" "${REFERENCE_ROOT}"; do
  if [ ! -e "${p}" ]; then
    echo "Error: required path not found: ${p}" >&2
    exit 1
  fi
done

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf "${EXP_ROOT}" "${SAMPLE_ROOT}" "${OUT_ROOT}"
  mkdir -p "${EXP_ROOT}" "${SAMPLE_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}"
fi

echo "Step 1/4: select deterministic samples"
python - "${SOURCE_ROOT}" "${DEPTH_ROOT}" "${REFERENCE_ROOT}" "${SAMPLE_ROOT}" "${NUM}" "${SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
depth_root = Path(sys.argv[2])
ref_root = Path(sys.argv[3])
sample_root = Path(sys.argv[4])
num = int(sys.argv[5])
seed = int(sys.argv[6])

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
sources = sorted(p for p in source_root.rglob("*") if p.suffix.lower() in exts)
refs = sorted(p for p in ref_root.rglob("*") if p.suffix.lower() in exts)
if not sources:
    raise SystemExit(f"no source images found under {source_root}")
if not refs:
    raise SystemExit(f"no reference images found under {ref_root}")

rng = random.Random(seed)
selected = rng.sample(sources, min(num, len(sources)))
manifest = []
for i, src in enumerate(selected):
    rel = src.relative_to(source_root)
    depth = depth_root / rel.with_suffix(".png")
    ref = refs[i % len(refs)]
    if not depth.exists():
        raise SystemExit(f"missing depth for {rel}: {depth}")
    stem = f"{i:04d}_{rel.parent.as_posix().replace('/', '_')}_{rel.stem}"
    manifest.append({
        "index": i,
        "id": stem,
        "relative": rel.as_posix(),
        "source": str(src),
        "depth": str(depth),
        "raw_ref": str(ref),
    })

sample_root.mkdir(parents=True, exist_ok=True)
(sample_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
with (sample_root / "manifest.tsv").open("w", encoding="utf-8") as f:
    f.write("id\tsource\tdepth\traw_ref\n")
    for item in manifest:
        f.write(f"{item['id']}\t{item['source']}\t{item['depth']}\t{item['raw_ref']}\n")
print(f"selected {len(manifest)} samples")
print(f"manifest: {sample_root / 'manifest.json'}")
PY

echo "Step 2/4: prepare per-experiment condition folders"
python - "${SAMPLE_ROOT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

sample_root = Path(sys.argv[1])
manifest = json.loads((sample_root / "manifest.json").read_text(encoding="utf-8"))
for sub in ["source", "depth", "raw_ref"]:
    d = sample_root / sub
    d.mkdir(parents=True, exist_ok=True)
for item in manifest:
    suffix = Path(item["source"]).suffix
    shutil.copy2(item["source"], sample_root / "source" / f"{item['id']}{suffix}")
    shutil.copy2(item["depth"], sample_root / "depth" / f"{item['id']}.png")
    ref_suffix = Path(item["raw_ref"]).suffix
    shutil.copy2(item["raw_ref"], sample_root / "raw_ref" / f"{item['id']}{ref_suffix}")
PY

if [ -f "${LIGHTFIELD_SCRIPT}" ]; then
  echo "Step 2b/4: build lightfield references"
  python "${LIGHTFIELD_SCRIPT}" \
    --input-dir "${SAMPLE_ROOT}/raw_ref" \
    --output-dir "${SAMPLE_ROOT}/lightfield_ref" \
    2>&1 | tee "${LOG_ROOT}/make_lightfield.log" || {
      echo "Warning: lightfield conversion failed; falling back to raw references" >&2
      rm -rf "${SAMPLE_ROOT}/lightfield_ref"
      cp -a "${SAMPLE_ROOT}/raw_ref" "${SAMPLE_ROOT}/lightfield_ref"
    }
else
  echo "Warning: LIGHTFIELD_SCRIPT not found; using raw references as lightfield_ref: ${LIGHTFIELD_SCRIPT}" >&2
  rm -rf "${SAMPLE_ROOT}/lightfield_ref"
  cp -a "${SAMPLE_ROOT}/raw_ref" "${SAMPLE_ROOT}/lightfield_ref"
fi

declare -a EXP_NAMES=(
  "e1_base_text_only"
  "e2_linked_text_only"
  "e3_base_text_style"
  "e4_linked_text_style"
  "e5_base_text_depth"
  "e6_linked_text_depth"
  "e7_base_text_style_depth"
  "e8_linked_text_style_depth"
)
declare -a EXP_PROMPTS=(
  "${BASE_PROMPT}"
  "${LINKED_PROMPT}"
  "${BASE_PROMPT}"
  "${LINKED_PROMPT}"
  "${BASE_PROMPT}"
  "${LINKED_PROMPT}"
  "${BASE_PROMPT}"
  "${LINKED_PROMPT}"
)
declare -a EXP_STYLE_SCALES=(
  "0"
  "0"
  "${IP_ADAPTER_SCALE}"
  "${IP_ADAPTER_SCALE}"
  "0"
  "0"
  "${IP_ADAPTER_SCALE}"
  "${IP_ADAPTER_SCALE}"
)
declare -a EXP_DEPTH_SCALES=(
  "0"
  "0"
  "0"
  "0"
  "${CONTROLNET_SCALE}"
  "${CONTROLNET_SCALE}"
  "${CONTROLNET_SCALE}"
  "${CONTROLNET_SCALE}"
)

echo "Step 3/4: run 8 experiments"
STATUS_TSV="${LOG_ROOT}/status.tsv"
printf "experiment\tstatus\tgpu\tlog\n" > "${STATUS_TSV}"

for i in "${!EXP_NAMES[@]}"; do
  name="${EXP_NAMES[$i]}"
  prompt="${EXP_PROMPTS[$i]}"
  style_scale="${EXP_STYLE_SCALES[$i]}"
  depth_scale="${EXP_DEPTH_SCALES[$i]}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  out_dir="${EXP_ROOT}/${name}"
  log_path="${LOG_ROOT}/${name}.log"

  echo "-----------------------------------------"
  echo "experiment: ${name}"
  echo "gpu:        ${gpu}"
  echo "style:      ${style_scale}"
  echo "depth:      ${depth_scale}"
  echo "out_dir:    ${out_dir}"
  echo "prompt:     ${prompt}"
  echo "-----------------------------------------"

  mkdir -p "${out_dir}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SOURCE_ROOT="${SAMPLE_ROOT}/source" \
  DEPTH_ROOT="${SAMPLE_ROOT}/depth" \
  REFERENCE_ROOT="${SAMPLE_ROOT}/lightfield_ref" \
  OUT_ROOT="${out_dir}" \
  PROMPT="${prompt}" \
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
  NUM="${NUM}" \
  SEED="${SEED}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  STEPS="${STEPS}" \
  STRENGTH="${STRENGTH}" \
  GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
  IP_ADAPTER_SCALE="${style_scale}" \
  CONTROLNET_SCALE="${depth_scale}" \
  RESIZE_MODE="${RESIZE_MODE}" \
  RESTORE_SOURCE_SIZE="${RESTORE_SOURCE_SIZE}" \
  bash "${GEN_SCRIPT}" 2>&1 | tee "${log_path}"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "${rc}" -eq 0 ]; then
    printf "%s\tok\t%s\t%s\n" "${name}" "${gpu}" "${log_path}" >> "${STATUS_TSV}"
  else
    printf "%s\tfailed:%s\t%s\t%s\n" "${name}" "${rc}" "${gpu}" "${log_path}" >> "${STATUS_TSV}"
    echo "Warning: ${name} failed with code ${rc}; continuing" >&2
  fi
done

echo "Step 4/4: export multi-grids"
python - "${SAMPLE_ROOT}" "${EXP_ROOT}" "${OUT_ROOT}" "${TILE_SIZE}" <<'PY'
import json
import math
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

sample_root = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
tile = int(sys.argv[4])
manifest = json.loads((sample_root / "manifest.json").read_text(encoding="utf-8"))
out_root.mkdir(parents=True, exist_ok=True)

experiments = [
    ("e1_base_text_only", "base text"),
    ("e2_linked_text_only", "linked text"),
    ("e3_base_text_style", "base + style"),
    ("e4_linked_text_style", "linked + style"),
    ("e5_base_text_depth", "base + depth"),
    ("e6_linked_text_depth", "linked + depth"),
    ("e7_base_text_style_depth", "base + style + depth"),
    ("e8_linked_text_style_depth", "linked + style + depth"),
]

def font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

label_font = font(22)
small_font = font(16)

def open_fit(path, fill=(245, 245, 245)):
    path = Path(path)
    if not path.exists():
        im = Image.new("RGB", (tile, tile), fill)
        d = ImageDraw.Draw(im)
        d.text((16, 16), "missing", fill=(180, 0, 0), font=label_font)
        d.text((16, 48), path.name[:40], fill=(80, 80, 80), font=small_font)
        return im
    im = Image.open(path).convert("RGB")
    im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile, tile), fill)
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return canvas

def find_one(directory, sample_id):
    directory = Path(directory)
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        matches = sorted(directory.rglob(f"{sample_id}*{ext}"))
        if matches:
            return matches[0]
    matches = sorted(directory.rglob(f"*{sample_id}*"))
    return matches[0] if matches else directory / f"{sample_id}.png"

def labeled(im, title):
    header = 34
    canvas = Image.new("RGB", (tile, tile + header), (255, 255, 255))
    canvas.paste(im, (0, header))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, tile - 1, header - 1], fill=(32, 36, 40))
    d.text((10, 7), title, fill=(255, 255, 255), font=small_font)
    return canvas

for item in manifest:
    sid = item["id"]
    source = find_one(sample_root / "source", sid)
    depth = find_one(sample_root / "depth", sid)
    raw_ref = find_one(sample_root / "raw_ref", sid)
    light_ref = find_one(sample_root / "lightfield_ref", sid)

    conds = [
        labeled(open_fit(source), "source"),
        labeled(open_fit(raw_ref), "raw ref"),
        labeled(open_fit(light_ref), "lightfield ref"),
        labeled(open_fit(depth), "depth"),
    ]
    exp_imgs = [(name, title, labeled(open_fit(find_one(exp_root / name, sid)), title)) for name, title in experiments]

    cols = 4
    rows = 1 + 4
    gap = 12
    margin = 18
    cell_w, cell_h = tile, tile + 34
    W = margin * 2 + cols * cell_w + (cols - 1) * gap
    H = margin * 2 + rows * cell_h + (rows - 1) * gap + 44
    grid = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(grid)
    d.text((margin, 12), f"{sid}  |  UWDF text-condition linkage", fill=(25, 25, 25), font=label_font)

    y = margin + 44
    for c, im in enumerate(conds):
        x = margin + c * (cell_w + gap)
        grid.paste(im, (x, y))

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for r, pair in enumerate(pairs, start=1):
        y = margin + 44 + r * (cell_h + gap)
        for j, idx in enumerate(pair):
            x = margin + j * (cell_w + gap)
            grid.paste(exp_imgs[idx][2], (x, y))

    out_path = out_root / f"{sid}_linkage_grid.png"
    grid.save(out_path)

print(f"exported {len(manifest)} grids to {out_root}")
PY

echo "Create archive: ${ARCHIVE_PATH}"
mkdir -p "$(dirname "${ARCHIVE_PATH}")"
tar -C "$(dirname "${OUT_ROOT}")" -czf "${ARCHIVE_PATH}" "$(basename "${OUT_ROOT}")"
ls -lh "${ARCHIVE_PATH}"

if [ "${UPLOAD}" = "1" ]; then
  echo "Upload archive and folder to: ${RCLONE_DEST}"
  rclone copy "${ARCHIVE_PATH}" "${RCLONE_DEST}"
  rclone copy "${OUT_ROOT}" "${RCLONE_DEST}/$(basename "${OUT_ROOT}")"
fi

echo "========================================="
echo "UWDF text-condition linkage test done"
echo "========================================="
echo "status:  ${STATUS_TSV}"
echo "samples: ${SAMPLE_ROOT}"
echo "grids:   ${OUT_ROOT}"
echo "archive: ${ARCHIVE_PATH}"
echo "========================================="
