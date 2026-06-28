#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

WORK_ROOT="${WORK_ROOT:-/media/SSD1/XCX/exp_2/synthesis_work}"
FULL_DATA="${FULL_DATA:-${WORK_ROOT}/cut/datasets/imagenet_ruod_cut_full_ssd}"
TINY_DATA="${TINY_DATA:-${WORK_ROOT}/cut/datasets/imagenet_ruod_cut_tiny_from_full_ssd}"
EXP_NAME="${EXP_NAME:-imagenet_ruod_cut_tiny_from_full_ssd}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/synthesis_smoke/cut_full_tiny_check}"

GPU="${GPU:-2}"
GPU_IDS="${GPU_IDS:-${GPU}}"
TRAIN_A_LIMIT="${TRAIN_A_LIMIT:-100}"
TRAIN_B_LIMIT="${TRAIN_B_LIMIT:-100}"
TEST_A_LIMIT="${TEST_A_LIMIT:-30}"
TEST_B_LIMIT="${TEST_B_LIMIT:-30}"
CUT_EPOCHS="${CUT_EPOCHS:-1}"
CUT_BATCH_SIZE="${CUT_BATCH_SIZE:-1}"
CUT_NUM_THREADS="${CUT_NUM_THREADS:-4}"
OVERWRITE="${OVERWRITE:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-${WORK_ROOT}/cut/results/${EXP_NAME}_val}"
RESTORE_DIR="${RESTORE_DIR:-${WORK_ROOT}/cut/generated_tiny_from_full/val}"

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "CUT full-data tiny usability check"
echo "========================================="
echo "FULL_DATA:       ${FULL_DATA}"
echo "TINY_DATA:       ${TINY_DATA}"
echo "EXP_NAME:        ${EXP_NAME}"
echo "GPU_IDS:         ${GPU_IDS}"
echo "TRAIN_A_LIMIT:   ${TRAIN_A_LIMIT}"
echo "TRAIN_B_LIMIT:   ${TRAIN_B_LIMIT}"
echo "TEST_A_LIMIT:    ${TEST_A_LIMIT}"
echo "TEST_B_LIMIT:    ${TEST_B_LIMIT}"
echo "CUT_EPOCHS:      ${CUT_EPOCHS}"
echo "CUT_BATCH_SIZE:  ${CUT_BATCH_SIZE}"
echo "CUT_NUM_THREADS: ${CUT_NUM_THREADS}"
echo "RESULTS_ROOT:    ${RESULTS_ROOT}"
echo "RESTORE_DIR:     ${RESTORE_DIR}"
echo "LOG_DIR:         ${LOG_DIR}"
echo "========================================="

if [[ ! -d "${FULL_DATA}" ]]; then
  echo "Error: full CUT dataset not found: ${FULL_DATA}" >&2
  exit 1
fi

echo
echo "Step 1/4: Build tiny CUT dataset from full SSD dataset"
FULL_DATA="${FULL_DATA}" \
TINY_DATA="${TINY_DATA}" \
TRAIN_A_LIMIT="${TRAIN_A_LIMIT}" \
TRAIN_B_LIMIT="${TRAIN_B_LIMIT}" \
TEST_A_LIMIT="${TEST_A_LIMIT}" \
TEST_B_LIMIT="${TEST_B_LIMIT}" \
OVERWRITE="${OVERWRITE}" \
python - <<'PY' 2>&1 | tee "${LOG_DIR}/prepare_tiny.log"
import json
import os
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

full = Path(os.environ["FULL_DATA"])
tiny = Path(os.environ["TINY_DATA"])
limits = {
    "trainA": int(os.environ["TRAIN_A_LIMIT"]),
    "trainB": int(os.environ["TRAIN_B_LIMIT"]),
    "testA": int(os.environ["TEST_A_LIMIT"]),
    "testB": int(os.environ["TEST_B_LIMIT"]),
}
overwrite = os.environ.get("OVERWRITE", "1") == "1"

if tiny.exists():
    if not overwrite:
        print(f"exists, reuse tiny dataset: {tiny}")
        raise SystemExit(0)
    shutil.rmtree(tiny)

for split in [*limits.keys(), "manifests"]:
    (tiny / split).mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

summary = {
    "full_data": str(full),
    "tiny_data": str(tiny),
    "limits": limits,
    "splits": {},
}

for split, limit in limits.items():
    src_dir = full / split
    dst_dir = tiny / split
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing split: {src_dir}")
    files = sorted(p for p in src_dir.iterdir() if p.is_file() or p.is_symlink())
    selected = files[:limit] if limit > 0 else files
    for src in tqdm(selected, desc=f"link {split}", unit="image"):
        link_or_copy(src.resolve(), dst_dir / src.name)

    src_manifest = full / "manifests" / f"{split}_manifest.jsonl"
    dst_manifest = tiny / "manifests" / f"{split}_manifest.jsonl"
    manifest_lines = []
    if src_manifest.exists():
        all_lines = [x for x in src_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        manifest_lines = all_lines[:len(selected)]
        dst_manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

    summary["splits"][split] = {
        "source_count": len(files),
        "selected": len(selected),
        "manifest_lines": len(manifest_lines),
        "destination": str(dst_dir),
    }

summary_path = tiny / "manifests" / "tiny_from_full_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"summary: {summary_path}")
PY

echo
echo "Tiny dataset counts:"
echo "  trainA: $(find "${TINY_DATA}/trainA" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  trainB: $(find "${TINY_DATA}/trainB" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  testA:  $(find "${TINY_DATA}/testA" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"
echo "  testB:  $(find "${TINY_DATA}/testB" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)"

echo
echo "Step 2/4: Train CUT for tiny check"
DATA_NAME="${EXP_NAME}" \
DATA_ROOT="${TINY_DATA}" \
EXP_NAME="${EXP_NAME}" \
GPU_IDS="${GPU_IDS}" \
BATCH_SIZE="${CUT_BATCH_SIZE}" \
NUM_THREADS="${CUT_NUM_THREADS}" \
N_EPOCHS="${CUT_EPOCHS}" \
N_EPOCHS_DECAY=0 \
PRINT_FREQ=10 \
SAVE_EPOCH_FREQ=1 \
bash scripts/exp_2/synthesis/run_cut_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

echo
echo "Step 3/4: Generate CUT fake_B for tiny val"
DATA_NAME="${EXP_NAME}" \
DATA_ROOT="${TINY_DATA}" \
EXP_NAME="${EXP_NAME}" \
SPLIT=val \
GPU_IDS="${GPU_IDS}" \
NUM_TEST="${TEST_A_LIMIT}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
RESTORE_DIR="${RESTORE_DIR}" \
MANIFEST="${TINY_DATA}/manifests/testA_manifest.jsonl" \
bash scripts/exp_2/synthesis/run_cut_generate.sh \
  2>&1 | tee "${LOG_DIR}/generate_val.log"

echo
echo "Step 4/4: Verify generated outputs"
generated_count="$(
  find "${RESTORE_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) 2>/dev/null | wc -l
)"
echo "Generated image count: ${generated_count}"
echo "Expected at least:      ${TEST_A_LIMIT}"

if [[ "${generated_count}" -lt "${TEST_A_LIMIT}" ]]; then
  echo "Error: generated output count is lower than expected." >&2
  exit 1
fi

echo
echo "CUT tiny usability check completed."
echo "Logs:    ${LOG_DIR}"
echo "Dataset: ${TINY_DATA}"
echo "Output:  ${RESTORE_DIR}"
