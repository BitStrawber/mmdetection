#!/usr/bin/env bash
set -euo pipefail

# Patch the old WaterGAN TensorFlow code for TensorFlow 1.15 compatibility.
# The original repo uses several early TensorFlow API names such as tf.pack
# that were removed/replaced in later TF1.x releases.
#
# Usage:
#   cd ~/xcx/exp_2/mmdetection
#   bash scripts/exp_2/synthesis/patch_watergan_tf15_compat.sh

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

if [[ ! -d "${WATERGAN_DIR}" ]]; then
  echo "Error: WaterGAN repo not found: ${WATERGAN_DIR}" >&2
  exit 1
fi

echo "========================================="
echo "Patch WaterGAN for TensorFlow 1.15"
echo "========================================="
echo "WATERGAN_DIR: ${WATERGAN_DIR}"
echo "========================================="

python - "${WATERGAN_DIR}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
files = [
    root / "mainmhl.py",
    root / "mainjamaica.py",
    root / "modelmhl.py",
    root / "modeljamaica.py",
    root / "ops.py",
    root / "utils.py",
]

replacements = [
    ("tf.pack(", "tf.stack("),
    ("tf.unpack(", "tf.unstack("),
    ("tf.mul(", "tf.multiply("),
    ("tf.sub(", "tf.subtract("),
    ("tf.neg(", "tf.negative("),
    ("tf.initialize_all_variables()", "tf.global_variables_initializer()"),
    ("tf.train.SummaryWriter", "tf.summary.FileWriter"),
]

# Very old DCGAN-style code sometimes used positional sigmoid CE:
# tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
# TF 1.15 accepts named args more reliably.
sigmoid_ce = re.compile(
    r"tf\.nn\.sigmoid_cross_entropy_with_logits\(([^,\n]+),\s*([^)]+)\)"
)
sigmoid_ce_targets = re.compile(
    r"tf\.nn\.sigmoid_cross_entropy_with_logits\(([^)]*)targets=",
    flags=re.DOTALL,
)

changed = []
for path in files:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    old = text
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = sigmoid_ce.sub(
        r"tf.nn.sigmoid_cross_entropy_with_logits(logits=\1, labels=\2)",
        text,
    )
    while True:
        new_text = sigmoid_ce_targets.sub(
            r"tf.nn.sigmoid_cross_entropy_with_logits(\1labels=",
            text,
            count=1,
        )
        if new_text == text:
            break
        text = new_text
    if text != old:
        backup = path.with_suffix(path.suffix + ".tf15bak")
        if not backup.exists():
            backup.write_text(old, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        changed.append(str(path))

print("patched files:")
for item in changed:
    print("  " + item)
if not changed:
    print("  none; files already look patched")
PY

echo
echo "Remaining suspicious old TensorFlow symbols:"
grep -RInE "tf\\.(pack|unpack|mul|sub|neg|initialize_all_variables|train\\.SummaryWriter)" \
  "${WATERGAN_DIR}"/*.py || true

echo
echo "Remaining sigmoid_cross_entropy targets= usages:"
grep -RIn "sigmoid_cross_entropy_with_logits.*targets=" "${WATERGAN_DIR}"/*.py || true

echo
echo "Done. Re-run WaterGAN after activating:"
echo "  conda activate /media/SSD1/conda_envs/watergan_tf1"
