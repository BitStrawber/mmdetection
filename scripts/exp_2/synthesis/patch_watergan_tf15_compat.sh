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

SCIPY_MISC_COMPAT = '''

# Compatibility for SciPy versions where scipy.misc image I/O was removed.
try:
    scipy.misc.imread
except AttributeError:
    from PIL import Image as _PILImage
    import numpy as _watergan_np

    def _watergan_imread(filename, flatten=False, mode=None):
        image = _PILImage.open(filename)
        if flatten:
            image = image.convert('L')
        elif mode is not None:
            image = image.convert(mode)
        else:
            image = image.convert('RGB')
        return _watergan_np.asarray(image)

    def _watergan_imresize(arr, size, interp='bilinear', mode=None):
        image = _PILImage.fromarray(_watergan_np.asarray(arr))
        if isinstance(size, (int, float)):
            if isinstance(size, int):
                scale = size / 100.0
            else:
                scale = float(size)
            new_size = (
                max(1, int(round(image.size[0] * scale))),
                max(1, int(round(image.size[1] * scale))),
            )
        else:
            # scipy.misc.imresize used (height, width); PIL uses (width, height).
            new_size = (int(size[1]), int(size[0]))
        resample = _PILImage.BILINEAR if interp != 'nearest' else _PILImage.NEAREST
        return _watergan_np.asarray(image.resize(new_size, resample))

    def _watergan_imsave(filename, arr):
        image = _PILImage.fromarray(_watergan_np.asarray(arr))
        image.save(filename)

    scipy.misc.imread = _watergan_imread
    scipy.misc.imresize = _watergan_imresize
    scipy.misc.imsave = _watergan_imsave
'''

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
    if "scipy.misc." in text and "Compatibility for SciPy versions where scipy.misc image I/O was removed" not in text:
        if "import scipy.misc" in text:
            text = text.replace("import scipy.misc", "import scipy.misc" + SCIPY_MISC_COMPAT, 1)
        elif "import scipy" in text:
            text = text.replace("import scipy", "import scipy" + SCIPY_MISC_COMPAT, 1)
        else:
            text = "import scipy.misc" + SCIPY_MISC_COMPAT + "\n" + text
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
echo "SciPy image I/O compatibility status:"
python - "${WATERGAN_DIR}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
marker = "Compatibility for SciPy versions where scipy.misc image I/O was removed"
for path in sorted(root.glob("*.py")):
    text = path.read_text(encoding="utf-8")
    uses_scipy_io = any(
        token in text
        for token in ("scipy.misc.imread", "scipy.misc.imresize", "scipy.misc.imsave")
    )
    if not uses_scipy_io:
        continue
    status = "OK compat marker present" if marker in text else "MISSING compat marker"
    print(f"  {path}: {status}")
PY

echo
echo "Remaining scipy.misc image I/O call sites are expected if the files above are OK:"
grep -RInE "scipy\\.misc\\.(imread|imresize|imsave)" "${WATERGAN_DIR}"/*.py || true

echo
echo "Done. Re-run WaterGAN after activating:"
echo "  conda activate /media/SSD1/conda_envs/watergan_tf1"
