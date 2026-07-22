#!/usr/bin/env bash
set -euo pipefail

# Add an opt-in guard around legacy WaterGAN air/depth preview writes. This
# changes only inference output I/O; fake-image computation is left untouched.

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

python - "${WATERGAN_DIR}" <<'PY'
from __future__ import print_function

import io
import os
import re
import sys

root = sys.argv[1]
paths = [
    os.path.join(root, name)
    for name in ("modelmhl.py", "modeljamaica.py")
    if os.path.isfile(os.path.join(root, name))
]

if not paths:
    raise RuntimeError("WaterGAN model files not found under {}".format(root))

patterns = (
    (
        re.compile(
            r"^(\s*)scipy\.misc\.imsave\(\s*out_name2\s*,\s*sample_im2\s*\)\s*$",
            re.MULTILINE,
        ),
        r"\1if os.environ.get('WATERGAN_SAVE_AUX_OUTPUTS', '0') == '1': scipy.misc.imsave(out_name2, sample_im2)",
    ),
    (
        re.compile(
            r"^(\s*)sio\.savemat\(\s*out_name3\s*,\s*\{'depth'\s*:\s*sample_im3\}\s*\)\s*$",
            re.MULTILINE,
        ),
        r"\1if os.environ.get('WATERGAN_SAVE_AUX_OUTPUTS', '0') == '1': sio.savemat(out_name3, {'depth': sample_im3})",
    ),
)

for path in paths:
    with io.open(path, "r", encoding="utf-8") as handle:
        original = handle.read()

    if "WATERGAN_SAVE_AUX_OUTPUTS" in original:
        print("Auxiliary-output guard already present: {}".format(path))
        continue

    updated = original
    replacements = 0
    for pattern, replacement in patterns:
        updated, count = pattern.subn(replacement, updated)
        replacements += count

    if replacements == 0:
        raise RuntimeError(
            "Could not locate legacy auxiliary-output writes in {}".format(path)
        )

    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(updated)

    print(
        "Added auxiliary-output guard: {} ({} statements)".format(
            path, replacements
        )
    )
PY

python -m py_compile "${WATERGAN_DIR}/modelmhl.py"

if [[ -f "${WATERGAN_DIR}/modeljamaica.py" ]]; then
  python -m py_compile "${WATERGAN_DIR}/modeljamaica.py"
fi

grep -q 'WATERGAN_SAVE_AUX_OUTPUTS' "${WATERGAN_DIR}/modelmhl.py" || {
  echo "Error: auxiliary-output guard was not installed in modelmhl.py" >&2
  exit 1
}

echo "WaterGAN auxiliary-output guard: OK"
