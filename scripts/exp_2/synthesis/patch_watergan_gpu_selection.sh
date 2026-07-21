#!/usr/bin/env bash
set -euo pipefail

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

python - "${WATERGAN_DIR}" <<'PY'
from __future__ import print_function

import io
import os
import re
import sys

root = sys.argv[1]
pattern = re.compile(
    r"os\.environ\[(['\"])CUDA_VISIBLE_DEVICES\1\]\s*=\s*(['\"])(\d+)\2"
)
found = False


def make_default(match):
    return "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '{}')".format(
        match.group(3)
    )

for name in ("mainmhl.py", "mainjamaica.py"):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        continue

    found = True
    with io.open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    updated, count = pattern.subn(make_default, text)
    if count:
        with io.open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        print("GPU selection patched: {}".format(path))
    elif "os.environ.setdefault('CUDA_VISIBLE_DEVICES'" in text:
        print("GPU selection already patched: {}".format(path))
    elif "CUDA_VISIBLE_DEVICES" in text:
        raise RuntimeError(
            "unrecognized CUDA_VISIBLE_DEVICES assignment in {}".format(path)
        )
    else:
        print("No internal GPU override found: {}".format(path))

if not found:
    raise RuntimeError("WaterGAN entry points not found under {}".format(root))
PY

python -m py_compile \
  "${WATERGAN_DIR}/mainmhl.py"

if [[ -f "${WATERGAN_DIR}/mainjamaica.py" ]]; then
  python -m py_compile "${WATERGAN_DIR}/mainjamaica.py"
fi
