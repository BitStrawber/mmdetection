#!/usr/bin/env bash
set -euo pipefail

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN}"

for path in \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py"; do
  [[ -f "${path}" ]] || {
    echo "Error: WaterGAN model file not found: ${path}" >&2
    exit 1
  }
done

python - \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py" <<'PY'
from __future__ import print_function

import io
import re
import sys

normalization = re.compile(
    r"^(?P<indent>[ \t]*)depth\s*=\s*np\.multiply\(\s*"
    r"self\.max_depth\s*,\s*np\.divide\(\s*depth\s*,\s*"
    r"depth\.max\(\)\s*\)\s*\)\s*$",
    flags=re.MULTILINE,
)
risky_normalization = re.compile(
    r"np\.divide\(\s*depth\s*,\s*depth\.max\(\)\s*\)"
)
marker = "WaterGAN safe depth normalization for invalid maps."


def safe_normalization(match):
    indent = match.group("indent")
    body = [
        "# " + marker,
        "depth = np.asarray(depth, dtype=np.float32)",
        "finite_mask = np.isfinite(depth)",
        "if not finite_mask.all():",
        "  print(\"Warning: non-finite depth values replaced: {}\".format(filename))",
        "  depth = np.where(finite_mask, depth, 0.0)",
        "depth_max = float(depth.max()) if depth.size else 0.0",
        "if not np.isfinite(depth_max) or depth_max <= 0.0:",
        "  print(\"Warning: zero/invalid depth map replaced: {}\".format(filename))",
        "  depth = np.zeros_like(depth, dtype=np.float32)",
        "else:",
        "  depth = np.multiply(",
        "    self.max_depth,",
        "    np.divide(depth, depth_max)",
        "  )",
    ]
    return "\n".join(indent + line for line in body)


for filename in sys.argv[1:]:
    with io.open(filename, "r", encoding="utf-8") as handle:
        text = handle.read()

    typo_count = text.count("self.resuts_dir")
    if typo_count:
        text = text.replace("self.resuts_dir", "self.results_dir")

    text, depth_count = normalization.subn(safe_normalization, text)

    if risky_normalization.search(text):
        raise RuntimeError(
            "unsafe depth normalization remains in {}".format(filename)
        )
    if depth_count == 0 and marker not in text:
        raise RuntimeError(
            "could not find WaterGAN depth normalization in {}".format(
                filename
            )
        )

    if typo_count or depth_count:
        with io.open(filename, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    print(
        "WaterGAN runtime safety: {} typo_fix={}, depth_fix={}".format(
            filename, typo_count, depth_count
        )
    )
PY

python -m py_compile \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py"

echo "WaterGAN depth normalization and results_dir checks: OK"
