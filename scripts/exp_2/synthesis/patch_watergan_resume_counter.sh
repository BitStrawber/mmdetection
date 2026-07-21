#!/usr/bin/env bash
set -euo pipefail

# Let legacy WaterGAN continue checkpoint step numbering after a restore. The
# caller supplies WATERGAN_COUNTER_START. Legacy code increments the counter
# before saving, so resuming DCGAN.model-N uses N-1 as the runtime start value.

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"

[[ -d "${WATERGAN_DIR}" ]] || {
  echo "Error: WaterGAN directory not found: ${WATERGAN_DIR}" >&2
  exit 1
}

python - "${WATERGAN_DIR}" <<'PY'
from __future__ import print_function

import io
import os
import re
import sys

root = sys.argv[1]
replacement = (
    'counter = int(os.environ.get("WATERGAN_COUNTER_START", "1"))'
)
pattern = re.compile(r'(?m)^(\s*)counter\s*=\s*1\s*$')

for name in ('modelmhl.py', 'modeljamaica.py'):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        print('{}: missing, skip'.format(path))
        continue

    with io.open(path, 'r', encoding='utf-8') as handle:
        text = handle.read()

    if 'WATERGAN_COUNTER_START' in text:
        print('{}: resume counter patch already present'.format(path))
        continue

    patched, count = pattern.subn(
        lambda match: match.group(1) + replacement,
        text,
    )
    if count == 0:
        raise RuntimeError(
            'Could not find a legacy counter = 1 statement in {}'.format(path)
        )

    with io.open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(patched)
    print('{}: patched {} counter initialization(s)'.format(path, count))
PY

echo
echo "WaterGAN resume counter status:"
grep -RIn 'WATERGAN_COUNTER_START' \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py" \
  2>/dev/null || true
