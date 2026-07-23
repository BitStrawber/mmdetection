#!/usr/bin/env bash
set -euo pipefail

WATERGAN_DIR="${WATERGAN_DIR:-/home/fcp/xcx/exp_2/syn/WaterGAN_legacy_20260714}"

python - "${WATERGAN_DIR}" <<'PY'
from __future__ import print_function

import re
import shutil
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
paths = [
    root / 'modelmhl.py',
    root / 'modeljamaica.py',
    root / 'utils.py',
]
marker = '# WaterGAN exp_2 trusted-PNG metadata allowance.'
injected = '''# WaterGAN exp_2 trusted-PNG metadata allowance.
# Some prepared ImageNet PNGs carry ICC profiles above Pillow's
# conservative 1 MiB metadata limit. This changes metadata decoding
# limits only; image pixels and model inputs remain unchanged.
from PIL import PngImagePlugin as _WaterganPngImagePlugin
_WaterganPngImagePlugin.MAX_TEXT_CHUNK = max(
    _WaterganPngImagePlugin.MAX_TEXT_CHUNK, 128 * 1024 * 1024
)
_WaterganPngImagePlugin.MAX_TEXT_MEMORY = max(
    _WaterganPngImagePlugin.MAX_TEXT_MEMORY, 256 * 1024 * 1024
)
'''

patched = 0
verified = 0
pattern = re.compile(
    r'(?P<indent>^[ \t]*)def _watergan_imread'
    r'\(filename, flatten=False, mode=None\):\r?\n'
    r'(?P<body_indent>[ \t]+)(?=with _PILImage\.open\(filename\) as image:)',
    flags=re.MULTILINE,
)

for path in paths:
    if not path.is_file():
        continue
    text = path.read_text(encoding='utf-8')
    if 'def _watergan_imread' not in text:
        continue
    if marker in text:
        print('{}: already patched'.format(path))
        verified += 1
        continue

    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            'Could not locate _watergan_imread body in {}'.format(path))
    body_indent = match.group('body_indent')
    block = ''.join(
        body_indent + line if line.strip() else line
        for line in injected.splitlines(True))
    text = text[:match.end()] + block + text[match.end():]
    backup = path.with_name(path.name + '.before_png_metadata_limit')
    if not backup.exists():
        shutil.copy2(str(path), str(backup))
    path.write_text(text, encoding='utf-8')
    print('{}: patched'.format(path))
    patched += 1
    verified += 1

if verified == 0:
    raise RuntimeError(
        'No WaterGAN _watergan_imread compatibility function was found')
print('WaterGAN trusted-PNG metadata patch: patched={}, verified={}'.format(
    patched, verified))
PY

python -m py_compile \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py"

grep -q \
  'WaterGAN exp_2 trusted-PNG metadata allowance' \
  "${WATERGAN_DIR}/modelmhl.py"

echo "WaterGAN Pillow PNG metadata compatibility: OK"
