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
paths = [root / 'modelmhl.py', root / 'modeljamaica.py']

helpers = r'''

# WaterGAN exp_2 compact-depth helpers. PNG decoding is numerically identical
# to the float32 depth array stored by materialize_watergan_official_mat_shard.
def _watergan_depth_sort_key(path):
    import os as _watergan_os
    stem = _watergan_os.path.splitext(_watergan_os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)

def _watergan_list_depth_files(depth_dataset):
    import os as _watergan_os
    from glob import glob as _watergan_glob
    root = _watergan_os.path.join('./data', depth_dataset)
    files = []
    for pattern in ('*.mat', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp'):
        files.extend(_watergan_glob(_watergan_os.path.join(root, pattern)))
    return sorted(files, key=_watergan_depth_sort_key)

def _watergan_load_depth_file(filename):
    import os as _watergan_os
    import numpy as _watergan_np
    suffix = _watergan_os.path.splitext(filename)[1].lower()
    if suffix == '.mat':
        import scipy.io as _watergan_sio
        return _watergan_sio.loadmat(filename)
    from PIL import Image as _WaterganDepthImage
    with _WaterganDepthImage.open(filename) as image:
        depth = _watergan_np.asarray(
            image.convert('L'), dtype=_watergan_np.float32
        ) / 255.0
    return {
        'depth': depth,
        'dph': depth,
        'D': depth,
        'data': depth,
    }
'''

depth_data_patterns = [
    re.compile(
        r'depth_data\s*=\s*sorted\(glob\(os\.path\.join\(\s*'
        r'["\']\.\/data["\']\s*,\s*config\.depth_dataset\s*,\s*'
        r'["\']\*\.mat["\']\s*\)\)\)',
        flags=re.DOTALL,
    ),
    re.compile(
        r'depth_data\s*=\s*sorted\(glob\(\s*'
        r'["\']\.\/data\/["\']\s*\+\s*config\.depth_dataset\s*\+\s*'
        r'["\']\/\*\.mat["\']\s*\)\)',
        flags=re.DOTALL,
    ),
]

for path in paths:
    if not path.is_file():
        raise RuntimeError('WaterGAN model file is missing: {}'.format(path))

    original = path.read_text(encoding='utf-8')
    text = original
    has_list = 'def _watergan_list_depth_files' in text
    has_load = 'def _watergan_load_depth_file' in text

    if has_list != has_load:
        raise RuntimeError(
            'partial compact-depth patch found in {}: list={}, load={}'.format(
                path, has_list, has_load
            )
        )

    if not has_list:
        marker = 'from utils import *'
        if marker in text:
            text = text.replace(marker, marker + helpers, 1)
        elif 'class WGAN' in text:
            text = text.replace('class WGAN', helpers + '\nclass WGAN', 1)
        else:
            raise RuntimeError(
                'could not find helper insertion point in {}'.format(path)
            )

    text = re.sub(
        r'(?<![A-Za-z0-9_])sio\.loadmat\(([^)\n]+)\)',
        r'_watergan_load_depth_file(\1)',
        text,
    )
    text = re.sub(
        r'(?<![A-Za-z0-9_])scipy\.io\.loadmat\(([^)\n]+)\)',
        r'_watergan_load_depth_file(\1)',
        text,
    )
    for pattern in depth_data_patterns:
        text = pattern.sub(
            'depth_data = _watergan_list_depth_files(config.depth_dataset)',
            text,
        )

    if '_watergan_list_depth_files(config.depth_dataset)' not in text:
        text = re.sub(
            r'^(\s*)depth_data\s*=.*config\.depth_dataset.*\*\.mat.*$',
            r'\1depth_data = _watergan_list_depth_files(config.depth_dataset)',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    if '_watergan_list_depth_files(config.depth_dataset)' not in text:
        raise RuntimeError(
            'could not find or patch depth_data discovery in {}'.format(path)
        )
    if text.count('_watergan_load_depth_file(') < 2:
        raise RuntimeError(
            'compact depth loader is not used by {}'.format(path)
        )

    if text != original:
        backup = path.with_name(path.name + '.before_png_depth_loader')
        if not backup.exists():
            shutil.copy2(str(path), str(backup))
        path.write_text(text, encoding='utf-8')
        print('{}: patched'.format(path))
    else:
        print('{}: already patched'.format(path))
PY

python -m py_compile \
  "${WATERGAN_DIR}/modelmhl.py" \
  "${WATERGAN_DIR}/modeljamaica.py"

echo "WaterGAN compact PNG depth loader: OK"
