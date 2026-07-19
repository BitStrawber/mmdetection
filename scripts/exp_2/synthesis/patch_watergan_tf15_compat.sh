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

WATERGAN_DEPTH_HELPERS = r'''

# WaterGAN exp_2 helpers: allow compact PNG depth maps and repeated water-domain
# sampling without physically duplicating 25w RUOD images.
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
    root = _watergan_os.path.join("./data", depth_dataset)
    files = []
    for pattern in ("*.mat", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files.extend(_watergan_glob(_watergan_os.path.join(root, pattern)))
    return sorted(files, key=_watergan_depth_sort_key)

def _watergan_load_depth_file(filename):
    import os as _watergan_os
    import numpy as _watergan_np
    suffix = _watergan_os.path.splitext(filename)[1].lower()
    if suffix == ".mat":
        import scipy.io as _watergan_sio
        return _watergan_sio.loadmat(filename)
    from PIL import Image as _WaterganDepthImage
    with _WaterganDepthImage.open(filename) as image:
        depth = _watergan_np.asarray(
            image.convert("L"), dtype=_watergan_np.float32
        ) / 255.0
    # Preserve the dictionary contract used by old model variants.
    return {
        "depth": depth,
        "dph": depth,
        "D": depth,
        "data": depth,
    }

def _watergan_effective_train_batches(air_data, depth_data, config):
    import numpy as _watergan_np
    train_limit = min(len(air_data), len(depth_data))
    train_size = config.train_size
    if train_size != _watergan_np.inf:
        train_limit = min(train_limit, int(train_size))
    return int(train_limit // config.batch_size)
'''

WATERGAN_READ_DEPTH = r'''def read_depth(path):
    import os as _watergan_os
    import numpy as _watergan_np
    suffix = _watergan_os.path.splitext(path)[1].lower()
    if suffix == ".mat":
        import scipy.io as _watergan_sio
        mat = _watergan_sio.loadmat(path)
        for key in ("depth", "dph", "D", "data"):
            if key in mat:
                arr = mat[key]
                break
        else:
            data_keys = [key for key in mat.keys() if not key.startswith("__")]
            if not data_keys:
                raise ValueError("No depth array found in MAT file: %s" % path)
            arr = mat[data_keys[0]]
        arr = _watergan_np.asarray(arr, dtype=_watergan_np.float32)
    else:
        from PIL import Image as _WaterganPILImage
        with _WaterganPILImage.open(path) as image:
            arr = _watergan_np.asarray(image.convert("L"), dtype=_watergan_np.float32)
    arr = _watergan_np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.size:
        max_value = float(_watergan_np.nanmax(arr))
        if max_value > 1.0:
            arr = arr / 255.0
    return _watergan_np.nan_to_num(arr).astype(_watergan_np.float32)
'''

WATERGAN_RUNTIME_HELPERS = r'''

# WaterGAN exp_2 runtime helpers: parallelize image decoding and avoid running
# expensive summaries/debug fetches after every tiny 48x64 training step.
def _watergan_env_positive_int(name, default):
    import os as _watergan_os
    try:
        value = int(_watergan_os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)

def _watergan_env_bool(name, default):
    import os as _watergan_os
    value = _watergan_os.environ.get(name, default)
    return str(value).strip().lower() not in ("0", "false", "no", "off")

_WATERGAN_IO_WORKERS = _watergan_env_positive_int(
    "WATERGAN_IO_WORKERS", 16
)
_WATERGAN_LOG_EVERY = _watergan_env_positive_int(
    "WATERGAN_LOG_EVERY", 10
)
_WATERGAN_THROTTLE_DIAGNOSTICS = _watergan_env_bool(
    "WATERGAN_THROTTLE_DIAGNOSTICS", "1"
)
_WATERGAN_IO_EXECUTOR = None

def _watergan_get_io_executor():
    global _WATERGAN_IO_EXECUTOR
    if _WATERGAN_IO_WORKERS <= 1:
        return None
    if _WATERGAN_IO_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        _WATERGAN_IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=_WATERGAN_IO_WORKERS
        )
    return _WATERGAN_IO_EXECUTOR

def _watergan_call_loader(task):
    loader, filename = task
    return loader(filename)

def _watergan_parallel_map(loader, filenames):
    filenames = list(filenames)
    executor = _watergan_get_io_executor()
    if executor is None:
        return [loader(filename) for filename in filenames]
    return list(executor.map(loader, filenames))

def _watergan_parallel_load_many(specs):
    lengths = []
    tasks = []
    for loader, filenames in specs:
        filenames = list(filenames)
        lengths.append(len(filenames))
        tasks.extend((loader, filename) for filename in filenames)
    executor = _watergan_get_io_executor()
    if executor is None:
        loaded = [_watergan_call_loader(task) for task in tasks]
    else:
        loaded = list(executor.map(_watergan_call_loader, tasks))
    result = []
    offset = 0
    for length in lengths:
        result.append(loaded[offset:offset + length])
        offset += length
    return result

def _watergan_should_log(counter):
    if not _WATERGAN_THROTTLE_DIAGNOSTICS:
        return True
    return counter == 1 or counter % _WATERGAN_LOG_EVERY == 0
'''

WATERGAN_TRAIN_LOAD_OLD = '''          if self.is_crop:
              air_batch = [self.read_img(air_batch_file) for air_batch_file in air_batch_files]
              water_batch = [self.read_img(water_batch_file) for water_batch_file in water_batch_files]
              depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
          else:
              air_batch = [scipy.misc.imread(air_batch_file) for air_batch_file in air_batch_files]
              water_batch = [scipy.misc.imread(water_batch_file) for water_batch_file in water_batch_files]
              depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]
'''

WATERGAN_TRAIN_LOAD_NEW = '''          if self.is_crop:
              air_batch, water_batch, depth_batch = _watergan_parallel_load_many((
                  (self.read_img, air_batch_files),
                  (self.read_img, water_batch_files),
                  (self.read_depth, depth_batch_files),
              ))
          else:
              air_batch, water_batch, depth_batch = _watergan_parallel_load_many((
                  (scipy.misc.imread, air_batch_files),
                  (scipy.misc.imread, water_batch_files),
                  (self.read_depth, depth_batch_files),
              ))
'''

WATERGAN_TRAIN_STEP_NEW = '''          should_log = _watergan_should_log(counter)
          d_feed = {
            self.z: batch_z,
            self.water_inputs: water_batch_images,
            self.air_inputs: air_batch_images,
            self.depth_inputs: depth_batch_images,
            self.R2: r2,
            self.R4: r4,
            self.R6: r6,
          }
          g_feed = {
            self.z: batch_z,
            self.air_inputs: air_batch_images,
            self.depth_inputs: depth_batch_images,
            self.R2: r2,
            self.R4: r4,
            self.R6: r6,
          }

          # Keep the original optimization schedule: one D update and two G
          # updates. Summaries and diagnostic forward passes are throttled.
          if should_log:
            _, summary_str = self.sess.run(
              [d_optim, self.d_sum], feed_dict=d_feed)
            self.writer.add_summary(summary_str, counter)
          else:
            self.sess.run(d_optim, feed_dict=d_feed)

          self.sess.run(g_optim, feed_dict=g_feed)
          if should_log:
            _, summary_str = self.sess.run(
              [g_optim, self.g_sum], feed_dict=g_feed)
            self.writer.add_summary(summary_str, counter)
          else:
            self.sess.run(g_optim, feed_dict=g_feed)

          counter += 1
          if should_log:
            loss_feed = dict(d_feed)
            errD_fake, errD_real, errG = self.sess.run(
              [self.d_loss_fake, self.d_loss_real, self.g_loss],
              feed_dict=loss_feed,
            )
            batch_number = idx // config.batch_size + 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, batch_number, water_batch_idxs,
                time.time() - start_time, errD_fake+errD_real, errG))
            debug_values = self.sess.run([
              'wc_generator/g_atten/g_eta_r:0',
              'wc_generator/g_atten/g_eta_g:0',
              'wc_generator/g_atten/g_eta_b:0',
              'wc_generator/g_vig/g_amp:0',
              'wc_generator/g_vig/g_c1:0',
              'wc_generator/g_vig/g_c2:0',
              'wc_generator/g_vig/g_c3:0',
            ])
            for debug_value in debug_values:
              print(debug_value)
'''

replacements = [
    ('flags.DEFINE_integer("train_size", np.inf,', 'flags.DEFINE_float("train_size", np.inf,'),
    ("flags.DEFINE_integer('train_size', np.inf,", "flags.DEFINE_float('train_size', np.inf,"),
    ("tf.pack(", "tf.stack("),
    ("tf.unpack(", "tf.unstack("),
    ("tf.mul(", "tf.multiply("),
    ("tf.sub(", "tf.subtract("),
    ("tf.neg(", "tf.negative("),
    ("tf.initialize_all_variables()", "tf.global_variables_initializer()"),
    ("tf.train.SummaryWriter", "tf.summary.FileWriter"),
    ("self.resuts_dir", "self.results_dir"),
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
batch_idx_division = re.compile(
    r"^(\s*\w*batch_idxs\s*=\s*)(.+?[^/])\s*/\s*config\.batch_size\s*$",
    flags=re.MULTILINE,
)
ceil_int_division = re.compile(
    r"int\(math\.ceil\(([^()\n]+?)\)\s*/\s*config\.batch_size\)"
)
imsave_function = re.compile(
    r"    def _watergan_imsave\(filename, arr\):\n"
    r"(?:        .*\n)+?"
    r"        image\.save\(filename\)\n",
    flags=re.MULTILINE,
)
WATERGAN_IMSAVE_FUNCTION = '''    def _watergan_imsave(filename, arr):
        arr = _watergan_np.asarray(arr)
        arr = _watergan_np.squeeze(arr)
        if arr.dtype.kind == 'f':
            values = arr
            if values.size:
                values = _watergan_np.nan_to_num(values)
                vmin = float(values.min())
                vmax = float(values.max())
                if vmin >= -1.0 and vmax <= 1.0:
                    values = (values + 1.0) * 127.5 if vmin < 0.0 else values * 255.0
            arr = _watergan_np.clip(values, 0, 255).astype(_watergan_np.uint8)
        elif arr.dtype != _watergan_np.uint8:
            arr = _watergan_np.clip(arr, 0, 255).astype(_watergan_np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        image = _PILImage.fromarray(arr)
        image.save(filename)
'''
scipy_misc_compat_block = re.compile(
    r"\n# Compatibility for SciPy versions where scipy\.misc image I/O was removed\.\n"
    r"try:\n"
    r"    scipy\.misc\.imread\n"
    r"except AttributeError:\n"
    r"(?:    .*\n)+?"
    r"    scipy\.misc\.imsave = _watergan_imsave\n",
    flags=re.MULTILINE,
)
read_depth_function = re.compile(
    r"^def read_depth\(.*?(?=^def\s|^class\s|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
direct_sio_loadmat = re.compile(
    r"(?<![A-Za-z0-9_])sio\.loadmat\(([^)\n]+)\)"
)
train_step_block = re.compile(
    r"^          # Update D network\n"
    r".*?"
    r"^            print\(self\.sess\.run\('wc_generator/g_vig/g_c3:0'\)\)\n",
    flags=re.MULTILINE | re.DOTALL,
)
train_load_triplet = re.compile(
    r"^(?P<indent>[ \t]*)air_batch\s*=\s*"
    r"\[(?P<air_loader>self\.read_img|scipy\.misc\.imread)"
    r"\(air_batch_file\)\s+for\s+air_batch_file\s+in\s+air_batch_files\]"
    r"[ \t]*\r?\n"
    r"(?P=indent)water_batch\s*=\s*"
    r"\[(?P<water_loader>self\.read_img|scipy\.misc\.imread)"
    r"\(water_batch_file\)\s+for\s+water_batch_file\s+in\s+water_batch_files\]"
    r"[ \t]*\r?\n"
    r"(?P=indent)depth_batch\s*=\s*"
    r"\[self\.read_depth\(depth_batch_file\)\s+for\s+depth_batch_file"
    r"\s+in\s+depth_batch_files\][ \t]*$",
    flags=re.MULTILINE,
)

def replace_train_load_triplet(match):
    indent = match.group("indent")
    inner = indent + "    "
    return (
        indent
        + "air_batch, water_batch, depth_batch = "
        + "_watergan_parallel_load_many((\n"
        + inner
        + "({0}, air_batch_files),\n".format(match.group("air_loader"))
        + inner
        + "({0}, water_batch_files),\n".format(match.group("water_loader"))
        + inner
        + "(self.read_depth, depth_batch_files),\n"
        + indent
        + "))"
    )

SCIPY_MISC_COMPAT = f'''

# Compatibility for SciPy versions where scipy.misc image I/O was removed.
try:
    scipy.misc.imread
except AttributeError:
    from PIL import Image as _PILImage
    import numpy as _watergan_np

    def _watergan_imread(filename, flatten=False, mode=None):
        with _PILImage.open(filename) as image:
            if flatten:
                image = image.convert('L')
            elif mode is not None:
                image = image.convert(mode)
            else:
                image = image.convert('RGB')
            return _watergan_np.asarray(image).copy()

    def _watergan_imresize(arr, size, interp='bilinear', mode=None):
        arr = _watergan_np.asarray(arr)
        arr = _watergan_np.squeeze(arr)
        if mode == 'F':
            image = _PILImage.fromarray(arr.astype(_watergan_np.float32), mode='F')
        elif arr.dtype.kind == 'f':
            values = arr
            if values.size:
                values = _watergan_np.nan_to_num(values)
                vmin = float(values.min())
                vmax = float(values.max())
                if vmin >= -1.0 and vmax <= 1.0:
                    values = (values + 1.0) * 127.5 if vmin < 0.0 else values * 255.0
            values = _watergan_np.clip(values, 0, 255).astype(_watergan_np.uint8)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[:, :, 0]
            image = _PILImage.fromarray(values)
        else:
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            image = _PILImage.fromarray(arr)
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
        resized = image.resize(new_size, resample)
        return _watergan_np.asarray(resized)

{WATERGAN_IMSAVE_FUNCTION}

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
    text = batch_idx_division.sub(r"\1int((\2) // config.batch_size)", text)
    text = ceil_int_division.sub(r"int(math.ceil(\1 / float(config.batch_size)))", text)
    text = re.sub(
        r"sample_batch_idxs\s*=\s*int\(\(self\.num_samples\s*/\)\s*//\s*config\.batch_size\)",
        r"sample_batch_idxs = int(self.num_samples // config.batch_size)",
        text,
    )
    if "def _watergan_imsave" in text:
        text = imsave_function.sub(WATERGAN_IMSAVE_FUNCTION, text)
    if "scipy.misc." in text:
        if "Compatibility for SciPy versions where scipy.misc image I/O was removed" in text:
            text = scipy_misc_compat_block.sub(SCIPY_MISC_COMPAT, text, count=1)
        elif "import scipy.misc" in text:
            text = text.replace("import scipy.misc", "import scipy.misc" + SCIPY_MISC_COMPAT, 1)
        elif "import scipy" in text:
            text = text.replace("import scipy", "import scipy" + SCIPY_MISC_COMPAT, 1)
        else:
            text = "import scipy.misc" + SCIPY_MISC_COMPAT + "\n" + text
    if path.name in ("modelmhl.py", "modeljamaica.py"):
        if "_watergan_list_depth_files" not in text:
            insert_after = "from utils import *"
            if insert_after in text:
                text = text.replace(insert_after, insert_after + WATERGAN_DEPTH_HELPERS, 1)
            else:
                text = WATERGAN_DEPTH_HELPERS + "\n" + text
        elif "_watergan_load_depth_file" not in text:
            class_marker = "class WGAN"
            if class_marker not in text:
                raise RuntimeError(
                    "Could not find class WGAN while adding PNG depth loader: %s"
                    % path
                )
            loader_start = WATERGAN_DEPTH_HELPERS.index(
                "def _watergan_load_depth_file"
            )
            loader_end = WATERGAN_DEPTH_HELPERS.index(
                "def _watergan_effective_train_batches"
            )
            loader = WATERGAN_DEPTH_HELPERS[loader_start:loader_end]
            text = text.replace(class_marker, loader + "\n" + class_marker, 1)
        text, _ = direct_sio_loadmat.subn(
            r"_watergan_load_depth_file(\1)",
            text,
        )
        if direct_sio_loadmat.search(text):
            raise RuntimeError(
                "Found an unpatched sio.loadmat call in %s" % path
            )
        if "_watergan_parallel_load_many" not in text:
            class_marker = "class WGAN"
            if class_marker not in text:
                raise RuntimeError(
                    "Could not find class WGAN while adding runtime helpers: %s"
                    % path
                )
            text = text.replace(
                class_marker,
                WATERGAN_RUNTIME_HELPERS + "\n" + class_marker,
                1,
            )
        text, train_load_replacements = train_load_triplet.subn(
            replace_train_load_triplet,
            text,
        )
        train_load_marker = (
            "air_batch, water_batch, depth_batch = "
            "_watergan_parallel_load_many(("
        )
        legacy_parallel_load_markers = (
            "air_batch = parallel_map(",
            "water_batch = parallel_map(",
            "depth_batch = parallel_map(",
        )
        has_legacy_parallel_load = all(
            marker in text for marker in legacy_parallel_load_markers
        )
        if train_load_marker not in text and not has_legacy_parallel_load:
            raise RuntimeError(
                "Could not find or patch WaterGAN training load statements "
                "in %s" % path
            )
        sample_loader_replacements = (
            (
                "[self.read_img_sample(sample_air_batch_file) "
                "for sample_air_batch_file in sample_air_batch_files]",
                "_watergan_parallel_map("
                "self.read_img_sample, sample_air_batch_files)",
            ),
            (
                "[self.read_img_sample(sample_water_batch_file) "
                "for sample_water_batch_file in sample_water_batch_files]",
                "_watergan_parallel_map("
                "self.read_img_sample, sample_water_batch_files)",
            ),
            (
                "[self.read_depth_small(sample_depth_batch_file) "
                "for sample_depth_batch_file in sample_depth_batch_files]",
                "_watergan_parallel_map("
                "self.read_depth_small, sample_depth_batch_files)",
            ),
            (
                "[self.read_depth_sample(sample_depth_batch_file) "
                "for sample_depth_batch_file in sample_depth_batch_files]",
                "_watergan_parallel_map("
                "self.read_depth_sample, sample_depth_batch_files)",
            ),
            (
                "[scipy.misc.imread(sample_air_batch_file) "
                "for sample_air_batch_file in sample_air_batch_files]",
                "_watergan_parallel_map("
                "scipy.misc.imread, sample_air_batch_files)",
            ),
            (
                "[scipy.misc.imread(sample_water_batch_file) "
                "for sample_water_batch_file in sample_water_batch_files]",
                "_watergan_parallel_map("
                "scipy.misc.imread, sample_water_batch_files)",
            ),
        )
        for sample_old, sample_new in sample_loader_replacements:
            text = text.replace(sample_old, sample_new)
        if "should_log = _watergan_should_log(counter)" not in text:
            text, train_step_replacements = train_step_block.subn(
                WATERGAN_TRAIN_STEP_NEW,
                text,
                count=1,
            )
            if train_step_replacements != 1:
                raise RuntimeError(
                    "Could not replace WaterGAN train step block in %s" % path
                )
        text = re.sub(
            r"depth_data\s*=\s*sorted\(glob\(os\.path\.join\(\s*\"\.\/data\",\s*config\.depth_dataset,\s*\"\*\.mat\"\s*\)\)\)",
            "depth_data = _watergan_list_depth_files(config.depth_dataset)",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"^(\s*)water_batch_idxs\s*=.*min\(.*len\(air_data\).*len\(water_data\).*config\.train_size.*$",
            r"\1water_batch_idxs = _watergan_effective_train_batches(air_data, depth_data, config)",
            text,
            flags=re.MULTILINE,
        )
        # Full-dataset inference only needs fake_*.png. Keep the legacy air
        # previews and depth MAT outputs opt-in so they cannot silently consume
        # hundreds of gigabytes during a sharded generation run.
        text = re.sub(
            r"^(\s*)scipy\.misc\.imsave\(out_name2\s*,\s*sample_im2\)\s*$",
            r"\1if os.environ.get('WATERGAN_SAVE_AUX_OUTPUTS', '0') == '1': scipy.misc.imsave(out_name2, sample_im2)",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^(\s*)sio\.savemat\(out_name3\s*,\s*\{'depth'\s*:\s*sample_im3\}\)\s*$",
            r"\1if os.environ.get('WATERGAN_SAVE_AUX_OUTPUTS', '0') == '1': sio.savemat(out_name3, {'depth': sample_im3})",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"water_data\[(randombatch\[[^\]]+\])\]",
            r"water_data[(\1) % len(water_data)]",
            text,
        )
    if path.name == "utils.py" and "def read_depth" in text:
        text = read_depth_function.sub(WATERGAN_READ_DEPTH + "\n\n", text, count=1)
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
echo "Remaining Python 2 style batch index divisions:"
grep -RInE "batch_idxs\\s*=.* / config\\.batch_size" "${WATERGAN_DIR}"/*.py || true

echo
echo "Remaining malformed division before ')' patterns:"
grep -RInE "/[[:space:]]*\\)" "${WATERGAN_DIR}"/*.py || true

echo
echo "WaterGAN PNG depth helper status:"
grep -RIn "_watergan_list_depth_files\\|_watergan_load_depth_file\\|def read_depth" \
  "${WATERGAN_DIR}"/*.py || true

echo
echo "Remaining direct model sio.loadmat calls:"
grep -RInE "(^|[^[:alnum:]_])sio\\.loadmat" \
  "${WATERGAN_DIR}"/modelmhl.py "${WATERGAN_DIR}"/modeljamaica.py || true

echo
echo "WaterGAN parallel I/O and logging status:"
grep -RIn "_watergan_parallel_load_many\\|_watergan_should_log(counter)" \
  "${WATERGAN_DIR}"/modelmhl.py "${WATERGAN_DIR}"/modeljamaica.py || true

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
