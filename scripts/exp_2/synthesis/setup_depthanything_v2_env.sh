#!/usr/bin/env bash
set -euo pipefail

# Create a standalone Depth Anything V2 environment and download the official
# repository/checkpoints. Run this once on the server before generating depth maps.

ENV_PREFIX="${ENV_PREFIX:-/media/SSD1/conda_envs/depthanything}"
SYN_PARENT="${SYN_PARENT:-/home/fcp/xcx/exp_2/syn}"
DEPTHANYTHING_DIR="${DEPTHANYTHING_DIR:-${SYN_PARENT}/Depth-Anything-V2}"
ENCODERS="${ENCODERS:-vits vitb}"

echo "========================================="
echo "Setup Depth Anything V2 environment"
echo "========================================="
echo "ENV_PREFIX:        ${ENV_PREFIX}"
echo "SYN_PARENT:        ${SYN_PARENT}"
echo "DEPTHANYTHING_DIR: ${DEPTHANYTHING_DIR}"
echo "ENCODERS:          ${ENCODERS}"
echo "========================================="

if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -p "${ENV_PREFIX}" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install opencv-python pillow numpy tqdm matplotlib safetensors huggingface_hub accelerate

mkdir -p "${SYN_PARENT}"
if [[ ! -d "${DEPTHANYTHING_DIR}/.git" ]]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git "${DEPTHANYTHING_DIR}"
else
  git -C "${DEPTHANYTHING_DIR}" pull
fi

if [[ -f "${DEPTHANYTHING_DIR}/requirements.txt" ]]; then
  python -m pip install -r "${DEPTHANYTHING_DIR}/requirements.txt"
fi

mkdir -p "${DEPTHANYTHING_DIR}/checkpoints"
for encoder in ${ENCODERS}; do
  case "${encoder}" in
    vits)
      repo="depth-anything/Depth-Anything-V2-Small"
      filename="depth_anything_v2_vits.pth"
      ;;
    vitb)
      repo="depth-anything/Depth-Anything-V2-Base"
      filename="depth_anything_v2_vitb.pth"
      ;;
    vitl)
      repo="depth-anything/Depth-Anything-V2-Large"
      filename="depth_anything_v2_vitl.pth"
      ;;
    *)
      echo "Warning: skip unknown encoder ${encoder}" >&2
      continue
      ;;
  esac
  echo "Download ${encoder}: ${repo}/${filename}"
  hf download "${repo}" "${filename}" --local-dir "${DEPTHANYTHING_DIR}/checkpoints"
done

python - <<'PY'
import cv2
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("cv2:", cv2.__version__)
PY

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"
