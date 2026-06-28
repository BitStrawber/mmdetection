#!/usr/bin/env bash
set -euo pipefail

# Clone and prepare the CompVis/stable-diffusion baseline environment.
# This script only prepares the project and environment. Dataset generation is
# handled by run_sd_img2img_underwater_generate*.sh so outputs keep the same
# ImageNet-style folders, manifests, progress bars, and shard logic as the other
# synthesis methods.
#
# Usage:
#   bash scripts/exp_2/synthesis/setup_compvis_stable_diffusion_env.sh
#
# Optional overrides:
#   SD_DIR=/home/fcp/xcx/exp_2/syn/stable-diffusion
#   ENV_PREFIX=/media/SSD1/conda_envs/stable_diffusion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SD_DIR="${SD_DIR:-/home/fcp/xcx/exp_2/syn/stable-diffusion}"
ENV_PREFIX="${ENV_PREFIX:-/media/SSD1/conda_envs/stable_diffusion}"
REPO_URL="${REPO_URL:-https://github.com/CompVis/stable-diffusion.git}"

echo "========================================="
echo "Setup CompVis Stable Diffusion baseline"
echo "========================================="
echo "SD_DIR:     ${SD_DIR}"
echo "ENV_PREFIX: ${ENV_PREFIX}"
echo "REPO_URL:   ${REPO_URL}"
echo "========================================="
echo

if [[ ! -d "${SD_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SD_DIR}")"
  echo "Step 1/4: Clone CompVis/stable-diffusion"
  git clone "${REPO_URL}" "${SD_DIR}"
else
  echo "Step 1/4: Repo already exists, skip clone"
  (
    cd "${SD_DIR}"
    git remote -v | head -n 2
  )
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo
  echo "Step 2/4: Create conda environment at ${ENV_PREFIX}"
  # The original environment.yaml pins old packages. Use it first so the repo
  # remains reproducible, then install diffusers tooling used by our batch
  # wrapper in the same environment.
  conda env create -p "${ENV_PREFIX}" -f "${SD_DIR}/environment.yaml"
else
  echo
  echo "Step 2/4: Conda environment already exists, skip create"
fi

echo
echo "Step 3/4: Install batch-generation dependencies"
conda run -p "${ENV_PREFIX}" python -m pip install \
  "huggingface-hub==0.20.3" \
  "diffusers==0.25.1" \
  "transformers==4.36.2" \
  "accelerate==0.25.0" \
  "safetensors==0.4.2" \
  "tqdm" \
  "pillow"

echo
echo "Step 4/4: Verify imports"
conda run -p "${ENV_PREFIX}" python - <<'PY'
import torch
import diffusers
import transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
PY

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"
echo
echo "For original CompVis scripts, place/download a checkpoint such as:"
echo "  ${SD_DIR}/models/ldm/stable-diffusion-v1/model.ckpt"
echo
echo "For this repo's ImageNet batch img2img baseline, use:"
echo "  SPLIT=train LIMIT=100 GPU=2 bash scripts/exp_2/synthesis/run_sd_img2img_underwater_generate.sh"
