#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_DIR="${SOURCE_DIR:-/media/HDD1/XCX/exp_2/synthetic_imagenet/uwdf/source/train}"
TEST_IMG="${TEST_IMG:-}"
OUT_DIR="${OUT_DIR:-/media/SSD1/XCX/exp_2/synthesis_work/stable_diffusion_diffusers/vae_text_smoke}"
MODEL="${MODEL:-runwayml/stable-diffusion-v1-5}"
GPU="${GPU:-2}"
STEPS="${STEPS:-20}"
STRENGTH="${STRENGTH:-0.35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEED="${SEED:-2026}"
HF_HOME="${HF_HOME:-/media/SSD1/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
ENABLE_ATTENTION_SLICING="${ENABLE_ATTENTION_SLICING:-0}"

PROMPT="${PROMPT:-a realistic underwater photograph of the same object and scene, blue-green water, underwater haze, natural color attenuation, low contrast, realistic lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-cartoon, painting, illustration, deformed object, extra objects, fish, coral, diver, text, watermark, blurry, low quality}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE

echo "========================================="
echo "Stable Diffusion VAE latent + text check"
echo "========================================="
echo "SOURCE_DIR:    ${SOURCE_DIR}"
echo "TEST_IMG:      ${TEST_IMG:-<auto>}"
echo "OUT_DIR:       ${OUT_DIR}"
echo "MODEL:         ${MODEL}"
echo "GPU:           ${GPU}"
echo "SIZE:          ${WIDTH}x${HEIGHT}"
echo "STEPS:         ${STEPS}"
echo "STRENGTH:      ${STRENGTH}"
echo "GUIDANCE:      ${GUIDANCE_SCALE}"
echo "SEED:          ${SEED}"
echo "HF_HOME:       ${HF_HOME}"
echo "PROMPT:        ${PROMPT}"
echo "NEG_PROMPT:    ${NEGATIVE_PROMPT}"
echo "========================================="

if [[ -z "${TEST_IMG}" ]]; then
  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Error: SOURCE_DIR not found: ${SOURCE_DIR}" >&2
    exit 1
  fi
  TEST_IMG="$(
    find "${SOURCE_DIR}" \
      \( -type f -o -type l \) \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) \
      | sort \
      | head -n 1
  )"
fi

if [[ -z "${TEST_IMG}" ]]; then
  echo "Error: no test image found under ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ ! -e "${TEST_IMG}" ]]; then
  echo "Error: TEST_IMG not found: ${TEST_IMG}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo
echo "Selected image:"
echo "  ${TEST_IMG}"
echo "Resolved path:"
readlink -f "${TEST_IMG}" || true

CUDA_VISIBLE_DEVICES="${GPU}" \
MODEL="${MODEL}" \
TEST_IMG="${TEST_IMG}" \
OUT_DIR="${OUT_DIR}" \
PROMPT="${PROMPT}" \
NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
WIDTH="${WIDTH}" \
HEIGHT="${HEIGHT}" \
STEPS="${STEPS}" \
STRENGTH="${STRENGTH}" \
GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
SEED="${SEED}" \
ENABLE_ATTENTION_SLICING="${ENABLE_ATTENTION_SLICING}" \
python - <<'PY'
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionImg2ImgPipeline

model = os.environ["MODEL"]
image_path = Path(os.environ["TEST_IMG"])
out_dir = Path(os.environ["OUT_DIR"])
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEGATIVE_PROMPT"]
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
steps = int(os.environ["STEPS"])
strength = float(os.environ["STRENGTH"])
guidance_scale = float(os.environ["GUIDANCE_SCALE"])
seed = int(os.environ["SEED"])
enable_attention_slicing = os.environ.get("ENABLE_ATTENTION_SLICING", "0") == "1"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("input:", image_path)
print("resolved:", image_path.resolve())
print("model:", model)

init_image = Image.open(image_path).convert("RGB")
init_image = ImageOps.fit(
    init_image,
    (width, height),
    method=Image.Resampling.BICUBIC,
    centering=(0.5, 0.5),
)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to("cuda")
if enable_attention_slicing:
    pipe.enable_attention_slicing()

generator = torch.Generator(device="cuda").manual_seed(seed)

with torch.inference_mode():
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )

out = out_dir / (
    f"{image_path.stem}_underwater_sd15"
    f"_s{str(strength).replace('.', '')}"
    f"_g{str(guidance_scale).replace('.', '')}"
    f"_steps{steps}.png"
)
result.images[0].save(out)

print("output:", out)
PY

echo
echo "Generated files:"
find "${OUT_DIR}" -maxdepth 1 -type f -name '*.png' | sort
