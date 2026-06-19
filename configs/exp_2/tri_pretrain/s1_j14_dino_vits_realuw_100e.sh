# J14 S1: RealUW + facebookresearch/DINO + ViT-Small/16, 100 epochs.
# Based on the official facebookresearch/dino README command:
#   python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
#     --arch vit_small --data_path /path/to/imagenet/train \
#     --output_dir /path/to/saving_dir
#
# The official command relies on main_dino.py defaults for ViT-S:
#   optimizer=adamw, lr=0.0005, weight_decay=0.04,
#   weight_decay_end=0.4, global_crops_scale=0.4 1,
#   local_crops_scale=0.05 0.4, batch_size_per_gpu=64,
#   epochs=100.

export EXP_ID="${EXP_ID:-j14}"
export DINO_NAME="${DINO_NAME:-j14_realuw_dino_vits_100e}"
export DINO_ARCH="${DINO_ARCH:-vit_small}"
export DINO_OPTIMIZER="${DINO_OPTIMIZER:-adamw}"
export DINO_LR="${DINO_LR:-0.0005}"
export DINO_WEIGHT_DECAY="${DINO_WEIGHT_DECAY:-0.04}"
export DINO_WEIGHT_DECAY_END="${DINO_WEIGHT_DECAY_END:-0.4}"
export DINO_GLOBAL_CROPS_SCALE="${DINO_GLOBAL_CROPS_SCALE:-0.4 1}"
export DINO_LOCAL_CROPS_SCALE="${DINO_LOCAL_CROPS_SCALE:-0.05 0.4}"
export DINO_EPOCHS="${DINO_EPOCHS:-100}"
export DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
export DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
export DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-50}"
