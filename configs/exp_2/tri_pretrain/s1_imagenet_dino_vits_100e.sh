# ImageNet-1K + facebookresearch/DINO + ViT-Small/16, 100 epochs.
#
# This mirrors the J14 RealUW ViT-S DINO recipe and only changes the output
# experiment name. The data path is supplied by REALUW_SSL_ROOT/imagefolder/train
# through run_exp_2_tri_pretrain_s1.sh.

export EXP_ID="${EXP_ID:-j14}"
export DINO_NAME="${DINO_NAME:-imagenet_dino_vits_100e}"
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
