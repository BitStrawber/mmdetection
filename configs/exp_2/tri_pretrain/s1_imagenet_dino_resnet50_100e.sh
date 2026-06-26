# ImageNet-1K + facebookresearch/DINO + ResNet-50, 100 epochs.
#
# This mirrors the official facebookresearch/dino ResNet-50 ImageNet command:
#   main_dino.py --arch resnet50 --optimizer sgd --lr 0.03
#     --weight_decay 1e-4 --weight_decay_end 1e-4
#     --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14
#     --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
#
# The data path is supplied by REALUW_SSL_ROOT/imagefolder/train through
# run_exp_2_tri_pretrain_s1.sh.

export EXP_ID="${EXP_ID:-j7}"
export DINO_NAME="${DINO_NAME:-imagenet_dino_resnet50_100e}"
export DINO_ARCH="${DINO_ARCH:-resnet50}"
export DINO_OPTIMIZER="${DINO_OPTIMIZER:-sgd}"
export DINO_LR="${DINO_LR:-0.03}"
export DINO_WEIGHT_DECAY="${DINO_WEIGHT_DECAY:-1e-4}"
export DINO_WEIGHT_DECAY_END="${DINO_WEIGHT_DECAY_END:-1e-4}"
export DINO_GLOBAL_CROPS_SCALE="${DINO_GLOBAL_CROPS_SCALE:-0.14 1}"
export DINO_LOCAL_CROPS_SCALE="${DINO_LOCAL_CROPS_SCALE:-0.05 0.14}"
export DINO_EPOCHS="${DINO_EPOCHS:-100}"
export DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
export DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
export DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-50}"
