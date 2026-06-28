# J7 continue: official ImageNet DINO ResNet-50 -> RealUW domain-adaptive DINO.
#
# This starts a new RealUW DINO run initialized from the official ImageNet DINO
# checkpoint. It does not resume ImageNet optimizer/scheduler state.
#
# Compared with from-scratch J7, the learning rate is reduced to avoid washing
# out ImageNet object-level representations during short domain adaptation.

export EXP_ID="${EXP_ID:-j7}"
export DINO_ARCH="${DINO_ARCH:-resnet50}"
export DINO_OPTIMIZER="${DINO_OPTIMIZER:-sgd}"
export DINO_LR="${DINO_LR:-0.003}"
export DINO_MIN_LR="${DINO_MIN_LR:-1e-6}"
export DINO_WEIGHT_DECAY="${DINO_WEIGHT_DECAY:-1e-4}"
export DINO_WEIGHT_DECAY_END="${DINO_WEIGHT_DECAY_END:-1e-4}"
export DINO_MOMENTUM_TEACHER="${DINO_MOMENTUM_TEACHER:-0.996}"
export DINO_WARMUP_EPOCHS="${DINO_WARMUP_EPOCHS:-2}"
export DINO_FREEZE_LAST_LAYER="${DINO_FREEZE_LAST_LAYER:-1}"
export DINO_GLOBAL_CROPS_SCALE="${DINO_GLOBAL_CROPS_SCALE:-0.14 1}"
export DINO_LOCAL_CROPS_NUMBER="${DINO_LOCAL_CROPS_NUMBER:-8}"
export DINO_LOCAL_CROPS_SCALE="${DINO_LOCAL_CROPS_SCALE:-0.05 0.14}"
export DINO_EPOCHS="${DINO_EPOCHS:-20}"
export DINO_BATCH_SIZE_PER_GPU="${DINO_BATCH_SIZE_PER_GPU:-64}"
export DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-10}"
export DINO_SAVECKP_FREQ="${DINO_SAVECKP_FREQ:-10}"
