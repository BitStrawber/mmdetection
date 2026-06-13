# J6 S1: RealUW + SparK + ResNet-50.
# Source this file before running run_exp_2_tri_pretrain_s1.sh, or let the
# launcher use the same defaults.

export EXP_ID="${EXP_ID:-j6}"
export SPARK_MODEL="${SPARK_MODEL:-resnet50}"
export SPARK_BS="${SPARK_BS:-4096}"
export SPARK_EPOCHS="${SPARK_EPOCHS:-1600}"
export SPARK_WARMUP_EPOCHS="${SPARK_WARMUP_EPOCHS:-40}"
export SPARK_BASE_LR="${SPARK_BASE_LR:-2e-4}"
export SPARK_WEIGHT_DECAY="${SPARK_WEIGHT_DECAY:-0.04}"
export SPARK_WEIGHT_DECAY_END="${SPARK_WEIGHT_DECAY_END:-0.2}"
export SPARK_MASK_RATIO="${SPARK_MASK_RATIO:-0.6}"
export SPARK_INPUT_SIZE="${SPARK_INPUT_SIZE:-224}"
export SPARK_OPTIMIZER="${SPARK_OPTIMIZER:-lamb}"
export SPARK_WORKERS="${SPARK_WORKERS:-8}"
