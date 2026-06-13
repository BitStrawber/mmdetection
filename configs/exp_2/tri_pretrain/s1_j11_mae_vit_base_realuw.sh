# J11 S1: RealUW + MAE + ViT-Base.

export EXP_ID="${EXP_ID:-j11}"
export CONFIG="${CONFIG:-configs/exp_2/mmpretrain/realuw_ssl_mae_vit-base-p16_8xb512-amp-coslr-300e.py}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
