# J12 S1: RealUW + SimMIM/MixMIM + Swin/SwinV2-Base.
# Set CONFIG to a verified SwinV2-Base masked-modeling config. The cloned
# MMPreTrain repo has Swin-Base SimMIM snapshots under upstream/ for reference,
# but those are not used by default because they do not strictly match SwinV2.

export EXP_ID="${EXP_ID:-j12}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
