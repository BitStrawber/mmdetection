# Tri-Pretrain Config Index

This directory keeps the experiment-level S1 configuration files for RealUW
self-supervised pretraining.

## Runtime Task Configs

These shell config files are sourced by:

```bash
scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

| Task | Runtime config |
| --- | --- |
| J6 | `s1_j6_spark_resnet50_realuw.sh` |
| J7 | `s1_j7_dino_resnet50_realuw.sh` |
| J11 | `s1_j11_mae_vit_base_realuw.sh` |
| J12 | `s1_j12_simmim_swin_base_realuw.sh` |
| J13 | `s1_j13_spark_convnextv2_tiny_realuw.sh` |

J12 intentionally has no default upstream config because the cloned MMPreTrain
repo provides Swin-Base SimMIM configs, not a strict SwinV2-Base masked-modeling
config.

## MMPreTrain RealUW Wrappers

Executable MMPreTrain wrapper configs are stored in:

```text
configs/exp_2/mmpretrain/
```

Current runnable wrappers:

```text
realuw_ssl_mae_vit-base-p16_8xb512-amp-coslr-300e.py
realuw_ssl_spark-convnextv2-tiny_16xb256-amp-coslr-800e.py
```

There is intentionally no J12 SwinV2 wrapper until a verified SwinV2-Base
masked-modeling config is available.

## Upstream Snapshots

The `upstream/` directory stores exact snapshots from cloned external repos for
traceability:

```text
upstream/dino/main_dino.py
upstream/SparK/arg_util.py
upstream/mmpretrain/mae/...
upstream/mmpretrain/simmim/...
upstream/mmpretrain/mixmim/...
upstream/mmpretrain/spark/...
```

These snapshots are not executed directly.
