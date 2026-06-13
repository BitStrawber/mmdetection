# RealUW Tri-Pretrain S1 Configs

These files define the S1 self-supervised pretraining stage for the five
Tri-pretrain tasks. All tasks use the full RealUW selected image pool. COCO
annotations are only used before this stage to select and locate images.

## Shared Dataset

Build once:

```bash
python tools/build_realuw_ssl_dataset.py \
  --preset exp2_bbox20pct \
  --out-root /media/HDD1/XCX/exp_2/REALUW_SSL \
  --val-ratio 0 \
  --write-imagefolder \
  --overwrite
```

Outputs used by these tasks:

```text
/media/HDD1/XCX/exp_2/REALUW_SSL/meta/train.txt
/media/HDD1/XCX/exp_2/REALUW_SSL/imagefolder/train/realuw/
```

`meta/train.txt` is useful for MMPreTrain custom datasets. The ImageFolder
layout is useful for MMPreTrain official ImageNet-style configs and the
facebookresearch/dino script.

## Final S1 Choices

| Task | Backbone | Strategy | Reference config/script | Local wrapper |
| --- | --- | --- | --- | --- |
| J6 | ResNet-50 | SparK, MAE-like masked modeling | `keyu-tian/SparK/pretrain/main.py --model resnet50` | launcher only |
| J7 | ResNet-50 | DINO self-distillation | `facebookresearch/dino/main_dino.py --arch resnet50` | launcher only |
| J11 | ViT-Base | MAE | `mmpretrain/configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py` | `realuw_ssl_mae_vit-base-p16_8xb512-amp-coslr-300e.py` |
| J12 | SwinV2-Base | SimMIM/MixMIM masked modeling | requires a verified SwinV2-Base masked-modeling config with `CONFIG=...` | no default yet |
| J13 | ConvNeXtV2-Tiny | SparK/FCMAE-like masked modeling | `mmpretrain/configs/spark/spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k.py` | `realuw_ssl_spark-convnextv2-tiny_16xb256-amp-coslr-800e.py` |

Upstream config snapshots are stored under:

```text
configs/exp_2/tri_pretrain/upstream/
```

They are for traceability and comparison. Runtime uses the local wrappers and
task config files in this repository.

## Run S1

Single task:

```bash
EXP_ID=j11 \
MMPRETRAIN_DIR=~/xcx/exp_2/mmpretrain \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

All tasks sequentially:

```bash
MMPRETRAIN_DIR=~/xcx/exp_2/mmpretrain \
DINO_DIR=~/xcx/exp_2/dino \
SPARK_DIR=~/xcx/exp_2/SparK \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1_all.sh
```

Run one complete task, including S1, backbone conversion and downstream RUOD:

```bash
EXP_ID=j11 \
MMPRETRAIN_DIR=~/xcx/exp_2/mmpretrain \
SPARK_DIR=~/xcx/exp_2/SparK \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_full_task.sh
```

Task-specific wrappers are also available:

```bash
bash scripts/exp_2/tri_pretrain/run_exp_2_j6.sh
bash scripts/exp_2/tri_pretrain/run_exp_2_j7.sh
bash scripts/exp_2/tri_pretrain/run_exp_2_j11.sh
bash scripts/exp_2/tri_pretrain/run_exp_2_j12.sh
bash scripts/exp_2/tri_pretrain/run_exp_2_j13.sh
```

## J12 Config Resolution

The cloned MMPreTrain repo contains Swin-Base SimMIM configs, for example:

```text
configs/exp_2/tri_pretrain/upstream/mmpretrain/simmim/simmim_swin-base-w6_16xb128-amp-coslr-100e_in1k-192px.py
```

These are kept only as references because they do not strictly match the
SwinV2-Base downstream model. For J12, pass a verified SwinV2-Base
masked-modeling config explicitly:

```bash
EXP_ID=j12 CONFIG=/path/to/mmpretrain/configs/.../your_swinv2_config.py bash ...
```

## J6 SparK Resolution

J6 does not use a local MMPreTrain wrapper. It runs the official SparK
pretraining entry with the recommended ResNet-50 masked-modeling setup:

```bash
EXP_ID=j6 \
SPARK_DIR=~/xcx/exp_2/SparK \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

Default important options:

```text
--model resnet50
--bs 4096
--ep 1600
--wp_ep 40
--base_lr 2e-4
--wd 0.04
--wde 0.2
--mask 0.6
--input_size 224
--opt lamb
```

The expected exported S1 checkpoint is:

```text
work_dirs/tri_pretrain/j6_realuw_spark_resnet50/resnet50_1kpretrained_timm_style.pth
```

## J13 ConvNeXtV2 SparK

J13 uses MMPreTrain SparK ConvNeXtV2-Tiny:

```bash
EXP_ID=j13 \
MMPRETRAIN_DIR=~/xcx/exp_2/mmpretrain \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

Default important options:

```text
config: configs/exp_2/mmpretrain/realuw_ssl_spark-convnextv2-tiny_16xb256-amp-coslr-800e.py
batch_size_per_gpu: 256
epochs: 800
warmup_epochs: 20
mask_ratio: 0.6
optimizer: Lamb
```

The expected exported checkpoint is searched from:

```text
work_dirs/tri_pretrain/j13_realuw_spark_convnextv2_tiny/
```

## S2 Principle

After S1, export the trained backbone weights and keep RUOD downstream settings
fixed for fair comparison:

```text
same RUOD train/val/test
same detector schedule
same evaluation metrics
only S1 backbone initialization changes
```

Generic conversion command:

```bash
python tools/convert_ssl_backbone_to_mmdet.py \
  --checkpoint work_dirs/tri_pretrain/j11_realuw_mae_vit_base/latest.pth \
  --out ../pretrained_weights/j11_realuw_mae_vit_base_backbone.pth
```

Then use the converted checkpoint in the downstream detector config:

```python
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../pretrained_weights/j11_realuw_mae_vit_base_backbone.pth')))
```
