# ImageNet-1K DFUI Stage Configurations

These are versioned copies of the previously validated J10 ResNet-50 DFUI
stage configurations from fcp. They are consumed by
`run_imagenet1k_dfui_then_ruod_det.sh` for the ImageNet-1K 100-epoch DINO
experiments on fuping.

The ViT-S stage configuration remains under `configs/exp_2/tri_pretrain` to
preserve its original relative inheritance path. Its required ViTDet base
configuration is stored at `configs/exp_2/cascade-rcnn_vit-small_dino-realuw_fpn_2x_ruod_j14.py`.

The launcher overrides dataset roots, source checkpoints, output work
directories, epoch counts, and class counts at runtime. The checked-in files
therefore preserve architecture and optimizer details without hard-coding the
fcp filesystem into a fuping run.
