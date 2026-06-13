# J11 Mask: Real underwater MAE pretraining + ViT-Base + Mask R-CNN -> UIIS10K.

_base_ = './mask-rcnn_vit-base_mae_fpn_2x_uiis10k_j3_mask.py'

pretrained = '../pretrained_weights/j11_realuw_mae_vit_base_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
