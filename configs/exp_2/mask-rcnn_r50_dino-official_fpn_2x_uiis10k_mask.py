# Official ImageNet DINO ResNet-50 100e -> Mask R-CNN -> UIIS10K.

_base_ = './mask-rcnn_r50_dino_fpn_2x_uiis10k_j4_mask.py'

pretrained = '../pretrained_weights/dino_rn50_official_100e_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
