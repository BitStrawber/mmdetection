# Official ImageNet DINO ResNet-50 100e -> Cascade R-CNN -> RUOD.

_base_ = './cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py'

pretrained = '../pretrained_weights/dino_rn50_official_100e_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
