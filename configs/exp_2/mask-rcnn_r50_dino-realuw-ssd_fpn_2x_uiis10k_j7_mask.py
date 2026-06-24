# J7 SSD: RealUW DINO ResNet-50 pretraining -> Mask R-CNN -> UIIS10K.

_base_ = './mask-rcnn_r50_dino-realuw_fpn_2x_uiis10k_j7_mask.py'

pretrained = '../pretrained_weights/j7_realuw_dino_resnet50_ssd100e_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
