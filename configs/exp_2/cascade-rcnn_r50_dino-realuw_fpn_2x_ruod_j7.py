# J7: Real underwater DINO pretraining + ResNet-50 + Cascade R-CNN -> RUOD.

_base_ = './cascade-rcnn_r50_fpn_2x_ruod_j2.py'

pretrained = '../pretrained_weights/j7_realuw_dino_resnet50_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
