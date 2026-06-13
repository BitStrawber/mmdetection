# J11: Real underwater MAE pretraining + ViT-Base + Cascade R-CNN -> RUOD.

_base_ = './cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py'

pretrained = '../pretrained_weights/j11_realuw_mae_vit_base_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
