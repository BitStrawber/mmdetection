# J10 DINO-route S1:
# DINO-pretrained ResNet-50 -> expanded underwater DFUI source.

_base_ = './cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py'

pretrained = '../pretrained_weights/dino_resnet50_pretrain.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
