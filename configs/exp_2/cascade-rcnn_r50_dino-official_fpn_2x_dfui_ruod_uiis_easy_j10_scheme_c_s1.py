# J10 DINO-official source comparison S1: DFUI + RUOD easy + UIIS easy.

_base_ = './cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py'

pretrained = '../pretrained_weights/dino_rn50_official_100e_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
