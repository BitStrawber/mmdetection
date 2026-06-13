# J6: RealUW SparK self-supervised pretraining + ResNet-50
#     -> Cascade R-CNN on RUOD.

_base_ = './cascade-rcnn_r50_fpn_2x_ruod_j2.py'

pretrained = '../pretrained_weights/j6_realuw_spark_resnet50_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
