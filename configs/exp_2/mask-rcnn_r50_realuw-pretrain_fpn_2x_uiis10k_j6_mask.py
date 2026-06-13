# Optional J6 mask evaluation: RealUW SparK self-supervised pretraining
# + ResNet-50 -> Mask R-CNN on UIIS10K.

_base_ = './mask-rcnn_r50_fpn_2x_uiis10k_j2_mask.py'

pretrained = '../pretrained_weights/j6_realuw_spark_resnet50_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

load_from = None
resume = False
