# J10 scheme C S1: Cascade-supervised underwater backbone adaptation.
#
# Goal:
#   ImageNet ResNet50 -> underwater detection-supervised adaptation -> backbone only.
#
# Difference from the old J10 S1:
#   - S1 still uses Cascade R-CNN detection supervision.
#   - The default setting is more conservative: frozen_stages=2 and smaller LR.
#   - S2 should load only the extracted backbone weights, not neck/RPN/ROI heads.

_base_ = './cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_s1.py'

# Center setting for the first scheme-C run.
# Override these through --cfg-options when sweeping:
#   model.backbone.frozen_stages=1/2/3
#   optim_wrapper.optimizer.lr=0.00375/0.001875/0.0009375
#   train_cfg.max_epochs=24/48/72
#   param_scheduler.1.end=24/48/72
#   param_scheduler.1.milestones=[16,22]/[32,44]/[48,66]
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001875, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        max_keep_ckpts=10))

load_from = None
resume = False
