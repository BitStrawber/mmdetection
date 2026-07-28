_base_ = [
    '../mask-rcnn_vit-small_dino-realuw_fpn_2x_uiis10k_j14_mask.py'
]

# Use the same 24-epoch ViT-S optimization budget as RUOD. The inherited model
# remains Mask R-CNN and the inherited optimizer remains layer-decayed AdamW.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1),
]
