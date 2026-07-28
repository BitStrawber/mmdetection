_base_ = [
    '../cascade-rcnn_vit-small_dino-realuw_fpn_2x_ruod_j14.py'
]

# Keep the established ViT-S/ViTDet model, optimizer, and RUOD pipeline while
# making the 24-epoch schedule explicit for controlled-pretraining comparisons.
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
