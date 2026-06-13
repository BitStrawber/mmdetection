# J13: Real underwater MAE pretraining + ConvNeXt-Tiny + Cascade R-CNN -> RUOD.
# Requires MMPreTrain with ConvNeXt registered.

_base_ = './cascade-rcnn_r50_fpn_2x_ruod_j2.py'

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

pretrained = '../pretrained_weights/j13_realuw_spark_convnextv2_tiny_backbone.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]))

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=False)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.7,
        decay_type='layer_wise',
        num_layers=6),
    optimizer=dict(
        type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05))

load_from = None
resume = False
