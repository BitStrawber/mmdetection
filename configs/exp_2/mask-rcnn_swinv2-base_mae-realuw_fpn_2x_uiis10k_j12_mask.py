# J12 Mask: Real underwater MAE pretraining + SwinV2-Base + Mask R-CNN -> UIIS10K.
# Requires MMPreTrain with SwinTransformerV2 registered.

_base_ = './mask-rcnn_r50_fpn_2x_uiis10k_j2_mask.py'

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

pretrained = '../pretrained_weights/j12_realuw_simmim_swinv2_base_backbone.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.SwinTransformerV2',
        arch='base',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]))

train_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=False)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

load_from = None
resume = False
