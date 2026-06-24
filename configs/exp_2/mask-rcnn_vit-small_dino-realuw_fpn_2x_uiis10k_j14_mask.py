# J14: RealUW DINO ViT-Small/16 pretraining -> ViTDet Mask R-CNN -> UIIS10K.
# The backbone shape follows facebookresearch/dino vit_small:
# embed_dim=384, depth=12, num_heads=6, patch_size=16.

_base_ = ['./mask-rcnn_vit-base_mae_fpn_2x_uiis10k_j3_mask.py']

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

pretrained = '../pretrained_weights/j14_realuw_dino_vits_100e_backbone.pth'

backbone_norm_cfg = dict(type='LN', requires_grad=True, eps=1e-6)
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=1024,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        use_rel_pos=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=384,
        in_channels=[96, 192, 384, 384],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=10),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=10)))

train_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=False)
test_dataloader = val_dataloader

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.7,
        decay_type='layer_wise',
        num_layers=12),
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1))

load_from = None
resume = False
