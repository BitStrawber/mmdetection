# J3 Mask: ImageNet MAE + ViT-Base + Mask R-CNN -> UIIS10K (2 GPU)
# ViT-Base + SimpleFPN + StandardRoIHead with mask

_base_ = ['../mask_rcnn/mask-rcnn_r50_fpn_2x_uiis10k.py']

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../pretrained_weights/mae_pretrain_vit_base.pth')),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256, norm_cfg=norm_cfg, num_classes=10),
        mask_head=dict(type='FCNMaskHead', num_convs=4, in_channels=256, conv_out_channels=256, num_classes=10)))

# batch_size=1 for ViT OOM
train_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=False)
test_dataloader = val_dataloader

# 100 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1))
