# J3: ImageNet MAE + ViT-Base + Mask R-CNN -> RUOD (2 GPU, 总BS=12)
# 使用MAE预训练的ViT-Base权重

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
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
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained_weights/mae_pretrain_vit_base.pth')),
    neck=dict(
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10)))

# 数据集路径配置
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'

# 修改backbone权重为MAE ViT
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained_weights/mae_pretrain_vit_base.pth')),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10)))

# 2 GPU配置 (总BS=12)
train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(backbone=dict(lr_mult=0.1)))

val_evaluator = dict(
    ann_file='annotations/instances_val.json',
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator