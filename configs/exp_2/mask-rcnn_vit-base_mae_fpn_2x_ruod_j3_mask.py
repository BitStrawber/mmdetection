# J3: ImageNet MAE + ViT-Base + Mask R-CNN -> RUOD (2 GPU, 总BS=12)
# 使用MAE预训练的ViT-Base权重

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py',
]

# 导入ViTDet模块 (官方方式)
custom_imports = dict(imports=['projects.ViTDet.vitdet'])

# Norm配置 (官方标准)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# 模型配置 (完全参考官方ViTDet)
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
        bbox_head=dict(
            _delete_=True,
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=10),
        mask_head=dict(
            type='FCNMaskHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=10)))

# 数据集路径配置 (绝对路径)
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/annotations/'

# 2 GPU配置
train_dataloader = dict(
    batch_size=4, 
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

val_evaluator = dict(
    ann_file=ann_root + 'instances_val.json',
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# LayerDecay优化器 (官方标准)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
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