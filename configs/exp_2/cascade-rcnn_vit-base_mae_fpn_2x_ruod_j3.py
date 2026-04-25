# J3: ImageNet MAE + ViT-Base + Cascade R-CNN -> RUOD (2 GPU, 总BS=12)
# 使用MAE预训练的ViT-Base权重

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

# 数据集路径配置 (与J2/J4保持一致)
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'

# 导入ViTDet模块
custom_imports = dict(imports=['projects.ViTDet.vitdet'])

# 修改backbone为ViT (完全替换整个backbone)
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=800,  # 减小image size
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN', requires_grad=True),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='../pretrained_weights/mae_pretrain_vit_base.pth')),
    # 替换neck为适配ViT的SimpleFPN
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5))

# 2 GPU配置 (与J2/J4保持一致)
train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file='annotations/instances_train.json'))
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file='annotations/instances_val.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file='annotations/instances_val.json')
test_evaluator = val_evaluator