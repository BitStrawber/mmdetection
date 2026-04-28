# J10 Stage 1: ImageNet预训练 → DFUI微调 (48 epoch)
# 使用ImageNet torchvision监督预训练权重，冻结backbone学习水下特征

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

# DFUI数据集路径 (COCO 2017格式)
data_root = '/media/HDD0/XCX/exp_2_data/dfui/'
ann_root = '/media/HDD0/XCX/exp_2_data/dfui/annotations/'

# ImageNet torchvision监督预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50')))

# DFUI 5个类别
model['roi_head']['bbox_head'] = [
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=5,  # DFUI 5类
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=5,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=5,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.03, 0.03, 0.07, 0.07]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
]

# 修改epoch为48
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)

# 修改lr scheduler为48 epoch
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

# 数据加载配置 - DFUI 5类
classes_dfui = ('echinus', 'holothurian', 'scallop', 'starfish', 'waterweeds')

train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_train2017.json',
        metainfo=dict(classes=classes_dfui)))
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_val2017.json',
        metainfo=dict(classes=classes_dfui)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val2017.json')
test_evaluator = val_evaluator