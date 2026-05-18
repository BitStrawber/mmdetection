# J10 RFTM S1: ResNetWithRFTM backbone + DFUI_NEW 预训练 (48ep, 11类)
# 参考论文: Learning Heavily-Degraded Prior for Underwater Object Detection
# 在ResNet layer1之后插入RFTM轻量特征增强模块，学习水下退化区域特征迁移

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_NEW/'
ann_root = data_root + 'annotations/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish',
           'waterweeds')

# 替换backbone为ResNetWithRFTM
model = dict(
    backbone=dict(
        type='ResNetWithRFTM',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,  # 冻结全部backbone，只训练RFTM+检测头
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        rftm_channels=256,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=11),
            dict(type='Shared2FCBBoxHead', num_classes=11),
            dict(type='Shared2FCBBoxHead', num_classes=11),
        ]))

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_root + 'instances_val.json',
    metric='bbox')
test_evaluator = val_evaluator

# 只训练RFTM+检测头，参数少，24epoch足够
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)
]
