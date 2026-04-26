# J2: ImageNet Supervised + ResNet-50 + Mask R-CNN -> RUOD (10类)
# 使用torchvision内置的ResNet-50监督预训练权重

_base_ = '../_base_/models/mask-rcnn_r50_fpn.py'

# 数据集路径配置 (绝对路径)
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/annotations/'

# 修改num_classes
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10)))

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=24,
    by_epoch=True,
    milestones=[16, 22],
    gamma=0.1)

# 2 GPU配置 (总BS=12)
train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json'))
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json'))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_root + 'instances_val.json',
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator