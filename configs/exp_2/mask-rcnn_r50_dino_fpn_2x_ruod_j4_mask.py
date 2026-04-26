# J4: ImageNet DINO + ResNet-50 + Mask R-CNN -> RUOD (10类)
# 使用DINO预训练的ResNet-50权重

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# 数据集路径配置
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/annotations/'

# 修改: backbone + num_classes
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained_weights/dino_resnet50_pretrain.pth')),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10)))

train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json'))

val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator