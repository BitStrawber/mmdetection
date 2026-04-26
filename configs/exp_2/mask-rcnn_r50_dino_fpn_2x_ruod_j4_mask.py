# J4: ImageNet DINO + ResNet-50 + Mask R-CNN -> RUOD
# 使用DINO预训练的ResNet-50权重

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

# 数据集路径配置 (绝对路径)
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/annotations/'

# 修改backbone使用DINO预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained_weights/dino_resnet50_pretrain.pth')),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10)))

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