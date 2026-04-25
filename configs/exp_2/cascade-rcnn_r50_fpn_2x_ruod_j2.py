# J2: ImageNet Supervised + ResNet-50 + Cascade R-CNN -> RUOD (2 GPU, 总BS=12)
# 使用torchvision内置的ResNet-50监督预训练权重

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

# 数据集路径配置
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'

# 2 GPU配置 (总BS=12)
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