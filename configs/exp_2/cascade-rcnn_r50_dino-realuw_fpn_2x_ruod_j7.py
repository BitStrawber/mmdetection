# J7: Real underwater DINO pretraining + ResNet-50 + Cascade R-CNN -> RUOD.

_base_ = './cascade-rcnn_r50_fpn_2x_ruod_j2.py'

pretrained = '../pretrained_weights/j7_realuw_dino_resnet50_backbone.pth'

data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = data_root + 'annotations/'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json'))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator

load_from = None
resume = False
