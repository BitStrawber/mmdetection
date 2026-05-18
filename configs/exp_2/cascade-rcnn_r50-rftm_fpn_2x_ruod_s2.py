# J10 RFTM S2: backbone+RFTM 权重 → RUOD 微调 (24ep, 10类)
# 加载S1训练的backbone+RFTM权重，检测头重新随机初始化

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'
ann_root = data_root + 'annotations/'

# 替换backbone为ResNetWithRFTM（S2使用RFTM模块但从头训练检测头）
# 权重通过load_from从S1的backbone_only.pth加载
model = dict(
    backbone=dict(
        type='ResNetWithRFTM',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        rftm_channels=256))

classes_ruod = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=6, num_workers=2, persistent_workers=True,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes_ruod)))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes_ruod)))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001))
