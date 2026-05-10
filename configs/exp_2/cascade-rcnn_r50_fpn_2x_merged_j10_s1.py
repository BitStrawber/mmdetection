# J10 S1: 筛选RUOD + DFUI 合并预训练 (96 epoch + 早停)
_base_ = '../../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_NEW/'
ann_root = data_root + 'annotations/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

model = dict(
    roi_head=dict(
        bbox_head=[dict(num_classes=10), dict(num_classes=10), dict(num_classes=10)]))

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_train2017.json',
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_val2017.json',
        metainfo=dict(classes=classes),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_root + 'instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=96, val_interval=1)

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=96, by_epoch=True, milestones=[64, 88], gamma=0.1)
]

# 早停: val_coco/bbox_mAP 连续10 epoch不涨就停止
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', patience=10, min_delta=0.001)
]
