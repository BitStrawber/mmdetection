# Faster R-CNN: 筛选RUOD (easy_merged, trainval=训练+验证)
_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py'

data_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

model = dict(
    roi_head=dict(
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=10)))

train_dataloader = dict(
    batch_size=6, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=data_root + 'annotations/easy_merged.json',
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=data_root + 'annotations/easy_merged.json',
        metainfo=dict(classes=classes),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/easy_merged.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=96, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=96, by_epoch=True,
         milestones=[64, 80], gamma=0.1)
]

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', patience=15, min_delta=0.001)
]
