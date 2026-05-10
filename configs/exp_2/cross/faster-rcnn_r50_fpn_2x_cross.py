# Faster R-CNN: 筛选后的RUOD训练 (48 epoch)
_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py'

# RUOD 数据集路径
ruod_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'
ann_root = ruod_root + 'annotations/'

# 10类
classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)))

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(
        data_root=ruod_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'easy_merged.json',
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=ruod_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'easy_merged.json',
        metainfo=dict(classes=classes),
        test_mode=True))

val_evaluator = dict(
    ann_file=ann_root + 'easy_merged.json',
    metric='bbox')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
