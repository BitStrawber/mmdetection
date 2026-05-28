# UIIS10K detection-only baseline converted from instance segmentation COCO.

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco/'
ann_root = data_root + 'annotations/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'instances_train_det.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'instances_val_det.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_root + 'instances_val_det.json',
    metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.015))

load_from = None
resume = False
