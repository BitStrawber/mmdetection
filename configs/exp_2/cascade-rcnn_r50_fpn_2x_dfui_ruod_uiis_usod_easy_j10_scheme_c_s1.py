# J10 scheme C S1 with USOD10K easy added as a 12th objectness class.

_base_ = './cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY/'
ann_root = data_root + 'annotations/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish',
           'waterweeds', 'object')

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=12),
            dict(type='Shared2FCBBoxHead', num_classes=12),
            dict(type='Shared2FCBBoxHead', num_classes=12),
        ]))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'instances_train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'instances_val.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_root + 'instances_val.json',
    metric='bbox')
test_evaluator = val_evaluator
