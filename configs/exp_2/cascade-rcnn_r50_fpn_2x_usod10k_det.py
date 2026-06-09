# USOD10K single-class detection config.
#
# USOD10K saliency masks are converted to COCO bboxes with one category:
# object.

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD1/XCX/exp_2/USOD10K_DET/'
ann_root = data_root + 'annotations/'

classes = ('object',)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
        ]))

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'instances_trainval.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
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

optim_wrapper = dict(
    optimizer=dict(lr=0.015))

load_from = None
resume = False
