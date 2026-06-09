# J10 MAE-route S1:
# ImageNet MAE pretrained ViT-Base -> expanded underwater DFUI source.

_base_ = './cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_RUOD_UIIS_USOD_EASY/'
ann_root = data_root + 'annotations/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish',
           'waterweeds', 'object')

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256, num_classes=12),
            dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256, num_classes=12),
            dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256, num_classes=12),
        ]))

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes),
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val.json', metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5,
                    save_best='coco/bbox_mAP', rule='greater'))

load_from = None
resume = False
