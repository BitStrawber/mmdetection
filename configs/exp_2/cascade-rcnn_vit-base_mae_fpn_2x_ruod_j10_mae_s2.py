# J10 MAE-route S2:
# RUOD fine-tuning with ViT-Base Cascade R-CNN. Load S1 backbone through
# cfg-options: load_from=/path/to/backbone_only.pth

_base_ = './cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py'

data_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'
ann_root = data_root + 'annotations/'

classes_ruod = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes_ruod),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes_ruod),
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val.json', metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5,
                    save_best='coco/bbox_mAP', rule='greater'))

load_from = None
resume = False
