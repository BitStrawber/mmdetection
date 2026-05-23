# J10 HDP/RFTM S2: paper-style RFTM prior -> RUOD detection finetune.
#
# Use with:
#   bash tools/dist_train.sh \
#     configs/exp_2/cascade-rcnn_r50-rftm-hdp_fpn_2x_ruod_j10_s2.py 2 \
#     --work-dir work_dirs/j10_hdp_s2 \
#     --cfg-options model.backbone.rftm_init=work_dirs/j10_hdp_s1/rftm_prior.pth
#
# The S1 checkpoint only contains backbone.rftm.* keys. The ResNet base still
# starts from ImageNet weights via init_cfg below.

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'
ann_root = data_root + 'annotations/'

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
        rftm_channels=256,
        rftm_init=None,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')))

classes_ruod = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes_ruod)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes_ruod),
        test_mode=True))

test_dataloader = val_dataloader
val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001))

load_from = None
resume = False
