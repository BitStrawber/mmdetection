_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# Enhanced COCO dataset (UWNR output)
data_root = '/media/HDD0/XCX/COCO/coco_convert_uwnr/'
train_ann = data_root + 'annotations/instances_train50000.json'
test_ann = '/media/HDD0/XCX/COCO/annotations/instances_val2017.json'
num_classes = 80

# Override bbox head num_classes from base (base has 80)
model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='torchvision://resnet50'
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ])
)

# 1333x800, BS=16(8×2), 24epoch
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=train_ann,
        img_prefix=data_root + 'images/'),
    val=dict(
        type='CocoDataset',
        ann_file=test_ann,
        img_prefix='/media/HDD0/XCX/COCO/val2017/'),
    test=dict(
        type='CocoDataset',
        ann_file=test_ann,
        img_prefix='/media/HDD0/XCX/COCO/val2017/'))

auto_scale_lr = dict(enable=False, base_batch_size=16)
evaluation = dict(interval=1, save_best='auto', classwise=True)