# J10 S2: DFUI_NEW 预训练权重 → RUOD训练
_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2/RUOD/coco/'
ann_root = data_root + 'annotations/'

load_from = 'work_dirs/j10_v2_s1/best_coco_bbox_mAP_epoch_20.pth'

classes_ruod = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=6, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes_ruod)))
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes_ruod)))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator
