# Cross: B训练 + A验证
_base_ = '../../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/cross_split/'

train_dataloader = dict(
    batch_size=6, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'train_B.json'))
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'train_A.json'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=ann_root + 'train_A.json')
test_evaluator = val_evaluator
