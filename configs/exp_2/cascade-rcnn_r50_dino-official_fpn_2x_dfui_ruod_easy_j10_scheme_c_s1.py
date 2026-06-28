# J10 DINO-official source comparison S1: DFUI + RUOD easy.

_base_ = './cascade-rcnn_r50_dino-official_fpn_2x_dfui_all_j10_scheme_c_s1.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_RUOD_EASY/'
ann_root = data_root + 'annotations/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=ann_root + 'instances_val.json',
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator
