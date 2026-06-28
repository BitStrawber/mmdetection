# J10 DINO-official source comparison S1: DFUI only.
#
# The detector head uses the unified 11-class source space for consistency
# across DFUI / DFUI+RUOD / DFUI+RUOD+UIIS source-comparison runs. Only the
# extracted backbone is passed to RUOD S2.

_base_ = './cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py'

data_root = '/media/HDD0/XCX/exp_2/DFUI_ALL/'
ann_root = data_root + 'annotations/'
pretrained = '../pretrained_weights/dino_rn50_official_100e_backbone.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

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

load_from = None
resume = False
