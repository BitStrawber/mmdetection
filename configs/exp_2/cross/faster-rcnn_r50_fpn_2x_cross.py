# Faster R-CNN: 筛选后的RUOD + DFUI 训练
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py'

# RUOD 筛选后数据集
ruod_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
cross_dir = ruod_root + 'cross_split/'

# DFUI 数据集
dfui_root = '/media/HDD0/XCX/exp_2_data/dfui/'

# 10类
classes = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
           'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)))

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                data_root=ruod_root,
                data_prefix=dict(img='train/'),
                ann_file=cross_dir + 'A_easy.json',
                metainfo=dict(classes=classes),
                filter_cfg=dict(filter_empty_gt=True, min_size=32)),
            dict(
                type='CocoDataset',
                data_root=ruod_root,
                data_prefix=dict(img='train/'),
                ann_file=cross_dir + 'B_easy.json',
                metainfo=dict(classes=classes),
                filter_cfg=dict(filter_empty_gt=True, min_size=32)),
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=ruod_root,
        data_prefix=dict(img='val/'),
        ann_file=ruod_root + 'annotations/instances_val.json',
        metainfo=dict(classes=classes)))

val_evaluator = dict(ann_file=ruod_root + 'annotations/instances_val.json')
