# J10 Stage 2: DFUI微调权重 → RUOD训练 (10类)
# 从Stage 1的best checkpoint加载backbone权重，检测头重新初始化

_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py'

# RUOD数据集路径
data_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/'
ann_root = '/media/HDD0/XCX/exp_2_data/exp_2/RUOD/coco/annotations/'

# 从Stage1的best checkpoint加载（训练时会动态替换）
# 注意：backbone权重会加载，bbox_head会重新初始化（类别数不同：5→10）
load_from = 'work_dirs/j10_s1/best_coco_bbox_mAP_epoch_XX.pth'

# 数据加载配置 - RUOD 10类
classes_ruod = ('holothurian', 'echinus', 'scallop', 'starfish', 'fish', 
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish')

train_dataloader = dict(
    batch_size=6, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_root + 'instances_train.json',
        metainfo=dict(classes=classes_ruod)))
val_dataloader = dict(
    batch_size=1, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='val/'),
        ann_file=ann_root + 'instances_val.json',
        metainfo=dict(classes=classes_ruod)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_root + 'instances_val.json')
test_evaluator = val_evaluator