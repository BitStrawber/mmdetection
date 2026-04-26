# J4: ImageNet DINO + ResNet-50 + Mask R-CNN -> UWIS10K (10类)
# 使用DINO预训练的ResNet-50权重

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_2x_uwis10k.py'

# backbone修改为DINO预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained_weights/dino_resnet50_pretrain.pth')))

# 2 GPU配置 (总BS=12)
train_dataloader = dict(batch_size=6, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader