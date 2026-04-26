# J2: ImageNet Supervised + ResNet-50 + Mask R-CNN -> RUOD (10类)
# 使用torchvision内置的ResNet-50监督预训练权重

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_2x_ruod.py'

# 2 GPU配置 (总BS=12)
train_dataloader = dict(
    batch_size=6, 
    num_workers=2)
val_dataloader = dict(
    batch_size=1, 
    num_workers=2)
test_dataloader = val_dataloader