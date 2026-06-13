_base_ = [
    'mmpretrain::configs/mae/'
    'mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'
]

# J11: RealUW + ViT-Base + MAE.
# This is the closest task to the official MMPreTrain MAE recipe.

data_root = '/media/HDD1/XCX/exp_2/REALUW_SSL/imagefolder'

train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        split='train'))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5))

load_from = None
resume = False
