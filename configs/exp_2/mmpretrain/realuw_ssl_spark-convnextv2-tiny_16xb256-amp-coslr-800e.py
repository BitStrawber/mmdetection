_base_ = [
    'mmpretrain::spark/'
    'spark_sparse-convnextv2-tiny_16xb256-amp-coslr-800e_in1k.py'
]

# J13: RealUW + ConvNeXtV2-Tiny + SparK/FCMAE-like masked modeling.
# This upstream config exists in the cloned MMPreTrain repo.

data_root = '/media/HDD1/XCX/exp_2/REALUW_SSL/imagefolder'

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        split='train'))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5))

load_from = None
resume = False
