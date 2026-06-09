#!/usr/bin/env python3
"""Build USOD10K easy detection subset with A/B cross filtering.

This is a thin wrapper around ``tools/uiis10k_cross_easy.py`` because USOD10K
has already been converted to COCO detection format with a single ``object``
category.
"""

import sys

from uiis10k_cross_easy import main, parse_args


def patched_parse_args():
    args = parse_args()
    if args.data_root == '/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco/':
        args.data_root = '/media/HDD1/XCX/exp_2/USOD10K_DET/'
    if args.ann == '/media/HDD0/XCX/exp_2_data/exp_2/UIIS10K/coco/annotations/instances_train_det.json':
        args.ann = '/media/HDD1/XCX/exp_2/USOD10K_DET/annotations/instances_trainval.json'
    if args.img_prefix == 'train/':
        args.img_prefix = 'images/'
    if args.cross_dir is None:
        args.cross_dir = '/media/HDD1/XCX/exp_2/USOD10K_DET/annotations/cross_split_det'
    if args.config == 'configs/exp_2/cascade-rcnn_r50_fpn_2x_uiis10k_det.py':
        args.config = 'configs/exp_2/cascade-rcnn_r50_fpn_2x_usod10k_det.py'
    if args.work_dir == 'work_dirs/uiis10k_cross_easy':
        args.work_dir = 'work_dirs/usod10k_cross_easy'
    if args.log_prefix == 'uiis10k_cross':
        args.log_prefix = 'usod10k_cross'
    return args


if __name__ == '__main__':
    import uiis10k_cross_easy

    uiis10k_cross_easy.parse_args = patched_parse_args
    sys.exit(main())
