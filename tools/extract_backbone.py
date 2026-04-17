"""Extract backbone+neck state_dict from a full Cascade RCNN checkpoint.

Usage:
    python tools/extract_backbone.py \
        --checkpoint work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/latest.pth \
        --output work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/backbone_only.pth
"""
import argparse
import torch

SAFE_PREFIXES = [
    'backbone.',
    'neck.',
    'rpn_head.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f'Loading {args.checkpoint} ...')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    state_dict = ckpt.get('state_dict', ckpt)
    print(f'  Total keys: {len(state_dict)}')

    extracted = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in SAFE_PREFIXES):
            extracted[k] = v

    print(f'  Extracted keys: {len(extracted)}')

    out_ckpt = {
        'state_dict': extracted,
        'meta': ckpt.get('meta', {}),
    }

    torch.save(out_ckpt, args.output)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()