#!/usr/bin/env python3
"""Extract only backbone weights from an MMDetection checkpoint."""

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f'Loading {args.checkpoint} ...')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    extracted = {
        key: value
        for key, value in state_dict.items()
        if key.startswith('backbone.')
    }
    if not extracted:
        raise RuntimeError(f'No backbone.* keys found in {args.checkpoint}')

    out_ckpt = {
        'state_dict': extracted,
        'meta': {
            **ckpt.get('meta', {}),
            'extracted_from': args.checkpoint,
            'extracted_prefixes': ['backbone.'],
        },
    }
    torch.save(out_ckpt, args.output)

    print(f'  Total keys: {len(state_dict)}')
    print(f'  Backbone keys: {len(extracted)}')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
