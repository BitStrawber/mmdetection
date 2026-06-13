#!/usr/bin/env python3
"""Convert SSL pretraining checkpoints to MMDetection backbone checkpoints.

This utility is intentionally conservative. It keeps likely backbone/encoder
parameters and drops common pretraining-only modules such as decoder, head,
projection and teacher momentum branches.
"""

import argparse
from collections import OrderedDict

import torch


DROP_KEYWORDS = (
    'decoder',
    'decode_head',
    'head',
    'neck',
    'projection',
    'projector',
    'predictor',
    'teacher',
    'momentum',
    'target_generator',
    'mask_token',
)


SOURCE_CONTAINERS = (
    'state_dict',
    'model',
    'student',
    'teacher',
)


STRIP_PREFIXES = (
    'module.',
    'student.module.',
    'student.backbone.',
    'student.encoder.',
    'teacher.module.',
    'teacher.backbone.',
    'teacher.encoder.',
    'model.backbone.',
    'model.encoder.',
    'backbone.',
    'encoder.',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--source',
        default='auto',
        choices=['auto', 'state_dict', 'model', 'student', 'teacher'],
        help='Checkpoint field to read.')
    parser.add_argument(
        '--prepend',
        default='backbone.',
        help='Prefix added to output keys. Use empty string to disable.')
    parser.add_argument(
        '--allow-decoder',
        action='store_true',
        help='Do not drop decoder/head/projector keys.')
    return parser.parse_args()


def get_state_dict(ckpt, source):
    if source != 'auto':
        return ckpt[source] if isinstance(ckpt, dict) and source in ckpt else ckpt
    if not isinstance(ckpt, dict):
        return ckpt
    for key in SOURCE_CONTAINERS:
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def strip_prefix(key):
    changed = True
    while changed:
        changed = False
        for prefix in STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def should_drop(key, allow_decoder):
    if allow_decoder:
        return False
    lower = key.lower()
    return any(part in lower for part in DROP_KEYWORDS)


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    source_state = get_state_dict(ckpt, args.source)

    converted = OrderedDict()
    dropped = 0
    for key, value in source_state.items():
        if should_drop(key, args.allow_decoder):
            dropped += 1
            continue
        new_key = strip_prefix(key)
        if not new_key or should_drop(new_key, args.allow_decoder):
            dropped += 1
            continue
        if args.prepend and not new_key.startswith(args.prepend):
            new_key = args.prepend + new_key
        converted[new_key] = value

    out = {
        'state_dict': converted,
        'meta': {
            'source_checkpoint': args.checkpoint,
            'source': args.source,
            'num_converted_keys': len(converted),
            'num_dropped_keys': dropped,
        }
    }
    torch.save(out, args.out)

    print('source:', args.checkpoint)
    print('output:', args.out)
    print('converted keys:', len(converted))
    print('dropped keys:', dropped)
    print('first keys:')
    for key in list(converted.keys())[:20]:
        print(' ', key)


if __name__ == '__main__':
    main()
