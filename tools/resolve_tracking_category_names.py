#!/usr/bin/env python3
"""Resolve candidate category names for tracking-style underwater datasets.

This tool does not overwrite original annotations. It keeps the original COCO
category names and writes a mapping table with candidate category names inferred
from available dataset metadata:

* MUOT_3M: image captions stored in the generated COCO JSON
* UW-COT220: per-sequence language.txt files
* UOT100: original sequence/category names

Optionally, it also writes derived COCO JSON files whose category ids are
reassigned to the resolved names. Unresolved samples keep their original COCO
category name.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_MUOT_ANNS = [
    'annotations/instances_train_bbox20pct.json',
    'annotations/instances_test_bbox20pct.json',
]

TEXT_STOP_WORDS = (
    'swims', 'swim', 'swimming', 'moves', 'move', 'moving', 'shuttles',
    'shuttle', 'rests', 'resting', 'crawls', 'crawling', 'glides',
    'gliding', 'floats', 'floating', 'is', 'are', 'was', 'were',
    'performing', 'nestled', 'surrounded', 'among', 'near', 'on', 'over',
    'above', 'below', 'through', 'with', 'in', 'inside', 'outside', 'at',
    'along', 'around', 'beside', 'between'
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--muot-root', default='/media/HDD1/XCX/exp_2/MUOT_3M')
    parser.add_argument(
        '--muot-ann',
        nargs='+',
        default=DEFAULT_MUOT_ANNS,
        help='MUOT_3M COCO files. Relative paths are resolved under --muot-root.')
    parser.add_argument(
        '--uwcot-root',
        default='/media/HDD1/XCX/exp_2/UW-COT220/UW-COT220/UW-COT220')
    parser.add_argument(
        '--uwcot-ann',
        default='/media/HDD1/XCX/exp_2/UW-COT220/annotations/instances_all_bbox20pct.json')
    parser.add_argument('--uot-root', default='/media/HDD1/XCX/exp_2/UOT100')
    parser.add_argument(
        '--uot-ann',
        default='/media/HDD1/XCX/exp_2/UOT100/annotations/instances_all_bbox20pct.json')
    parser.add_argument(
        '--out-dir',
        default='logs/tracking_category_resolution',
        help='Output directory for CSV/JSON mappings and derived COCO files.')
    parser.add_argument(
        '--no-coco',
        action='store_true',
        help='Only write mapping tables, not derived COCO JSON files.')
    return parser.parse_args()


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_text(path):
    path = Path(path)
    if not path.is_file():
        return ''
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read().strip()


def clean_phrase(text):
    text = str(text).strip()
    text = text.replace('鈥檚', "'s").replace('茅', 'e')
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' \t.,;:')


def extract_from_sentence(text):
    text = clean_phrase(text)
    if not text:
        return '', 'low'

    pattern = (
        r'^(?:a|an|the)\s+'
        r'(?:(?:group|pair|school|pod)\s+of\s+)?'
        r'(.+?)'
        r'(?:\s+(?:' + '|'.join(TEXT_STOP_WORDS) + r')\b|[,.;:]|$)'
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1)
    else:
        candidate = re.split(r'[,.;:]', text, maxsplit=1)[0]

    candidate = re.sub(r'\b(?:its|their|his|her)\b.*$', '', candidate, flags=re.IGNORECASE)
    candidate = clean_phrase(candidate)
    candidate = re.sub(r'\s+', ' ', candidate).strip()

    words = candidate.split()
    if len(words) > 6:
        candidate = ' '.join(words[:6])

    if not candidate:
        return '', 'low'
    return candidate, 'medium-high'


def resolve_uot_category(sequence, original_category):
    category = str(original_category or '').strip()
    if category:
        return category, 'original_coco_category', 'high'
    sequence = str(sequence or '').strip()
    if sequence:
        return sequence, 'sequence_name_fallback', 'medium'
    return '', 'missing_category', 'low'


def coco_categories_by_id(coco):
    return {
        cat.get('id'): str(cat.get('name', cat.get('id')))
        for cat in coco.get('categories', [])
    }


def anns_by_image(coco):
    grouped = defaultdict(list)
    for ann in coco.get('annotations', []):
        grouped[ann.get('image_id')].append(ann)
    return grouped


def original_category_for_image(image, image_anns, categories_by_id):
    anns = image_anns.get(image.get('id'), [])
    if not anns:
        return ''
    return categories_by_id.get(anns[0].get('category_id'), str(anns[0].get('category_id')))


def add_row(rows, dataset, split, sequence, original_category, resolved_category,
            source, confidence, text, images, annotations):
    rows.append({
        'dataset': dataset,
        'split': split,
        'sequence': sequence,
        'original_category': original_category,
        'resolved_category': resolved_category,
        'source': source,
        'confidence': confidence,
        'text': text,
        'images': images,
        'annotations': annotations,
        'output_category': resolved_category or original_category,
    })


def aggregate_coco_sequences(coco, dataset, split, resolver):
    categories_by_id = coco_categories_by_id(coco)
    image_anns = anns_by_image(coco)
    seq_items = defaultdict(lambda: {
        'images': 0,
        'annotations': 0,
        'text': '',
        'original_category': '',
    })

    for image in coco.get('images', []):
        sequence = str(image.get('sequence') or Path(str(image.get('file_name', ''))).parent.name)
        item = seq_items[sequence]
        item['images'] += 1
        anns = image_anns.get(image.get('id'), [])
        item['annotations'] += len(anns)
        if not item['original_category']:
            item['original_category'] = original_category_for_image(image, image_anns, categories_by_id)
        if not item['text']:
            item['text'] = str(image.get('caption') or image.get('language') or '')

    rows = []
    for sequence, item in sorted(seq_items.items()):
        resolved, source, confidence, text = resolver(sequence, item)
        add_row(
            rows=rows,
            dataset=dataset,
            split=split,
            sequence=sequence,
            original_category=item['original_category'],
            resolved_category=resolved,
            source=source,
            confidence=confidence,
            text=text,
            images=item['images'],
            annotations=item['annotations'],
        )
    return rows


def muot_resolver(sequence, item):
    text = item.get('text', '')
    resolved, confidence = extract_from_sentence(text)
    return resolved, 'caption_txt' if text else 'missing_caption', confidence, text


def uwcot_resolver(root):
    def resolver(sequence, item):
        text = read_text(Path(root) / sequence / 'language.txt')
        resolved, confidence = extract_from_sentence(text)
        return resolved, 'language_txt' if text else 'missing_language', confidence, text
    return resolver


def write_mapping(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'tracking_category_resolution.csv'
    json_path = out_dir / 'tracking_category_resolution.json'
    fields = [
        'dataset', 'split', 'sequence', 'original_category',
        'resolved_category', 'output_category', 'source', 'confidence',
        'images', 'annotations', 'text',
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    write_json(json_path, rows)
    return csv_path, json_path


def update_coco_categories(coco, rows):
    row_by_sequence = {row['sequence']: row for row in rows}
    image_to_category = {}
    category_names = []
    category_id_by_name = {}

    def get_category_id(name):
        if name not in category_id_by_name:
            category_id_by_name[name] = len(category_id_by_name) + 1
            category_names.append(name)
        return category_id_by_name[name]

    for image in coco.get('images', []):
        sequence = str(image.get('sequence') or Path(str(image.get('file_name', ''))).parent.name)
        row = row_by_sequence.get(sequence)
        category_name = row['output_category'] if row else 'object'
        image_to_category[image.get('id')] = get_category_id(category_name)
        image['resolved_category'] = category_name
        if row:
            image['category_resolution_source'] = row['source']
            image['category_resolution_confidence'] = row['confidence']

    for ann in coco.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id in image_to_category:
            ann['category_id'] = image_to_category[image_id]

    coco['categories'] = [
        {'id': category_id_by_name[name], 'name': name}
        for name in category_names
    ]
    coco.setdefault('info', {})
    coco['info']['category_resolution'] = {
        'note': 'Derived category names; original categories are preserved in mapping CSV/JSON.',
    }
    return coco


def process_muot(args, out_dir, all_rows, write_coco=True):
    for ann_arg in args.muot_ann:
        ann_path = resolve_path(args.muot_root, ann_arg)
        if not ann_path.is_file():
            print('[skip] MUOT_3M missing:', ann_path)
            continue
        coco = read_json(ann_path)
        split = 'train' if 'train' in ann_path.name else 'test'
        rows = aggregate_coco_sequences(coco, 'MUOT_3M', split, muot_resolver)
        all_rows.extend(rows)
        if write_coco:
            updated = update_coco_categories(coco, rows)
            out_path = Path(out_dir) / ('muot3m_{}_bbox20pct_resolved.json'.format(split))
            write_json(out_path, updated)
            print('MUOT_3M {} resolved coco: {}'.format(split, out_path))


def process_uwcot(args, out_dir, all_rows, write_coco=True):
    ann_path = Path(args.uwcot_ann)
    if not ann_path.is_file():
        print('[skip] UW-COT220 missing:', ann_path)
        return
    coco = read_json(ann_path)
    rows = aggregate_coco_sequences(
        coco, 'UW-COT220', 'all', uwcot_resolver(args.uwcot_root))
    all_rows.extend(rows)
    if write_coco:
        updated = update_coco_categories(coco, rows)
        out_path = Path(out_dir) / 'uwcot220_all_bbox20pct_resolved.json'
        write_json(out_path, updated)
        print('UW-COT220 resolved coco:', out_path)


def process_uot(args, out_dir, all_rows, write_coco=True):
    ann_path = Path(args.uot_ann)
    if not ann_path.is_file():
        print('[skip] UOT100 missing:', ann_path)
        return
    coco = read_json(ann_path)

    def resolver(sequence, item):
        resolved, source, confidence = resolve_uot_category(
            sequence=sequence,
            original_category=item.get('original_category', ''),
        )
        text = read_text(Path(args.uot_root) / sequence / 'description.txt')
        return resolved, source, confidence, text

    rows = aggregate_coco_sequences(coco, 'UOT100', 'all', resolver)
    all_rows.extend(rows)
    if write_coco:
        updated = update_coco_categories(coco, rows)
        out_path = Path(out_dir) / 'uot100_all_bbox20pct_resolved.json'
        write_json(out_path, updated)
        print('UOT100 resolved coco:', out_path)


def print_summary(rows):
    by_dataset = defaultdict(lambda: {'rows': 0, 'resolved': 0, 'images': 0})
    by_source = defaultdict(int)
    for row in rows:
        item = by_dataset[row['dataset']]
        item['rows'] += 1
        item['images'] += int(row.get('images') or 0)
        if row.get('resolved_category'):
            item['resolved'] += 1
        by_source[row.get('source', '')] += 1

    print('\nSummary')
    print('=' * 80)
    for dataset, item in sorted(by_dataset.items()):
        print(
            '{}: sequences={}, resolved={}, images={}'.format(
                dataset, item['rows'], item['resolved'], item['images']))
    print('\nSources')
    for source, count in sorted(by_source.items()):
        print('  {}: {}'.format(source, count))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    all_rows = []
    write_coco = not args.no_coco

    process_muot(args, out_dir, all_rows, write_coco=write_coco)
    process_uwcot(args, out_dir, all_rows, write_coco=write_coco)
    process_uot(args, out_dir, all_rows, write_coco=write_coco)

    csv_path, json_path = write_mapping(all_rows, out_dir)
    print('mapping csv:', csv_path)
    print('mapping json:', json_path)
    print_summary(all_rows)


if __name__ == '__main__':
    main()
