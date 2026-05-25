#!/usr/bin/env python3
"""UVOT400 跟踪格式 → COCO + 20%筛选"""
import json, os, glob
from PIL import Image

BASE = '/media/HDD1/XCX/exp_2/UVOT400'
THRESHOLD = 0.2


def convert_split(data_dir, split_name):
    out_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'instances_{split_name}.json')

    if os.path.exists(out):
        print(f"{split_name} 已存在: {out}")
        return out

    images, anns = [], []
    img_id, ann_id = 0, 0
    cats = [{'id': 1, 'name': 'object'}]

    for root, _, files in os.walk(data_dir):
        if 'groundtruth_rect.txt' not in files:
            continue
        gt_file = os.path.join(root, 'groundtruth_rect.txt')

        # 图片可能在 imgs/ 或当前目录
        img_dir = os.path.join(root, 'imgs')
        if not os.path.isdir(img_dir):
            img_dir = root

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) +
                           glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]')) +
                           glob.glob(os.path.join(img_dir, '*.[jJ][pP][eE][gG]')))
        if not img_files:
            # 也检查子目录
            for sub in sorted(glob.glob(os.path.join(root, '*/'))):
                img_files = sorted(glob.glob(os.path.join(sub, '*.[jJ][pP][gG]')) +
                                   glob.glob(os.path.join(sub, '*.[pP][nN][gG]')))
                if img_files:
                    img_dir = sub
                    break
        if not img_files:
            print(f"  跳过 {os.path.basename(root)}: 无图片")
            continue

        with open(gt_file) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if i >= len(img_files):
                break
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            if w <= 0 or h <= 0:
                continue
            try:
                pil = Image.open(img_files[i])
                W, H = pil.size
            except:
                W, H = 1920, 1080

            images.append({
                'id': img_id,
                'file_name': os.path.relpath(img_files[i], data_dir),
                'width': W, 'height': H
            })
            anns.append({
                'id': ann_id, 'image_id': img_id,
                'category_id': 1, 'bbox': [x, y, w, h],
                'area': w * h, 'iscrowd': 0
            })
            img_id += 1
            ann_id += 1

    if not images:
        print(f"{split_name}: 无数据")
        return None

    coco = {
        'info': {'description': f'UVOT400_{split_name}'},
        'licenses': [], 'categories': cats,
        'images': images, 'annotations': anns
    }
    with open(out, 'w') as f:
        json.dump(coco, f)
    print(f"{split_name}: {len(images)} imgs, {len(anns)} anns → {out}")
    return out


def filter_coco(coco, threshold=THRESHOLD):
    img_map = {i['id']: i for i in coco['images']}
    img_max = {}
    for a in coco['annotations']:
        iid = a['image_id']
        _, _, w, h = a['bbox']
        area = w * h
        if iid not in img_max or area > img_max[iid]:
            img_max[iid] = area
    keep = set()
    for iid, ma in img_max.items():
        im = img_map[iid]
        ia = im.get('width', 0) * im.get('height', 0)
        if ia > 0 and ma / ia >= threshold:
            keep.add(iid)
    return {
        'info': coco.get('info', {}), 'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [i for i in coco['images'] if i['id'] in keep],
        'annotations': [a for a in coco['annotations'] if a['image_id'] in keep]
    }


if __name__ == '__main__':
    for split in ['train', 'test']:
        if split == 'train':
            data_dir = os.path.join(BASE, 'train')
        else:
            data_dir = os.path.join(BASE, 'test')

        out = convert_split(data_dir, split)
        if out:
            with open(out) as f:
                coco = json.load(f)
            filtered = filter_coco(coco)
            total, kept = len(coco['images']), len(filtered['images'])
            filtered_out = out.replace('.json', f'_bbox{int(THRESHOLD*100)}pct.json')
            with open(filtered_out, 'w') as f:
                json.dump(filtered, f)
            print(f"  筛选后: {total} → {kept} ({kept/total*100:.1f}%)")
