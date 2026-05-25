#!/usr/bin/env python3
"""CoralSCOP: 每图独立JSON → COCO格式合并 + 20%筛选"""
import json, os, glob, argparse

def merge_coralscop(data_dir, split='train'):
    jsons = sorted(glob.glob(os.path.join(data_dir, split, 'jsons', '*.json')))
    if not jsons:
        jsons = sorted(glob.glob(os.path.join(data_dir, split, '*.json')))

    images, annotations = [], []
    img_id, ann_id = 0, 0
    cat_set, categories = set(), []

    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        img_info = data.get('image', data.get('images', [{}])[0] if isinstance(data.get('images'), list) else {})
        if 'file_name' not in img_info:
            continue

        # 找图片实际路径
        img_path = None
        for ext in ['', '.jpg', '.png', '.jpeg']:
            fn = img_info['file_name']
            if not fn.endswith(ext):
                fn += ext
            p = os.path.join(os.path.dirname(jf), fn)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path and os.path.exists(os.path.join(os.path.dirname(jf), img_info['file_name'])):
            img_path = os.path.join(os.path.dirname(jf), img_info['file_name'])
        if not img_path:
            continue

        images.append({
            'id': img_id,
            'file_name': os.path.relpath(img_path, data_dir),
            'width': img_info.get('width', 0),
            'height': img_info.get('height', 0)
        })

        anns = data.get('annotations', [])
        for ann in anns:
            bbox = ann.get('bbox', [0, 0, 0, 0])
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            ann_id += 1
        img_id += 1

    categories = [{'id': 1, 'name': 'coral'}]

    coco = {
        'info': {'description': f'CoralSCOP {split}'},
        'licenses': [], 'categories': categories,
        'images': images, 'annotations': annotations
    }

    ann_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    out = os.path.join(ann_dir, f'instances_{split}.json')
    with open(out, 'w') as f:
        json.dump(coco, f)

    print(f"CoralSCOP {split}: {len(images)} imgs, {len(annotations)} anns → {out}")
    return out


def filter_coco(coco, threshold=0.2):
    """筛选最大bbox >= threshold * 图片面积 的图片"""
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
    }, len(coco['images']), len(keep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/media/HDD1/XCX/exp_2/CoralSCOP')
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()

    for split in ['train', 'test']:
        out_path = merge_coralscop(args.data_dir, split)
        if out_path:
            with open(out_path) as f:
                coco = json.load(f)
            filtered, total, kept = filter_coco(coco, args.threshold)
            out_filtered = out_path.replace('.json', f'_bbox{int(args.threshold*100)}pct.json')
            with open(out_filtered, 'w') as f:
                json.dump(filtered, f)
            print(f"  {split}: {total} → {kept} ({kept/total*100:.1f}%)")
