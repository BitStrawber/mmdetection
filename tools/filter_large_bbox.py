#!/usr/bin/env python3
"""
筛选最大bbox占图面积>=阈值的图片
用法:
  python tools/filter_large_bbox.py --json path/to/annotations.json --out path/to/filtered.json --threshold 0.2
"""
import json, argparse

def filter_bbox_coverage(coco_path, output_path, threshold=0.2):
    with open(coco_path) as f:
        coco = json.load(f)
    
    # 构建 image_id → image_info 映射
    img_map = {img['id']: img for img in coco['images']}
    
    # 统计每张图的最大bbox面积
    img_max_bbox = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        bbox_area = ann['bbox'][2] * ann['bbox'][3]
        if iid not in img_max_bbox or bbox_area > img_max_bbox[iid]:
            img_max_bbox[iid] = bbox_area
    
    # 筛选
    keep_ids = set()
    for iid, max_area in img_max_bbox.items():
        img = img_map[iid]
        img_area = img['width'] * img['height']
        if img_area > 0 and max_area / img_area >= threshold:
            keep_ids.add(iid)
    
    total = len(img_map)
    kept = len(keep_ids)
    
    print(f"\n筛选条件: 最大bbox面积 >= {threshold*100:.0f}% 图片面积")
    print(f"  原始: {total} images, {len(coco['annotations'])} annotations")
    print(f"  保留: {kept} ({kept/total*100:.1f}%), 丢弃: {total-kept}")
    
    # 输出筛选后的COCO
    filtered = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [img for img in coco['images'] if img['id'] in keep_ids],
        'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in keep_ids]
    }
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered, f)
    
    print(f"  输出: {output_path}")
    print(f"  最终: {len(filtered['images'])} images, {len(filtered['annotations'])} annotations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='输入COCO JSON')
    parser.add_argument('--out', required=True, help='输出路径')
    parser.add_argument('--threshold', type=float, default=0.2, help='最小覆盖率')
    args = parser.parse_args()
    filter_bbox_coverage(args.json, args.out, args.threshold)
