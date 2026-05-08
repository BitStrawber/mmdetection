#!/usr/bin/env python3
"""
筛选最大bbox占图面积>=阈值的图片
支持单JSON或FathomNet按类分文件夹格式

用法:
  # 单个COCO JSON
  python tools/filter_large_bbox.py --json path/to.json --out path/to_filtered.json -t 0.2

  # FathomNet按类分文件夹 (遍历所有class_folder/annotations.json)
  python tools/filter_large_bbox.py --fathom_dir /media/HDD1/XCX/exp_2/FathomNet --out /media/HDD1/XCX/exp_2/FathomNet/filtered_20pct.json -t 0.2
"""
import json, os, glob, argparse

def filter_coco(coco, threshold=0.2):
    img_map = {img['id']: img for img in coco['images']}
    img_max_bbox = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        area = ann['bbox'][2] * ann['bbox'][3]
        if iid not in img_max_bbox or area > img_max_bbox[iid]:
            img_max_bbox[iid] = area
    
    keep_ids = set()
    for iid, max_area in img_max_bbox.items():
        img = img_map[iid]
        img_area = img['width'] * img['height']
        if img_area > 0 and max_area / img_area >= threshold:
            keep_ids.add(iid)
    
    filtered = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [img for img in coco['images'] if img['id'] in keep_ids],
        'annotations': [a for a in coco['annotations'] if a['image_id'] in keep_ids]
    }
    return filtered, len(coco['images']), len(keep_ids)

def filter_single(json_path, output_path, threshold):
    with open(json_path) as f:
        coco = json.load(f)
    filtered, total, kept = filter_coco(coco, threshold)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered, f)
    print(f"\n筛选: {threshold*100:.0f}%")
    print(f"  原始: {total} → 保留: {kept} ({kept/total*100:.1f}%), 丢弃: {total-kept}")
    print(f"  输出: {output_path}")

def filter_fathomnet(fathom_dir, output_path, threshold):
    """遍历FathomNet各类子文件夹, 合并筛选"""
    import glob
    
    search = os.path.join(fathom_dir, '*/annotations.json')
    files = sorted(glob.glob(search))
    
    if not files:
        print(f"未找到任何类别的annotations.json: {search}")
        return
    
    print(f"找到 {len(files)} 个类别")
    
    total_all, kept_all = 0, 0
    all_filtered = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    
    for fpath in files:
        cls_name = os.path.basename(os.path.dirname(fpath))
        with open(fpath) as f:
            coco = json.load(f)
        
        filtered, total, kept = filter_coco(coco, threshold)
        total_all += total
        kept_all += kept
        
        if kept > 0:
            all_filtered['images'].extend(filtered['images'])
            all_filtered['annotations'].extend(filtered['annotations'])
            if filtered['categories']:
                for c in filtered['categories']:
                    if c not in all_filtered['categories']:
                        all_filtered['categories'].append(c)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_filtered, f)
    
    print(f"\n{'='*50}")
    print(f"筛选条件: 最大bbox >= {threshold*100:.0f}% 图片面积")
    print(f"  总类别: {len(files)}")
    print(f"  原始图片: {total_all} → 保留: {kept_all} ({kept_all/total_all*100:.1f}%)")
    print(f"  输出: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', help='单个COCO JSON路径')
    parser.add_argument('--fathom_dir', help='FathomNet按类分文件夹的根目录')
    parser.add_argument('--out', required=True, help='输出路径')
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help='最小覆盖率')
    args = parser.parse_args()
    
    if args.fathom_dir:
        filter_fathomnet(args.fathom_dir, args.out, args.threshold)
    elif args.json:
        filter_single(args.json, args.out, args.threshold)
    else:
        print("请指定 --json 或 --fathom_dir")

