#!/usr/bin/env python3
"""
筛选最大bbox占图面积>=阈值的图片 (支持多种数据集格式)

用法:
  # FathomNet (按类文件夹)
  python tools/filter_large_bbox.py --fathom_dir /path/to/FathomNet -t 0.2

  # WebUOT / DFUI (训练+测试)
  python tools/filter_large_bbox.py \
      --json_train /path/to/instances_train.json \
      --json_val /path/to/instances_val.json \
      -t 0.2

  # 单个JSON
  python tools/filter_large_bbox.py --json /path/to/annotations.json -t 0.2
"""
import json, os, glob, argparse
from tqdm import tqdm


def filter_coco(coco, threshold=0.2):
    """返回 (filtered_coco, total_images, kept_images)"""
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


def process_single(json_path, threshold):
    """处理单个JSON, 自动生成输出路径"""
    with open(json_path) as f:
        coco = json.load(f)
    filtered, total, kept = filter_coco(coco, threshold)
    
    base = os.path.splitext(json_path)[0]
    out_path = f"{base}_bbox{int(threshold*100)}pct.json"
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(filtered, f)
    
    name = os.path.basename(json_path)
    print(f"  {name:40s} {total:>6d} → {kept:>6d} ({kept/total*100:5.1f}%)")
    return out_path, total, kept


def process_fathomnet(fathom_dir, threshold):
    """遍历FathomNet各类文件夹"""
    search = os.path.join(fathom_dir, '*/annotations.json')
    files = sorted(glob.glob(search))
    
    if not files:
        print(f"未找到: {search}"); return
    
    print(f"\nFathomNet: {len(files)}个类别\n")
    
    total_all, kept_all = 0, 0
    all_filtered = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    
    for fpath in tqdm(files, desc="筛选"):
        with open(fpath) as f:
            coco = json.load(f)
        filtered, total, kept = filter_coco(coco, threshold)
        total_all += total
        kept_all += kept
        
        if kept > 0:
            all_filtered['images'].extend(filtered['images'])
            all_filtered['annotations'].extend(filtered['annotations'])
            for c in filtered.get('categories', []):
                if c not in all_filtered['categories']:
                    all_filtered['categories'].append(c)
    
    out_file = f"fathomnet_bbox{int(threshold*100)}pct.json"
    out_path = os.path.join(fathom_dir, out_file)
    with open(out_path, 'w') as f:
        json.dump(all_filtered, f)
    
    print(f"\nFathomNet 总计: {total_all} → {kept_all} ({kept_all/total_all*100:.1f}%)")
    print(f"  输出: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fathom_dir', help='FathomNet根目录(含各类子文件夹)')
    parser.add_argument('--json', help='单个COCO JSON')
    parser.add_argument('--json_train', help='训练集JSON')
    parser.add_argument('--json_val', help='验证/测试集JSON')
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    args = parser.parse_args()
    
    threshold = args.threshold
    print(f"{'='*60}")
    print(f"筛选条件: 最大bbox >= {threshold*100:.0f}% 图片面积")
    print(f"{'='*60}")
    
    if args.fathom_dir:
        print("\n--- FathomNet ---")
        process_fathomnet(args.fathom_dir, threshold)
    
    if args.json_train and args.json_val:
        print("\n--- 训练集 ---")
        process_single(args.json_train, threshold)
        print("\n--- 验证集 ---")
        process_single(args.json_val, threshold)
    elif args.json:
        process_single(args.json, threshold)
    
    print(f"\n{'='*60}")
    print("全部完成!")


if __name__ == '__main__':
    main()
