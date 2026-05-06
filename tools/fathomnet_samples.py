#!/usr/bin/env python3
"""
FathomNet: 每类选1张图，画bbox，统计bbox占比
"""
import os, json, cv2, argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD1/XCX/exp_2/FathomNet')
    parser.add_argument('--out', default='/media/HDD1/XCX/exp_2/FathomNet/samples')
    args = parser.parse_args()
    
    ann_file = os.path.join(args.data_root, 'annotations.json')
    if not os.path.exists(ann_file):
        print(f"找不到: {ann_file}"); return
    
    with open(ann_file) as f:
        data = json.load(f)
    
    id2name = {c['id']: c['name'] for c in data['categories']}
    
    # 按类别分组
    cat_imgs = defaultdict(list)
    for ann in data['annotations']:
        cat_imgs[ann['category_id']].append(ann['image_id'])
    
    print(f"类别数: {len(cat_imgs)}")
    os.makedirs(args.out, exist_ok=True)
    
    # 每个类别取第一张图的所有bbox
    img_map = {img['id']: img for img in data['images']}
    ratios = []
    
    for cat_id, img_ids in sorted(cat_imgs.items()):
        cat_name = id2name[cat_id]
        img_id = img_ids[0]
        img_info = img_map[img_id]
        img_path = os.path.join(args.data_root, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
        
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        h, w = frame.shape[:2]
        img_area = w * h
        total_bbox_area = 0
        
        # 画该图的所有bbox
        for ann in data['annotations']:
            if ann['image_id'] != img_id: continue
            x, y, bw, bh = ann['bbox']
            x, y, bw, bh = int(x), int(y), int(bw), int(bh)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, cat_name[:30], (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            total_bbox_area += bw * bh
        
        ratio = total_bbox_area / img_area * 100
        ratios.append((cat_name, ratio, w, h))
        
        out_path = os.path.join(args.out, f"{cat_name.replace('/', '_')[:50]}.jpg")
        cv2.imwrite(out_path, frame)
    
    # 排序输出统计
    ratios.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print(f"{'Class':<40s} {'Bbox%':>6s}  {'WxH':>12s}")
    print(f"{'-'*60}")
    
    bins = [("0-10%", 0, 10), ("10-30%", 10, 30), ("30-50%", 30, 50),
            (">50%", 50, 100)]
    counts = {b[0]: 0 for b in bins}
    
    for name, ratio, w, h in ratios:
        print(f"{name[:38]:<40s} {ratio:5.1f}%  {w:>5d}x{h:<5d}")
        for label, lo, hi in bins:
            if lo <= ratio < hi:
                counts[label] += 1
                break
    
    print(f"\n--- 分布统计 ---")
    for label in [b[0] for b in bins]:
        print(f"  {label}: {counts[label]} ({counts[label]/len(ratios)*100:.1f}%)")
    
    print(f"\n样本保存到: {args.out}/")
    print(f"总计: {len(ratios)} 类")

if __name__ == '__main__':
    main()
