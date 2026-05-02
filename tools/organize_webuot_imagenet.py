#!/usr/bin/env python3
"""
将WebUOT COCO数据集重组为ImageNet分类格式 (class_name/image.jpg)
基于已生成的COCO JSON + 未筛选的所有帧
"""
import os, sys, json, argparse, shutil
from tqdm import tqdm
from collections import defaultdict

def organize_imagenet_style(data_root, mode='train', subset='all', symlink=False, out_root=None):
    json_path = os.path.join(data_root, 'annotations', f'instances_{mode}_webuot.json')
    if not os.path.exists(json_path):
        print(f"找不到: {json_path}"); sys.exit(1)
    
    # 处理subset选择
    suffix = ''
    if subset != 'all':
        # 如果选择了筛选子集
        subset_dir = os.path.join(data_root, f'{mode}_bbox_{subset}')
        json_path = os.path.join(subset_dir, f'instances_{mode}.json')
        frames_dir = os.path.join(subset_dir, 'images')
        out_root = os.path.join(data_root, f'imagenet_{mode}_{subset}')
    if out_root is None:
        out_root = os.path.join(data_root, f'imagenet_{mode}')
    
    print(f"\n处理 {mode} -> {out_root}")
    
    if not os.path.exists(json_path):
        print(f"找不到: {json_path}"); sys.exit(1)
    
    with open(json_path) as f:
        coco = json.load(f)
    
    # 构建 id→category_name 映射
    id2name = {c['id']: c['name'] for c in coco['categories']}
    img2file = {img['id']: img['file_name'] for img in coco['images']}
    
    # 按类别分组
    cat_images = defaultdict(list)
    for ann in coco['annotations']:
        cat_name = id2name[ann['category_id']]
        img_file = img2file[ann['image_id']]
        cat_images[cat_name].append(img_file)
    
    # 创建ImageNet格式目录
    print(f"类别数: {len(cat_images)}, 总图片: {sum(len(v) for v in cat_images.values())}")
    for cat, imgs in sorted(cat_images.items()):
        cat_dir = os.path.join(out_root, cat.replace('/', '_'))
        os.makedirs(cat_dir, exist_ok=True)
        
        for img_file in tqdm(imgs, desc=cat, leave=False):
            src = os.path.join(frames_dir, img_file)
            dst = os.path.join(cat_dir, img_file)
            if os.path.exists(src):
                if symlink:
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)
            else:
                # 尝试从全量train_frames找
                src2 = os.path.join(data_root, f'{mode}_frames', img_file)
                if os.path.exists(src2):
                if symlink:
                    os.symlink(os.path.abspath(src2), dst)
                else:
                    shutil.copy2(src2, dst)
    
    # 统计
    print(f"\n输出: {out_root}")
    for cat in sorted(cat_images.keys()):
        d = os.path.join(out_root, cat.replace('/', '_'))
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"  {cat}: {n}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--mode', default='train', help='train, test, or all')
    parser.add_argument('--subset', default='all', help='筛选子集, e.g. 30pct or all')
    parser.add_argument('--symlink', action='store_true', help='使用软链接')
    args = parser.parse_args()
    
    modes = ['train', 'test'] if args.mode == 'all' else [args.mode]
    out_root = os.path.join(args.data_root, 'imagenet_all') if args.mode == 'all' else None
    
    for mode in modes:
        if out_root is None:
            out_root = os.path.join(args.data_root, f'imagenet_{mode}')
        organize_imagenet_style(args.data_root, mode, args.subset, args.symlink, out_root)
