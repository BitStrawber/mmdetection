#!/usr/bin/env python3
"""UVOT400 跟踪格式 → COCO + 20%筛选 (单文件版，不移除__pycache__)"""
import json, os, sys, glob, re
from PIL import Image

BASE = '/media/HDD1/XCX/exp_2/UVOT400'
THRESHOLD = 0.2


def main():
    for split in ['train', 'test']:
        data_dir = os.path.join(BASE, split)
        if not os.path.isdir(data_dir):
            print(f"[跳过] {data_dir} 不存在")
            continue

        # 将输出保存到数据集 split 根目录（例如 .../UVOT400/train/instances_train.json）
        out_dir = data_dir
        out_json = os.path.join(out_dir, f'instances_{split}.json')

        # 已存在则跳过
        if os.path.exists(out_json):
            print(f"[存在] {split}: {out_json}")
        else:
            images, anns = [], []
            img_id, ann_id = 0, 0

            # 遍历所有视频子文件夹
            for root, _, files in os.walk(data_dir):
                if 'groundtruth_rect.txt' not in files:
                    continue

                gt_file = os.path.join(root, 'groundtruth_rect.txt')
                # 尝试在多种候选目录查找图片（imgs, images, 根目录），支持嵌套子目录
                candidate_dirs = [os.path.join(root, 'imgs'), os.path.join(root, 'images'), root]
                img_files = []
                exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
                for d in candidate_dirs:
                    if not os.path.isdir(d):
                        continue
                    for ext in exts:
                        img_files.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
                img_files = sorted(set(img_files))
                if not img_files:
                    print(f"  [跳过] {os.path.basename(root)}: 无图片 (检查目录: {candidate_dirs})")
                    continue
                # 简要调试信息，帮助定位路径问题
                print(f"  [发现] {os.path.basename(root)}: img_dir_candidates={candidate_dirs}, found={len(img_files)} images, sample={img_files[:3]}")

                with open(gt_file) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                seq_imgs, seq_anns = 0, 0
                for i, line in enumerate(lines):
                    if i >= len(img_files):
                        break
                    # 支持多种分隔符（逗号、空白、制表符）并清理括号
                    s = line.strip()
                    s = s.replace('(', ' ').replace(')', ' ')
                    parts = re.split(r'[,\s]+', s)
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, w, h = map(float, parts[:4])
                    except Exception:
                        # 解析失败则跳过并记录
                        print(f"    [解析失败] {os.path.basename(root)} line {i}: '{line}' -> parts={parts}")
                        continue
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
                        'width': W,
                        'height': H
                    })
                    anns.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    img_id += 1
                    ann_id += 1
                    seq_imgs += 1

                print(f"  {os.path.basename(root)}: {seq_imgs} imgs (lines={len(lines)}, available_images={len(img_files)})")

            if not images:
                print(f"[无数据] {split}")
                continue

            # 保存COCO
            coco = {
                'info': {'description': f'UVOT400_{split}'},
                'licenses': [], 'categories': [{'id': 1, 'name': 'object'}],
                'images': images, 'annotations': anns
            }
            with open(out_json, 'w') as f:
                json.dump(coco, f)
            print(f"[完成] {split}: {len(images)} imgs, {len(anns)} anns → {out_json}")

        # 筛选20%
        with open(out_json) as f:
            coco = json.load(f)
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
            if ia > 0 and ma / ia >= THRESHOLD:
                keep.add(iid)
        filtered = {
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', []),
            'categories': coco['categories'],
            'images': [i for i in coco['images'] if i['id'] in keep],
            'annotations': [a for a in coco['annotations'] if a['image_id'] in keep]
        }
        out_filtered = out_json.replace('.json', f'_bbox{int(THRESHOLD*100)}pct.json')
        with open(out_filtered, 'w') as f:
            json.dump(filtered, f)
        total, kept = len(coco['images']), len(filtered['images'])
        print(f"[筛选] {split}: {total} → {kept} ({kept/total*100:.1f}%)")


if __name__ == '__main__':
    main()
