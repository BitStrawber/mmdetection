#!/usr/bin/env python3
"""
合并 RUOD easy_merged + DFUI trainval → DFUI_NEW (COCO格式)
图片复制 + JSON合并
"""
import json, os, shutil
from tqdm import tqdm

RUOD_ROOT = '/media/HDD0/XCX/exp_2/RUOD/coco'
DFUI_ROOT = '/media/HDD0/XCX/exp_2/dfui'
NEW_ROOT  = '/media/HDD0/XCX/exp_2/DFUI_NEW'

ruod_json = os.path.join(RUOD_ROOT, 'annotations', 'easy_merged.json')
dfui_json = os.path.join(DFUI_ROOT, 'annotations', 'instances_trainval2017.json')

ruod_img_dir = os.path.join(RUOD_ROOT, 'train')
dfui_img_dir = os.path.join(DFUI_ROOT, 'images')
new_img_dir  = os.path.join(NEW_ROOT, 'images')

os.makedirs(os.path.join(NEW_ROOT, 'annotations'), exist_ok=True)
os.makedirs(new_img_dir, exist_ok=True)

# 加载JSON
ruod = json.load(open(ruod_json))
dfui = json.load(open(dfui_json))

print(f"RUOD: {len(ruod['images'])} images")
print(f"DFUI: {len(dfui['images'])} images")

# 重新分配全局ID
img_id = 0
ann_id = 0
all_images = []
all_annotations = []

# === 处理 RUOD ===
print("复制RUOD图片...")
for img in tqdm(ruod['images']):
    fname = os.path.basename(img['file_name'])
    src = os.path.join(ruod_img_dir, fname)
    dst = os.path.join(new_img_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    
    old_id = img['id']
    all_images.append({'id': img_id, 'file_name': fname,
                        'width': img['width'], 'height': img['height']})
    
    for ann in ruod['annotations']:
        if ann['image_id'] == old_id:
            all_annotations.append({'id': ann_id, 'image_id': img_id,
                                     'category_id': ann['category_id'],
                                     'bbox': ann['bbox'], 'area': ann['area'],
                                     'iscrowd': ann.get('iscrowd', 0)})
            ann_id += 1
    img_id += 1

# === 处理 DFUI ===
print("复制DFUI图片...")
for img in tqdm(dfui['images']):
    fname = os.path.basename(img['file_name'])
    src = os.path.join(dfui_img_dir, fname)
    dst = os.path.join(new_img_dir, fname)
    if os.path.exists(src):
        # 如果重名，加前缀
        if os.path.exists(dst):
            fname = f'dfui_{fname}'
            dst = os.path.join(new_img_dir, fname)
        shutil.copy2(src, dst)
    
    old_id = img['id']
    all_images.append({'id': img_id, 'file_name': fname,
                        'width': img['width'], 'height': img['height']})
    
    for ann in dfui['annotations']:
        if ann['image_id'] == old_id:
            all_annotations.append({'id': ann_id, 'image_id': img_id,
                                     'category_id': ann['category_id'],
                                     'bbox': ann['bbox'], 'area': ann['area'],
                                     'iscrowd': ann.get('iscrowd', 0)})
            ann_id += 1
    img_id += 1

# 保存合并JSON
merged = {
    'info': {'description': 'RUOD filtered + DFUI trainval'},
    'licenses': [],
    'categories': ruod['categories'],  # 用RUOD的10类
    'images': all_images,
    'annotations': all_annotations
}

json.dump(merged, open(os.path.join(NEW_ROOT, 'annotations', 'instances_train.json'), 'w'))
print(f"\n完成: {img_id} images, {ann_id} annotations")
print(f"输出: {NEW_ROOT}/")
print(f"  images/")
print(f"  annotations/instances_train.json")
