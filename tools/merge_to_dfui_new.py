#!/usr/bin/env python3
"""
合并 RUOD easy_merged + DFUI trainval → DFUI_NEW (按DFUI比例划分)
"""
import json, os, shutil, random
from tqdm import tqdm

RUOD_ROOT = '/media/HDD0/XCX/exp_2/RUOD/coco'
DFUI_ROOT = '/media/HDD0/XCX/exp_2/dfui'
NEW_ROOT  = '/media/HDD0/XCX/exp_2/DFUI_NEW'
random.seed(42)

ruod_json = os.path.join(RUOD_ROOT, 'annotations', 'easy_merged.json')
dfui_json = os.path.join(DFUI_ROOT, 'annotations', 'instances_trainval2017.json')
ruod_img_dir = os.path.join(RUOD_ROOT, 'train')
dfui_img_dir = os.path.join(DFUI_ROOT, 'images')
new_img_dir  = os.path.join(NEW_ROOT, 'images')

os.makedirs(os.path.join(NEW_ROOT, 'annotations'), exist_ok=True)
os.makedirs(new_img_dir, exist_ok=True)

ruod = json.load(open(ruod_json))
dfui = json.load(open(dfui_json))
print(f"RUOD: {len(ruod['images'])} images, DFUI: {len(dfui['images'])} images")

# DFUI → RUOD category_id 映射
# DFUI: echinus=0, holothurian=1, scallop=2, starfish=3, waterweeds=4
# RUOD: holothurian=1, echinus=2, scallop=3, starfish=4, fish=5, corals=6, diver=7, cuttlefish=8, turtle=9, jellyfish=10
dfui_to_ruod_cat = {0: 2, 1: 1, 2: 3, 3: 4, 4: 11}

# 创建统一的categories (10+1)
unified_categories = ruod['categories'][:]
unified_categories.append({'id': 11, 'name': 'waterweeds'})
print(f"统一类别: {[c['name'] for c in unified_categories]}")

# 合并并shuffle
all_items = []
for img in ruod['images']:
    anns = [a for a in ruod['annotations'] if a['image_id'] == img['id']]
    all_items.append(('ruod', img, anns))
for img in dfui['images']:
    anns = [a for a in dfui['annotations'] if a['image_id'] == img['id']]
    all_items.append(('dfui', img, anns))

random.shuffle(all_items)
n = len(all_items)
n_train = int(n * 0.8)

splits = {
    'train': all_items[:n_train],
    'val':   all_items[n_train:]
}

for split_name, items in splits.items():
    img_id = 0
    ann_id = 0
    images_list = []
    anns_list = []
    
    print(f"\n{split_name} (n)")


    for source, img, anns in tqdm(items, desc=split_name):
        fname = os.path.basename(img['file_name'])
        
        if source == 'ruod':
            src = os.path.join(ruod_img_dir, fname)
        else:
            src = os.path.join(dfui_img_dir, fname)
        
        dst = os.path.join(new_img_dir, fname)
        # 处理重名
        if os.path.exists(dst):
            fname = f'{source}_{fname}'
            dst = os.path.join(new_img_dir, fname)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
        
        images_list.append({'id': img_id, 'file_name': fname,
                           'width': img['width'], 'height': img['height']})
        for ann in anns:
            cat_id = ann['category_id']
            if source == 'dfui':
                cat_id = dfui_to_ruod_cat.get(cat_id, cat_id)
            anns_list.append({'id': ann_id, 'image_id': img_id,
                             'category_id': cat_id,
                             'bbox': ann['bbox'], 'area': ann['area'],
                             'iscrowd': ann.get('iscrowd', 0)})
            ann_id += 1
        img_id += 1
    
    coco = {
        'info': {'description': f'DFUI_NEW {split_name}'},
        'licenses': [],
        'categories': unified_categories,
        'images': images_list,
        'annotations': anns_list
    }
    json.dump(coco, open(os.path.join(NEW_ROOT, 'annotations', f'instances_{split_name}2017.json'), 'w'))
    print(f"  {split_name}: {img_id} imgs, {ann_id} anns")

print(f"\n完成!")
print(f"  Train: {n_train} ({100*n_train/n:.0f}%)")
print(f"  Val:   {n - n_train} ({100*(n-n_train)/n:.0f}%)")
print(f"  输出: {NEW_ROOT}/")
