#!/usr/bin/env python3
"""
FathomNet 批量下载 (纯标准库, 无额外依赖)
"""
import os, sys, json, argparse
from urllib.request import urlretrieve, Request, urlopen
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://database.fathomnet.org/api/"

def is_biological(concept):
    """通过分类树判断是否为海洋生物"""
    from urllib.parse import quote
    for provider in ['mbari', 'fathomnet']:
        try:
            data = api_get(f"taxa/query/{provider}/{quote(concept)}")
            if not data: continue
            if any(t.get('rank') and str(t['rank']) not in ('None', '') for t in data):
                return True
        except:
            continue
    return False

def api_get(endpoint, params=None):
    from urllib.parse import urlencode
    url = BASE_URL + endpoint
    if params:
        url += '?' + urlencode(params)
    req = Request(url, headers={'Accept': 'application/json'})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def download_img(img_data, out_dir):
    url = img_data['url']
    ext = os.path.splitext(url.split('?')[0])[-1] or '.jpg'
    fname = f"{img_data['uuid']}{ext}"
    fpath = os.path.join(out_dir, fname)
    if not os.path.exists(fpath):
        try:
            urlretrieve(url, fpath)
        except:
            return None
    return fpath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/media/HDD1/XCX/exp_2/FathomNet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_concepts', type=int, default=0)
    parser.add_argument('--max_per_concept', type=int, default=0)
    parser.add_argument('--min_images', type=int, default=0, help='跳过图片数<此值的类别')
    parser.add_argument('--per_class_json', action='store_true', help='每个类别生成独立的annotations.json')
    parser.add_argument('--bio_only', action='store_true', help='只下载海洋生物（通过分类树过滤）')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # 1. 获取概念列表
    print("获取概念列表...")
    concepts = api_get("boundingboxes/list/concepts")
    
    print(f"概念数: {len(concepts)}")
    
    # 过滤空概念
    concepts = [c for c in concepts if c.strip()]
    print(f"有效概念: {len(concepts)}")
    
    # 只保留海洋生物
    if args.bio_only:
        print("检查海洋生物分类...")
        bio = []
        for c in concepts:
            if is_biological(c):
                bio.append(c)
        print(f"  海洋生物: {len(bio)} / {len(concepts)}")
        concepts = bio
    
    # 2. 获取图片总数
    count = api_get("images/count")
    print(f"图片总数: {count['count']}")
    if args.max_concepts > 0:
        concepts = concepts[:args.max_concepts]
    
    cat2id = {c: i+1 for i, c in enumerate(concepts)}
    
    all_images, all_anns = [], []
    img_id, ann_id = 0, 0
    
    for ci, concept in enumerate(concepts):
        print(f"\n[{ci+1}/{len(concepts)}] {concept}")
        
        # 2. 获取图片
        from urllib.parse import quote
        quoted_concept = quote(concept, safe='')
        imgs = api_get(f"images/query/concept/{quoted_concept}")
        
        if not imgs:
            print("  无图片")
            continue
        
        if args.min_images > 0 and len(imgs) < args.min_images:
            print(f"  跳过 (仅{len(imgs)}张, < {args.min_images})")
            continue
        
        if args.max_per_concept > 0:
            imgs = imgs[:args.max_per_concept]
        
        # 3. 下载图片
        concept_dir = os.path.join(args.out, concept.replace('/', '_'))
        os.makedirs(concept_dir, exist_ok=True)
        
        print(f"  下载 {len(imgs)} 张...")
        with ThreadPoolExecutor(args.workers) as ex:
            paths = list(ex.map(lambda d: download_img(d, concept_dir), imgs))
        
        # 4. 生成标注
        saved = 0
        class_images, class_anns = [], []
        for img_data, fpath in zip(imgs, paths):
            if fpath is None: continue
            saved += 1
            
            class_images.append({
                "id": img_id,
                "file_name": os.path.relpath(fpath, args.out),
                "width": img_data.get('width', 0),
                "height": img_data.get('height', 0)
            })
            for box in img_data.get('boundingBoxes', []):
                w, h = box['width'], box['height']
                class_anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat2id[concept],
                    "bbox": [box['x'], box['y'], w, h],
                    "area": w * h, "iscrowd": 0
                })
                ann_id += 1
            img_id += 1
        
        # 全量标注
        all_images.extend(class_images)
        all_anns.extend(class_anns)
        
        # 每个类单独保存
        if args.per_class_json:
            per_coco = {"info": {"description": f"FathomNet - {concept}"}, "licenses": [],
                        "categories": [{"id": cat2id[concept], "name": concept}],
                        "images": class_images, "annotations": class_anns}
            with open(os.path.join(concept_dir, 'annotations.json'), 'w') as f:
                json.dump(per_coco, f)
        
        print(f"  完成 ({saved} images, {ann_id} bboxes)")
        
        # 定期保存
        if (ci+1) % 50 == 0:
            coco = {"info": {"description": "FathomNet"}, "licenses": [],
                    "categories": [{"id": cat2id[c], "name": c} for c in concepts],
                    "images": all_images, "annotations": all_anns}
            with open(os.path.join(args.out, 'annotations.json'), 'w') as f:
                json.dump(coco, f)
            print(f"  已保存 annotations.json")
    
    # 5. 最终保存
    coco = {"info": {"description": "FathomNet"}, "licenses": [],
            "categories": [{"id": cat2id[c], "name": c} for c in concepts],
            "images": all_images, "annotations": all_anns}
    with open(os.path.join(args.out, 'annotations.json'), 'w') as f:
        json.dump(coco, f)
    
    print(f"\n{'='*40}")
    print(f"完成! 图片: {img_id}, 标注: {ann_id}")
    print(f"输出: {args.out}/")

if __name__ == '__main__':
    main()
