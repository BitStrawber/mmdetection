#!/usr/bin/env python3
"""
FathomNet 批量下载 (纯标准库, 无额外依赖)
"""
import os, sys, json, argparse
from urllib.request import urlretrieve, Request, urlopen
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://database.fathomnet.org/"

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
    parser.add_argument('--out', default='/media/HDD0/XCX/exp_2_data/exp_2/FathomNet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_concepts', type=int, default=0)
    parser.add_argument('--max_per_concept', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # 1. 获取概念列表
    print("获取概念列表...")
    try:
        concepts = api_get("boundingboxes/concepts")
    except:
        print("API不可用，尝试备用接口...")
        concepts = ["fish", "coral", "jellyfish", "shark", "turtle"]  # 后备
    
    print(f"概念数: {len(concepts)}")
    if args.max_concepts > 0:
        concepts = concepts[:args.max_concepts]
    
    cat2id = {c: i+1 for i, c in enumerate(concepts)}
    
    all_images, all_anns = [], []
    img_id, ann_id = 0, 0
    
    for ci, concept in enumerate(concepts):
        print(f"\n[{ci+1}/{len(concepts)}] {concept}")
        
        # 2. 获取图片
        try:
            imgs = api_get("images/find", {"concept": concept})
        except:
            print("  API错误，跳过")
            continue
        
        if not imgs:
            print("  无图片")
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
        for img_data, fpath in zip(imgs, paths):
            if fpath is None: continue
            saved += 1
            
            all_images.append({
                "id": img_id,
                "file_name": os.path.relpath(fpath, args.out),
                "width": img_data.get('width', 0),
                "height": img_data.get('height', 0)
            })
            for box in img_data.get('boundingBoxes', []):
                w, h = box['width'], box['height']
                all_anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat2id[concept],
                    "bbox": [box['x'], box['y'], w, h],
                    "area": w * h, "iscrowd": 0
                })
                ann_id += 1
            img_id += 1
        
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
