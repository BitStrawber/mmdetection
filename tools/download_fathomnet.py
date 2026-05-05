#!/usr/bin/env python3
"""
FathomNet 批量下载 (两阶段: 筛选 → 下载)
"""
import os, sys, json, argparse
from urllib.request import urlretrieve, Request, urlopen
from urllib.parse import quote, urlencode
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://database.fathomnet.org/api/"


def api_get(endpoint):
    url = BASE_URL + endpoint
    req = Request(url, headers={'Accept': 'application/json'})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def is_biological(concept):
    for provider in ['mbari', 'fathomnet']:
        try:
            data = api_get(f"taxa/query/{provider}/{quote(concept)}")
            if data and any(t.get('rank') and str(t['rank']) not in ('None', '') for t in data):
                return True
        except:
            continue
    return False


def download_img(img_data, out_dir):
    url = img_data['url']
    ext = os.path.splitext(url.split('?')[0])[-1] or '.jpg'
    fpath = os.path.join(out_dir, f"{img_data['uuid']}{ext}")
    if not os.path.exists(fpath):
        try:
            urlretrieve(url, fpath)
        except:
            return None
    return fpath


def stage_discover(args):
    """Stage 1: 筛选需要下载的concept, 写入文件"""
    list_file = os.path.join(args.out, 'concepts_list.txt')
    if os.path.exists(list_file):
        print(f"概念列表已存在: {list_file}")
        print(f"跳过筛选, 直接使用已有列表")
        return
    
    print("获取概念列表...")
    concepts = api_get("boundingboxes/list/concepts")
    concepts = [c for c in concepts if c.strip()]
    print(f"有效概念: {len(concepts)}")
    
    if args.max_concepts > 0:
        concepts = concepts[:args.max_concepts]
    
    # 按图片数过滤
    if args.min_images > 0:
        print(f"过滤图片数≥{args.min_images}的类别...")
        enough = []
        for i, c in enumerate(concepts):
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(concepts)}...", end='\r')
            try:
                imgs = api_get(f"images/query/concept/{quote(c)}")
                if len(imgs) >= args.min_images:
                    enough.append(c)
            except:
                continue
        print(f"  {len(concepts)}/{len(concepts)} - 完成!")
        print(f"  满足条件: {len(enough)} / {len(concepts)}")
        concepts = enough
    
    # 只保留海洋生物
    if args.bio_only:
        print("检查海洋生物分类...")
        bio = []
        for i, c in enumerate(concepts):
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(concepts)}...", end='\r')
            if is_biological(c):
                bio.append(c)
        print(f"  {len(concepts)}/{len(concepts)} - 完成!")
        print(f"  海洋生物: {len(bio)} / {len(concepts)}")
        concepts = bio
    
    # 写入文件
    with open(list_file, 'w') as f:
        f.write('\n'.join(concepts))
    print(f"\n筛选完成! {len(concepts)} 个概念 → {list_file}")


def stage_download(args):
    """Stage 2: 根据concepts_list.txt下载"""
    list_file = os.path.join(args.out, 'concepts_list.txt')
    if not os.path.exists(list_file):
        print(f"概念列表不存在: {list_file}")
        print("请先运行 --discover 筛选概念")
        sys.exit(1)
    
    with open(list_file) as f:
        concepts = [l.strip() for l in f if l.strip()]
    print(f"读取概念列表: {len(concepts)} 个")
    
    if args.max_concepts > 0:
        concepts = concepts[:args.max_concepts]
    
    cat2id = {c: i+1 for i, c in enumerate(concepts)}
    all_images, all_anns = [], []
    img_id, ann_id = 0, 0
    
    for ci, concept in enumerate(concepts):
        print(f"\n[{ci+1}/{len(concepts)}] {concept}")
        
        try:
            imgs = api_get(f"images/query/concept/{quote(concept)}")
        except:
            print("  API错误, 跳过")
            continue
        
        if not imgs:
            print("  无图片"); continue
        if args.max_per_concept > 0:
            imgs = imgs[:args.max_per_concept]
        
        concept_dir = os.path.join(args.out, concept.replace('/', '_'))
        os.makedirs(concept_dir, exist_ok=True)
        
        print(f"  下载 {len(imgs)} 张...")
        with ThreadPoolExecutor(args.workers) as ex:
            paths = list(ex.map(lambda d: download_img(d, concept_dir), imgs))
        
        saved = 0
        class_images, class_anns = [], []
        for img_data, fpath in zip(imgs, paths):
            if fpath is None: continue
            saved += 1
            class_images.append({
                "id": img_id, "file_name": os.path.relpath(fpath, args.out),
                "width": img_data.get('width', 0), "height": img_data.get('height', 0)
            })
            for box in img_data.get('boundingBoxes', []):
                w, h = box['width'], box['height']
                class_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat2id[concept],
                    "bbox": [box['x'], box['y'], w, h], "area": w * h, "iscrowd": 0
                })
                ann_id += 1
            img_id += 1
        
        all_images.extend(class_images)
        all_anns.extend(class_anns)
        
        if args.per_class_json:
            per_coco = {"info": {"description": f"FathomNet - {concept}"}, "licenses": [],
                        "categories": [{"id": cat2id[concept], "name": concept}],
                        "images": class_images, "annotations": class_anns}
            with open(os.path.join(concept_dir, 'annotations.json'), 'w') as f:
                json.dump(per_coco, f)
        
        print(f"  完成 ({saved} images, {ann_id} bboxes)")
        
        if (ci+1) % 50 == 0:
            coco = {"info": {"description": "FathomNet"}, "licenses": [],
                    "categories": [{"id": cat2id[c], "name": c} for c in concepts],
                    "images": all_images, "annotations": all_anns}
            with open(os.path.join(args.out, 'annotations.json'), 'w') as f:
                json.dump(coco, f)
            print(f"  已保存 annotations.json")
    
    coco = {"info": {"description": "FathomNet"}, "licenses": [],
            "categories": [{"id": cat2id[c], "name": c} for c in concepts],
            "images": all_images, "annotations": all_anns}
    with open(os.path.join(args.out, 'annotations.json'), 'w') as f:
        json.dump(coco, f)
    
    print(f"\n{'='*40}")
    print(f"完成! 图片: {img_id}, 标注: {ann_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/media/HDD1/XCX/exp_2/FathomNet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_concepts', type=int, default=0)
    parser.add_argument('--max_per_concept', type=int, default=0)
    parser.add_argument('--min_images', type=int, default=0)
    parser.add_argument('--per_class_json', action='store_true')
    parser.add_argument('--bio_only', action='store_true')
    parser.add_argument('--discover', action='store_true', help='Stage1: 筛选concept并保存到文件')
    parser.add_argument('--download', action='store_true', help='Stage2: 根据文件下载 (可断点续传)')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    if args.discover:
        stage_discover(args)
    elif args.download:
        stage_download(args)
    else:
        # 默认: 先discover再download
        stage_discover(args)
        stage_download(args)
