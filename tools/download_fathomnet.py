#!/usr/bin/env python3
"""
批量下载 FathomNet 完整数据集
参考: https://www.fathomnet.org/post/how-to-download-images-and-bounding-boxes
"""
import os, json, argparse
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from fathomnet.api import images, boundingboxes

def download_image(img, out_dir):
    url = img.url
    ext = os.path.splitext(url.split('?')[0])[-1] or '.png'
    fname = os.path.join(out_dir, f"{img.uuid}{ext}")
    if os.path.exists(fname): return fname
    try:
        urlretrieve(url, fname)
        return fname
    except:
        return None

def download_concept(concept, out_root, coco_cats, cat2id, img_id, ann_id, workers=8):
    out_dir = os.path.join(out_root, concept.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)
    
    imgs = images.find_by_concept(concept)
    if not imgs:
        return img_id, ann_id, 0
    
    # 下载图片（多线程）
    urls = [(img, out_dir) for img in imgs]
    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(lambda x: download_image(*x), urls), 
                          desc=f"  {concept} ({len(imgs)})", leave=False))
    
    # 生成COCO标注
    images_list, anns_list = [], []
    for img, fname in zip(imgs, results):
        if fname is None: continue
        images_list.append({"id": img_id, "file_name": os.path.relpath(fname, out_root),
                           "width": img.width, "height": img.height})
        for box in img.boundingBoxes:
            anns_list.append({"id": ann_id, "image_id": img_id,
                "category_id": cat2id[concept], "bbox": [box.x, box.y, box.width, box.height],
                "area": box.width * box.height, "iscrowd": 0})
            ann_id += 1
        img_id += 1
    
    return img_id, ann_id, len(images_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', default='/media/HDD0/XCX/exp_2_data/exp_2/FathomNet')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max_concepts', type=int, default=0, help='0=全部')
    args = parser.parse_args()
    
    os.makedirs(args.out_root, exist_ok=True)
    
    # 获取所有概念
    all_concepts = boundingboxes.find_concepts()
    print(f"总概念数: {len(all_concepts)}")
    
    # 统计总数
    total = boundingboxes.count_all()
    print(f"总bbox数: {total.count}")
    
    # 限制数量
    concepts = all_concepts[:args.max_concepts] if args.max_concepts > 0 else all_concepts
    
    # 构建类别映射
    cat2id = {c: i+1 for i, c in enumerate(concepts)}
    coco_cats = [{"id": cat2id[c], "name": c} for c in concepts]
    
    all_images, all_anns = [], []
    img_id, ann_id = 0, 0
    
    for concept in tqdm(concepts, desc="Overall"):
        img_id, ann_id, n = download_concept(concept, args.out_root, coco_cats, cat2id, img_id, ann_id, args.workers)
    
    # 保存COCO
    coco = {"info": {"description": "FathomNet"}, "licenses": [],
            "categories": coco_cats, "images": all_images, "annotations": all_anns}
    
    with open(os.path.join(args.out_root, 'annotations.json'), 'w') as f:
        json.dump(coco, f)
    
    print(f"\n完成! 图片: {img_id}, 标注: {ann_id}")

if __name__ == '__main__':
    main()
