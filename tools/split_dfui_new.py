#!/usr/bin/env python3
"""
对 DFUI_NEW 的 instances_train.json 按 85%/15% 随机划分 train/val
用法:
    python tools/split_dfui_new.py [--ratio 0.15] [--seed 42]
"""
import json, os, random, argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Split DFUI_NEW into train/val')
    parser.add_argument('--ann', default='/media/HDD0/XCX/exp_2/DFUI_NEW/annotations/instances_train.json',
                        help='输入标注文件路径')
    parser.add_argument('--output-dir', default='/media/HDD0/XCX/exp_2/DFUI_NEW/annotations/',
                        help='输出目录')
    parser.add_argument('--ratio', type=float, default=0.15,
                        help='验证集比例 (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"加载: {args.ann}")
    coco = json.load(open(args.ann))
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    print(f"  总图片: {len(images)}")
    print(f"  总标注: {len(annotations)}")
    print(f"  类别: {[c['name'] for c in categories]}")

    # 按图片ID建立标注索引
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann['image_id']].append(ann)

    # 打乱图片
    random.shuffle(images)

    # 划分
    val_count = max(1, int(len(images) * args.ratio))
    val_images = images[:val_count]
    train_images = images[val_count:]

    val_ids = set(img['id'] for img in val_images)
    train_ids = set(img['id'] for img in train_images)

    # 重新编号图片ID和标注ID（确保连续）
    def rebuild_split(img_list, id_set, prefix='train'):
        new_images = []
        new_annotations = []
        img_id_map = {}
        new_img_id = 0
        new_ann_id = 0

        for img in img_list:
            old_id = img['id']
            img_id_map[old_id] = new_img_id
            new_img = dict(img)
            new_img['id'] = new_img_id
            new_images.append(new_img)

            for ann in ann_by_img[old_id]:
                new_ann = dict(ann)
                new_ann['id'] = new_ann_id
                new_ann['image_id'] = new_img_id
                new_annotations.append(new_ann)
                new_ann_id += 1

            new_img_id += 1

        return {
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', []),
            'categories': categories,
            'images': new_images,
            'annotations': new_annotations
        }

    train_coco = rebuild_split(train_images, train_ids, 'train')
    val_coco = rebuild_split(val_images, val_ids, 'val')

    # 写出
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'instances_train.json')
    val_path = os.path.join(args.output_dir, 'instances_val.json')

    json.dump(train_coco, open(train_path, 'w'))
    json.dump(val_coco, open(val_path, 'w'))

    print(f"\n划分完成:")
    print(f"  训练集: {len(train_coco['images'])} 图片, {len(train_coco['annotations'])} 标注 → {train_path}")
    print(f"  验证集: {len(val_coco['images'])} 图片, {len(val_coco['annotations'])} 标注 → {val_path}")
    print(f"  验证集比例: {len(val_images)}/{len(images)} = {len(val_images)/len(images)*100:.1f}%")


if __name__ == '__main__':
    main()
