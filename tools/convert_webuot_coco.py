#!/usr/bin/env python3
"""
WebUOT-1M → COCO 检测格式 (单进程版本)
"""
import os, sys, json, argparse, openpyxl, cv2, glob
from tqdm import tqdm

def read_excel_categories(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    headers = [str(c.value) if c.value else '' for c in next(ws.iter_rows(min_row=1, max_row=1))]
    print(f"Excel列名: {headers}")
    
    video_col = cat_col = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if video_col is None and ('video' in hl or 'name' in hl or 'id' in hl):
            video_col = i
        if cat_col is None and ('class' in hl or 'categor' in hl or 'type' in hl or 'label' in hl):
            cat_col = i
    
    if video_col is None or cat_col is None:
        video_col, cat_col = 0, 1
    
    name2cat = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[video_col] and row[cat_col]:
            name2cat[str(row[video_col]).strip()] = str(row[cat_col]).strip()
    
    categories = sorted(set(name2cat.values()))
    cat2id = {c: i+1 for i, c in enumerate(categories)}
    coco_cats = [{"id": cat2id[c], "name": c} for c in categories]
    print(f"类别数: {len(categories)}")
    return name2cat, cat2id, coco_cats


def convert_webuot_to_coco(data_root, xlsx_path, mode='train', sample_rate=1):
    name2cat, cat2id, coco_cats = read_excel_categories(xlsx_path)
    
    data_dir = os.path.join(data_root, mode.capitalize())
    frames_dir = os.path.join(data_root, f'{mode}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    video_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    print(f"处理 {len(video_dirs)} 个视频...")
    
    final_images, final_annotations = [], []
    stats = {'total': 0, 'valid': 0, 'absent': 0, 'nobox': 0}
    found, missed = 0, 0
    img_id, ann_id = 0, 0
    
    pbar = tqdm(video_dirs)
    for vdir in pbar:
        vpath = os.path.join(data_dir, vdir)
        
        if vdir not in name2cat:
            missed += 1; continue
        found += 1
        cat_id = cat2id[name2cat[vdir]]
        
        gt_path = os.path.join(vpath, 'groundtruth_rect.txt')
        absent_path = os.path.join(vpath, 'absent.txt')
        if not os.path.exists(gt_path):
            continue
        
        with open(gt_path) as f:
            gt_lines = [l.strip() for l in f]
        with open(absent_path) as f:
            absent_lines = [l.strip() for l in f]
        
        mp4_file = os.path.join(vpath, vdir + '.mp4')
        if not os.path.exists(mp4_file):
            mp4s = glob.glob(os.path.join(vpath, '*.mp4'))
            if mp4s: mp4_file = mp4s[0]
            else: continue
        
        cap = cv2.VideoCapture(mp4_file)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            stats['total'] += 1
            
            if frame_idx >= len(gt_lines):
                frame_idx += 1; continue
            if frame_idx % sample_rate != 0:
                frame_idx += 1; continue
            
            absent = int(absent_lines[frame_idx]) if frame_idx < len(absent_lines) else 0
            if absent == 1:
                stats['absent'] += 1; frame_idx += 1; continue
            
            parts = gt_lines[frame_idx].split(',')
            x, y, w, h = map(int, map(float, parts))
            if w <= 0 or h <= 0:
                stats['nobox'] += 1; frame_idx += 1; continue
            
            img_filename = f"{vdir}_f{frame_idx:06d}.jpg"
            img_path = os.path.join(frames_dir, img_filename)
            cv2.imwrite(img_path, frame)
            
            final_images.append({"id": img_id, "file_name": img_filename,
                                 "width": frame.shape[1], "height": frame.shape[0]})
            final_annotations.append({"id": ann_id, "image_id": img_id,
                "category_id": cat_id, "bbox": [x, y, w, h],
                "area": w * h, "iscrowd": 0})
            
            stats['valid'] += 1; img_id += 1; ann_id += 1; frame_idx += 1
        
        cap.release()
        pbar.set_postfix(valid=stats['valid'])
    
    coco = {"info": {"year": 2024, "version": "1.0", "description": "WebUOT-1M"},
            "licenses": [], "categories": coco_cats,
            "images": final_images, "annotations": final_annotations}
    
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    output = os.path.join(ann_dir, f'instances_{mode}_webuot.json')
    with open(output, 'w') as f:
        json.dump(coco, f)
    
    print(f"\n{'='*50}")
    print(f"{mode} 完成!")
    print(f"  视频: {len(video_dirs)} | 匹配类别: {found} | 未匹配: {missed}")
    print(f"  总帧: {stats['total']} | 有效: {stats['valid']}")
    print(f"  absent: {stats['absent']} | 无效bbox: {stats['nobox']}")
    print(f"  图片: {img_id} | 标注: {ann_id} | 类别: {len(coco_cats)}")
    print(f"  输出: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--sample_rate', type=int, default=1)
    args = parser.parse_args()
    
    xlsx = f'WebUOT-1M-{args.mode.capitalize()}.xlsx'
    xlsx_path = os.path.join(args.data_root, xlsx)
    if not os.path.exists(xlsx_path):
        print(f"找不到Excel: {xlsx_path}"); sys.exit(1)
    
    convert_webuot_to_coco(args.data_root, xlsx_path, args.mode, args.sample_rate)
