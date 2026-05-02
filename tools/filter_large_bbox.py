#!/usr/bin/env python3
"""
筛选bbox覆盖率>阈值的帧，输出独立的COCO数据集
"""
import os, sys, cv2, glob, json, argparse, openpyxl
from tqdm import tqdm

def read_excel_categories(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    headers = [str(c.value) if c.value else '' for c in next(ws.iter_rows(min_row=1, max_row=1))]
    video_col = cat_col = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if video_col is None and ('video' in hl or 'name' in hl or 'id' in hl): video_col = i
        if cat_col is None and ('class' in hl or 'categor' in hl or 'type' in hl or 'label' in hl): cat_col = i
    if video_col is None or cat_col is None: video_col, cat_col = 0, 1
    name2cat = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[video_col] and row[cat_col]:
            name2cat[str(row[video_col]).strip()] = str(row[cat_col]).strip()
    return name2cat


def filter_and_export(data_root, xlsx_path, mode='train', threshold=0.3, sample_rate=1, count_only=False):
    name2cat = read_excel_categories(xlsx_path)
    
    categories = sorted(set(name2cat.values()))
    cat2id = {c: i+1 for i, c in enumerate(categories)}
    coco_cats = [{"id": cat2id[c], "name": c} for c in categories]
    
    data_dir = os.path.join(data_root, mode.capitalize())
    
    # 输出目录
    out_name = f"{mode}_bbox_{int(threshold*100)}pct"
    out_dir = os.path.join(data_root, out_name)
    frames_dir = os.path.join(out_dir, 'images')
    os.makedirs(frames_dir, exist_ok=True)
    
    video_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    
    images, annotations = [], []
    img_id, ann_id = 0, 0
    total, valid, large, absent, missed = 0, 0, 0, 0, 0
    
    pbar = tqdm(video_dirs)
    for vdir in pbar:
        vpath = os.path.join(data_dir, vdir)
        if vdir not in name2cat:
            missed += 1; continue
        cat_id = cat2id[name2cat[vdir]]
        
        gt_path = os.path.join(vpath, 'groundtruth_rect.txt')
        absent_path = os.path.join(vpath, 'absent.txt')
        if not os.path.exists(gt_path): continue
        
        with open(gt_path) as f: gt_lines = [l.strip() for l in f]
        with open(absent_path) as f: absent_lines = [l.strip() for l in f]
        
        mp4_file = os.path.join(vpath, vdir + '.mp4')
        if not os.path.exists(mp4_file):
            mp4s = glob.glob(os.path.join(vpath, '*.mp4'))
            if mp4s: mp4_file = mp4s[0]
            else: continue
        
        cap = cv2.VideoCapture(mp4_file)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_area = img_w * img_h
        
        if count_only and img_area > 0:
            # 快速模式：不逐帧解码，直接用gt_lines算
            for fi in range(0, min(len(gt_lines), total_frames), args.sample_rate):
                total += 1
                absent_val = int(absent_lines[fi]) if fi < len(absent_lines) else 0
                if absent_val == 1:
                    absent += 1; continue
                parts = gt_lines[fi].split(',')
                x, y, w, h = map(int, map(float, parts))
                if w <= 0 or h <= 0: continue
                if (w * h) / img_area > threshold:
                    large += 1
            cap.release()
            pbar.set_postfix(large=large)
            continue
        
        # 完整模式：逐帧解码
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            total += 1
            
            if frame_idx >= len(gt_lines):
                frame_idx += 1; continue
            if frame_idx % sample_rate != 0:
                frame_idx += 1; continue
            
            absent_val = int(absent_lines[frame_idx]) if frame_idx < len(absent_lines) else 0
            if absent_val == 1:
                absent += 1; frame_idx += 1; continue
            
            parts = gt_lines[frame_idx].split(',')
            x, y, w, h = map(int, map(float, parts))
            img_area = frame.shape[0] * frame.shape[1]
            bbox_area = w * h
            
            if bbox_area / img_area <= threshold:
                frame_idx += 1; continue
            
            large += 1
            if not count_only:
                img_filename = f"{vdir}_f{frame_idx:06d}.jpg"
                img_path = os.path.join(frames_dir, img_filename)
                cv2.imwrite(img_path, frame)
            else:
                img_filename = f"{vdir}_f{frame_idx:06d}.jpg"
            
            images.append({"id": img_id, "file_name": img_filename,
                          "width": frame.shape[1], "height": frame.shape[0]})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cat_id,
                              "bbox": [x, y, w, h], "area": bbox_area, "iscrowd": 0})
            img_id += 1; ann_id += 1
            frame_idx += 1
        
        cap.release()
        pbar.set_postfix(large=large, cat=name2cat.get(vdir,'?')[:15])
    
    # 输出COCO
    coco = {"info": {"year": 2024, "version": "1.0", "description": f"WebUOT-1M bbox>{int(threshold*100)}%"},
            "licenses": [], "categories": coco_cats,
            "images": images, "annotations": annotations}
    
    if not count_only:
        out_json = os.path.join(out_dir, f'instances_{mode}.json')
        with open(out_json, 'w') as f:
            json.dump(coco, f)
        print(f"  输出: {out_dir}")
        print(f"    images/ + instances_{mode}.json")
    else:
        print(f"  (仅统计模式)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--count_only', action='store_true', help='仅统计, 不导出图片')
    args = parser.parse_args()
    
    xlsx = f'WebUOT-1M-{args.mode.capitalize()}.xlsx'
    xlsx_path = os.path.join(args.data_root, xlsx)
    if not os.path.exists(xlsx_path):
        print(f"找不到Excel: {xlsx_path}"); sys.exit(1)
    
    filter_and_export(args.data_root, xlsx_path, args.mode, args.threshold, args.sample_rate, args.count_only)
