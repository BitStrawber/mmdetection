#!/usr/bin/env python3
"""
提取WebUOT-1M所有视频的第一帧，绘制bbox + 类别名
"""
import os, cv2, glob, argparse, openpyxl
from tqdm import tqdm

def read_excel_categories(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    headers = [str(c.value) if c.value else '' for c in next(ws.iter_rows(min_row=1, max_row=1))]
    
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
    return name2cat


def extract_first_frames(data_root, output_dir, xlsx_path, mode='train'):
    name2cat = read_excel_categories(xlsx_path)
    data_dir = os.path.join(data_root, mode.capitalize())
    os.makedirs(output_dir, exist_ok=True)
    
    video_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    print(f"处理 {len(video_dirs)} 个视频, 类别 {len(set(name2cat.values()))} 种...")
    
    saved = 0
    for vdir in tqdm(video_dirs):
        vpath = os.path.join(data_dir, vdir)
        
        gt_path = os.path.join(vpath, 'groundtruth_rect.txt')
        if not os.path.exists(gt_path):
            continue
        
        with open(gt_path) as f:
            first_line = f.readline().strip()
        x, y, w, h = map(int, map(float, first_line.split(',')))
        
        mp4_file = os.path.join(vpath, vdir + '.mp4')
        if not os.path.exists(mp4_file):
            mp4s = glob.glob(os.path.join(vpath, '*.mp4'))
            if mp4s: mp4_file = mp4s[0]
            else: continue
        
        cap = cv2.VideoCapture(mp4_file)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue
        
        # bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 类别名
        cls_name = name2cat.get(vdir, 'unknown')
        cv2.putText(frame, cls_name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # 视频名（小字）
        cv2.putText(frame, vdir, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
        out_path = os.path.join(output_dir, f"{vdir}_{cls_name}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
    
    print(f"完成! 保存 {saved}/{len(video_dirs)} 到 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--output', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/first_frames')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    args = parser.parse_args()
    
    xlsx = f'WebUOT-1M-{args.mode.capitalize()}.xlsx'
    xlsx_path = os.path.join(args.data_root, xlsx)
    if not os.path.exists(xlsx_path):
        print(f"找不到Excel: {xlsx_path}"); exit(1)
    
    extract_first_frames(args.data_root, args.output, xlsx_path, args.mode)
