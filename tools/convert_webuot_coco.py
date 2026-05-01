#!/usr/bin/env python3
"""
WebUOT-1M 跟踪数据集 → COCO 检测格式
使用Excel中的类别名 + groundtruth_rect.txt的bbox + 视频抽帧
"""
import os, sys, json, argparse, openpyxl
from tqdm import tqdm
import cv2

def read_excel_categories(xlsx_path):
    """读取Excel，返回 {视频名: 类别名} 映射"""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    
    # 找表头
    headers = [str(c.value) if c.value else '' for c in next(ws.iter_rows(min_row=1, max_row=1))]
    print(f"Excel列名: {headers}")
    
    # 找视频名和类别列
    video_col = cat_col = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if 'video' in hl or 'name' in hl or 'id' in hl:
            video_col = i
        if 'class' in hl or 'categor' in hl or 'type' in hl or 'label' in hl or 'object' in hl or 'track' in hl:
            cat_col = i
    
    if video_col is None or cat_col is None:
        print(f"警告: 无法自动识别列，使用前两列 (video={headers[0]}, category={headers[1]})")
        video_col, cat_col = 0, 1
    
    name2cat = {}
    all_cats = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[video_col]:
            vname = str(row[video_col]).strip()
            if row[cat_col]:
                cat = str(row[cat_col]).strip()
                name2cat[vname] = cat
                all_cats.add(cat)
    
    # 排序类别，分配id
    categories = sorted(all_cats)
    cat2id = {c: i+1 for i, c in enumerate(categories)}
    
    coco_cats = [{"id": cat2id[c], "name": c} for c in categories]
    print(f"类别数: {len(categories)}")
    print(f"类别: {categories[:10]}...")
    
    return name2cat, cat2id, coco_cats


def convert_webuot_to_coco(data_root, xlsx_path, mode='train', sample_rate=1):
    """主转换函数"""
    # 读取Excel类别信息
    name2cat, cat2id, coco_cats = read_excel_categories(xlsx_path)
    
    images, annotations = [], []
    ann_id, img_id = 0, 0
    total_frames, valid_frames, skipped_absent, skipped_nobox = 0, 0, 0, 0
    found_cat, not_found_cat = 0, 0
    
    data_dir = os.path.join(data_root, mode.capitalize())
    if not os.path.isdir(data_dir):
        print(f"目录不存在: {data_dir}")
        return
    
    video_dirs = sorted(os.listdir(data_dir))
    print(f"处理 {len(video_dirs)} 个视频...")
    
    # 帧输出目录
    frames_dir = os.path.join(data_root, f'{mode}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    for vdir in tqdm(video_dirs):
        vpath = os.path.join(data_dir, vdir)
        if not os.path.isdir(vpath):
            continue
        
        # 获取该视频的类别
        if vdir not in name2cat:
            not_found_cat += 1
            continue
        category_id = cat2id[name2cat[vdir]]
        found_cat += 1
        
        # 读取标注
        gt_path = os.path.join(vpath, 'groundtruth_rect.txt')
        absent_path = os.path.join(vpath, 'absent.txt')
        if not os.path.exists(gt_path):
            continue
        
        with open(gt_path) as f:
            gt_lines = [l.strip() for l in f.readlines()]
        with open(absent_path) as f:
            absent_lines = [l.strip() for l in f.readlines()]
        
        # 检查是否已有解帧图片(imgs/)
        imgs_dir = os.path.join(vpath, 'imgs')
        mp4_file = None
        if os.path.isdir(imgs_dir) and os.listdir(imgs_dir):
            # 已有解帧图片
            jpg_files = sorted(os.listdir(imgs_dir))
            for frame_idx, jpg_name in enumerate(jpg_files):
                total_frames += 1
                if frame_idx >= len(gt_lines):
                    break
                
                # 采样
                if frame_idx % sample_rate != 0:
                    continue
                
                if frame_idx < len(absent_lines) and int(absent_lines[frame_idx]) == 1:
                    skipped_absent += 1
                    continue
                
                bbox_parts = gt_lines[frame_idx].split(',')
                x, y, w, h = map(int, map(float, bbox_parts))
                if w <= 0 or h <= 0:
                    skipped_nobox += 1
                    continue
                
                img_filename = f"{vdir}_f{frame_idx:06d}.jpg"
                img_path = os.path.join(frames_dir, img_filename)
                
                src_img = os.path.join(imgs_dir, jpg_name)
                frame = cv2.imread(src_img)
                if frame is None:
                    continue
                cv2.imwrite(img_path, frame)
                
                h_img, w_img = frame.shape[:2]
                images.append({"id": img_id, "file_name": img_filename,
                              "width": w_img, "height": h_img})
                annotations.append({"id": ann_id, "image_id": img_id,
                    "category_id": category_id, "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": 0})
                ann_id += 1; img_id += 1; valid_frames += 1
        else:
            # 需要从mp4抽帧
            for fname in os.listdir(vpath):
                if fname.endswith('.mp4'):
                    mp4_file = os.path.join(vpath, fname)
                    break
            
            if not mp4_file:
                continue
            
            cap = cv2.VideoCapture(mp4_file)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames += 1
                
                if frame_idx >= len(gt_lines):
                    frame_idx += 1
                    continue
                
                # 采样
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue
                
                if frame_idx < len(absent_lines) and int(absent_lines[frame_idx]) == 1:
                    skipped_absent += 1
                    frame_idx += 1
                    continue
                
                bbox_parts = gt_lines[frame_idx].split(',')
                x, y, w, h = map(int, map(float, bbox_parts))
                if w <= 0 or h <= 0:
                    skipped_nobox += 1
                    frame_idx += 1
                    continue
                
                img_filename = f"{vdir}_f{frame_idx:06d}.jpg"
                img_path = os.path.join(frames_dir, img_filename)
                cv2.imwrite(img_path, frame)
                
                h_img, w_img = frame.shape[:2]
                images.append({"id": img_id, "file_name": img_filename,
                              "width": w_img, "height": h_img})
                annotations.append({"id": ann_id, "image_id": img_id,
                    "category_id": category_id, "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": 0})
                ann_id += 1; img_id += 1; valid_frames += 1
                frame_idx += 1
            cap.release()
    
    # 生成COCO JSON
    coco = {
        "info": {"year": 2024, "version": "1.0", "description": "WebUOT-1M"},
        "licenses": [],
        "categories": coco_cats,
        "images": images,
        "annotations": annotations
    }
    
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    output_path = os.path.join(ann_dir, f'instances_{mode}_webuot.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    
    print(f"\n{'='*50}")
    print(f"完成! mode={mode}")
    print(f"  输出: {output_path}")
    print(f"  视频数: {len(video_dirs)} | 匹配到类别: {found_cat} | 未匹配: {not_found_cat}")
    print(f"  总帧数: {total_frames} | 有效帧: {valid_frames}")
    print(f"  跳过(absent): {skipped_absent} | 跳过(无效bbox): {skipped_nobox}")
    print(f"  最终图片: {img_id} | 标注: {ann_id}")
    print(f"  类别数: {len(coco_cats)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--sample_rate', type=int, default=1, 
                       help='采样率, 1=每帧, 5=每5帧取1帧 (减少数据量)')
    args = parser.parse_args()
    
    xlsx_name = f'WebUOT-1M-{args.mode.capitalize()}.xlsx'
    xlsx_path = os.path.join(args.data_root, xlsx_name)
    
    if not os.path.exists(xlsx_path):
        print(f"Excel不存在: {xlsx_path}")
        sys.exit(1)
    
    convert_webuot_to_coco(args.data_root, xlsx_path, args.mode, args.sample_rate)
