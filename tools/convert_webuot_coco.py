#!/usr/bin/env python3
"""WebUOT-1M 跟踪数据集 → COCO 检测格式"""
import os, sys, json, argparse
from tqdm import tqdm
import cv2

def convert_webuot_to_coco(data_root, mode='train'):
    images, annotations = [], []
    ann_id = 0
    img_id = 0
    total_frames = 0     # 总帧数
    valid_frames = 0     # 目标存在帧数
    skipped_absent = 0   # 跳过的absent帧
    
    # 类别 - WebUOT-1M是单目标跟踪，只有一个类别
    categories = [{"id": 1, "name": "object"}]
    
    data_dir = os.path.join(data_root, mode.capitalize())
    if not os.path.isdir(data_dir):
        print(f"目录不存在: {data_dir}")
        return None
    
    video_dirs = sorted(os.listdir(data_dir))
    print(f"处理 {len(video_dirs)} 个视频...")
    
    # 创建帧输出目录
    frames_dir = os.path.join(data_root, f'{mode}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    for vdir in tqdm(video_dirs):
        vpath = os.path.join(data_dir, vdir)
        if not os.path.isdir(vpath):
            continue
        
        # 读取标注
        gt_path = os.path.join(vpath, 'groundtruth_rect.txt')
        absent_path = os.path.join(vpath, 'absent.txt')
        
        if not os.path.exists(gt_path):
            continue
        
        with open(gt_path) as f:
            gt_lines = [l.strip() for l in f.readlines()]
        
        with open(absent_path) as f:
            absent_lines = [l.strip() for l in f.readlines()]
        
        # 查找视频文件
        mp4_file = None
        for fname in os.listdir(vpath):
            if fname.endswith('.mp4'):
                mp4_file = os.path.join(vpath, fname)
                break
        
        if mp4_file:
            # 从视频提取帧
            cap = cv2.VideoCapture(mp4_file)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳过目标缺失的帧
                if frame_idx < len(absent_lines) and int(absent_lines[frame_idx]) == 1:
                    frame_idx += 1
                    valid_frames += 1
                    total_frames += 1
                    skipped_absent += 1
                    continue
                
                # 获取bbox
                if frame_idx < len(gt_lines):
                    bbox_parts = gt_lines[frame_idx].split(',')
                    x, y, w, h = map(int, bbox_parts)
                    
                    if w > 0 and h > 0:
                        # 保存帧图片
                        img_filename = f"{vdir}_frame{frame_idx:06d}.jpg"
                        img_path = os.path.join(frames_dir, img_filename)
                        cv2.imwrite(img_path, frame)
                        
                        # 记录图像信息
                        h_img, w_img = frame.shape[:2]
                        images.append({
                            "id": img_id,
                            "file_name": img_filename,
                            "width": w_img,
                            "height": h_img
                        })
                        
                        # 记录标注
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        img_id += 1
                        
                        valid_frames += 1
                        total_frames += 1
                        ann_id += 1
                        img_id += 1
                
                frame_idx += 1
                total_frames += 1
            cap.release()
        else:
            # 如果已有jpg帧文件（已解帧）
            jpg_files = sorted([f for f in os.listdir(vpath) if f.endswith('.jpg')])
            for frame_idx, jpg_name in enumerate(jpg_files):
                total_frames += 1
                if frame_idx < len(absent_lines) and int(absent_lines[frame_idx]) == 1:
                    skipped_absent += 1
                    continue
                
                if frame_idx < len(gt_lines):
                    bbox_parts = gt_lines[frame_idx].split(',')
                    x, y, w, h = map(int, bbox_parts)
                    if w > 0 and h > 0:
                        jpg_path = os.path.join(vpath, jpg_name)
                        frame = cv2.imread(jpg_path)
                        h_img, w_img = frame.shape[:2]
                        
                        images.append({
                            "id": img_id,
                            "file_name": jpg_name,
                            "width": w_img,
                            "height": h_img
                        })
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                                                
                        valid_frames += 1
                        ann_id += 1
                        img_id += 1
    
    # 生成COCO JSON
    coco = {
        "info": {"year": 2024, "version": "1.0", "description": "WebUOT-1M"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    
    ann_dir = os.path.join(data_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    output_path = os.path.join(ann_dir, f'instances_{mode}_webuot.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    
    print(f"完成! {img_id} images, {ann_id} annotations")
    print(f"输出: {output_path}")
    print(f"视频数: {len(video_dirs)}")
    print(f"总帧数: {total_frames} | 有效帧(目标存在): {valid_frames} | 跳过(absent): {skipped_absent}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    args = parser.parse_args()
    
    convert_webuot_to_coco(args.data_root, args.mode)
