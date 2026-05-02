#!/usr/bin/env python3
"""
提取WebUOT-1M所有视频的第一帧 + bbox标注，可视化存到指定文件夹
"""
import os, cv2, glob, argparse
from tqdm import tqdm

def extract_first_frames(data_root, output_dir, mode='train'):
    data_dir = os.path.join(data_root, mode.capitalize())
    os.makedirs(output_dir, exist_ok=True)
    
    video_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    print(f"处理 {len(video_dirs)} 个视频...")
    
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
        
        if not ret:
            continue
        
        # 画bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, vdir, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out_path = os.path.join(output_dir, f"{vdir}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
    
    print(f"完成! 保存 {saved}/{len(video_dirs)} 张到 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M')
    parser.add_argument('--output', default='/media/HDD0/XCX/exp_2_data/exp_2/WebUOT-1M/first_frames')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    args = parser.parse_args()
    extract_first_frames(args.data_root, args.output, args.mode)
