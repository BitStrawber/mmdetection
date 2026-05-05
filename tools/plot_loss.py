#!/usr/bin/env python3
"""从mmdet训练log中提取loss并画图"""
import re, argparse
import matplotlib.pyplot as plt

def parse_log(log_path):
    losses = []
    with open(log_path) as f:
        for line in f:
            # 匹配 mmdet 3.x 格式: ... loss: X.XXX ...
            m = re.search(r'loss:\s*([\d.]+)', line)
            if m and 'loss_rpn' not in line and 'lr' not in line.lower():
                losses.append(float(m.group(1)))
    return losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='训练log文件路径')
    parser.add_argument('--out', default=None, help='输出图片路径')
    parser.add_argument('--window', type=int, default=50, help='平滑窗口大小')
    args = parser.parse_args()
    
    losses = parse_log(args.log)
    print(f"提取到 {len(losses)} 个loss值")
    
    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.3, linewidth=0.5, label='raw')
    
    if len(losses) > args.window:
        import pandas as pd
        smooth = pd.Series(losses).rolling(args.window).mean()
        plt.plot(smooth, linewidth=1.5, label=f'smooth({args.window})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {args.log}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"保存: {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
