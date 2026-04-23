"""从mmdetection日志文件生成loss曲线图

用法:
    python plot_loss.py --expA /path/to/log_expA.txt --expB3 /path/to/log_expB3.txt --output /path/to/output.png
    python plot_loss.py --expA /path/to/log_expA.txt --output /path/to/output.png
"""
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path):
    epochs = []
    losses = []
    lrs = []

    pattern = re.compile(r'Epoch\(train\)\s+\[(\d+)\]\s+\[.*?\]\s+.*?loss:\s*([0-9.]+)')
    lr_pattern = re.compile(r'lr:\s*([0-9.e\-]+)')

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expA', required=True, help='Exp A训练日志路径')
    parser.add_argument('--expB3', default=None, help='Exp B-3训练日志路径')
    parser.add_argument('--expB1', default=None, help='Exp B-1训练日志路径')
    parser.add_argument('--output', default='loss_curve.png', help='输出图片路径')
    parser.add_argument('--title', default='Training Loss Curve', help='图表标题')
    args = parser.parse_args()

    plt.figure(figsize=(12, 6))

    datasets = [
        ('Exp A (ImageNet+RUOD)', args.expA, 'blue'),
        ('Exp B-3 (UWNR Pretrain+RUOD)', args.expB3, 'red'),
        ('Exp B-1 (COCO-UWNR)', args.expB1, 'green'),
    ]

    for name, path, color in datasets:
        if path and Path(path).exists():
            epochs, losses = parse_log(path)
            if epochs:
                plt.plot(epochs, losses, label=name, color=color, alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f'保存到 {args.output}')


if __name__ == '__main__':
    main()
