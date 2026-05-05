#!/usr/bin/env python3
"""J10 两阶段 loss 图 (分开保存)"""
import re, matplotlib.pyplot as plt

def parse_log_split(log_path, start1, start2):
    with open(log_path) as f:
        lines = f.readlines()
    s1, s2 = [], []
    stage = 0
    for line in lines:
        if start1 in line: stage = 1
        if start2 in line: stage = 2
        if stage == 0: continue
        m = re.search(r' loss:\s*([\d.]+)', line)
        if m:
            (s1 if stage == 1 else s2).append(float(m.group(1)))
    return s1, s2

log = 'j10_run.log'
s1_losses, s2_losses = parse_log_split(log, '>>> Stage 1', '>>> Stage 2')

for losses, name in [(s1_losses, 'J10_S1_DFUI_48ep'), (s2_losses, 'J10_S2_RUOD_24ep')]:
    print(f"{name}: {len(losses)} losses")
    
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(losses, alpha=0.4, linewidth=0.5, color='#E74C3C')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(name.replace('_', ' '), fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=150)
    print(f"  保存: {name}.png")
    plt.close()
