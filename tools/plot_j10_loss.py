#!/usr/bin/env python3
"""J10 两阶段 loss 图"""
import re, matplotlib.pyplot as plt, pandas as pd

def parse_log_split(log_path, start1, start2):
    with open(log_path) as f:
        lines = f.readlines()
    
    s1, s2 = [], []
    stage = 0  # 0=before S1, 1=S1, 2=S2
    for line in lines:
        if start1 in line: stage = 1
        if start2 in line: stage = 2
        if stage == 0: continue
        
        m = re.search(r' loss:\s*([\d.]+)', line)
        if m:
            if stage == 1: s1.append(float(m.group(1)))
            else: s2.append(float(m.group(1)))
    return s1, s2

log = 'j10_run.log'
s1_losses, s2_losses = parse_log_split(log, '>>> Stage 1', '>>> Stage 2')

print(f"J10 S1 (DFUI): {len(s1_losses)} losses")
print(f"J10 S2 (RUOD): {len(s2_losses)} losses")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, losses, title in zip(axes, [s1_losses, s2_losses], 
                              ['J10 S1: DFUI (48 epoch)', 'J10 S2: RUOD (24 epoch)']):
    ax.plot(losses, alpha=0.2, linewidth=0.5, color='steelblue')
    if len(losses) > 50:
        smooth = pd.Series(losses).rolling(50).mean()
        ax.plot(smooth, linewidth=1.5, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('j10_loss.png', dpi=150)
print("保存: j10_loss.png")
plt.show()
