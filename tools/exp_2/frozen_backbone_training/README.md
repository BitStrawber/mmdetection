# Additional Frozen-Backbone Training

This workflow prepares four complete detectors from four source detector checkpoints:

- RUOD + ResNet-50 + DFUI
- UIIS10K + ResNet-50 + SFUI
- RUOD + ViT-S + DFUI
- UIIS10K + ViT-S + SFUI

The preparation step extracts only `backbone.*` tensors, removes the prefix,
and writes a derived MMDetection config with a frozen backbone.

## Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate detector

bash scripts/exp_2/tri_pretrain/run_additional_four_frozen_backbones.sh
```

Defaults:

```text
R50_GPUS=2,3
VITS_RUOD_GPUS=4,5
VITS_UIIS_GPUS=6,7
```

Both R50 jobs run serially, each using the two GPUs in `R50_GPUS`.
The two ViT-S jobs run in parallel on separate GPU pairs.

After each training run, the script compares the best checkpoint backbone
against the extracted backbone tensor by tensor. Any parameter or buffer
change causes the task to fail.

Outputs:

```text
frozen_backbone_additional_20261104/
├── backbones/
├── configs/
├── logs/
└── work_dirs/
```
