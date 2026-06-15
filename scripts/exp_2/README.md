# exp_2 Scripts

Run scripts from the repository root:

```bash
cd ~/xcx/exp_2/mmdetection
```

## Layout

- `base/`: baseline experiment launchers such as J2/J3/J4.
- `j10/`: J10, J10 HDP/RFTM, and J10 scheme C tuning launchers.
- `uiis/`: UIIS10K easy-data and J10 full pipeline launchers.
- `usod/`: USOD10K objectness easy-data, DFUI merge, and MAE-route launchers.
- `utils/`: GPU occupier, monitor, and watcher scripts.

## Common Commands

J10 scheme C weight-decay tuning:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_wd_sweep_parallel.sh \
  2>&1 | tee logs/j10_scheme_c_tuning/f1_wd_sweep_master.log
```

UIIS easy + J10 full pipeline:

```bash
bash scripts/exp_2/uiis/run_exp_2_uiis_easy_j10_full.sh
```

UIIS10K easy Faster R-CNN self-eval:

```bash
bash scripts/exp_2/uiis/run_exp_2_uiis_easy_faster_self_eval.sh
```

This trains Faster R-CNN on
`UIIS10K/coco/annotations/cross_split_det/easy_merged.json` and evaluates on
the same easy set. It uses bbox annotations converted from the original
instance-segmentation labels.

USOD10K objectness expansion:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh
```

USOD10K objectness expansion + J10 scheme C:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_easy_j10_scheme_c_full.sh \
  2>&1 | tee logs/j10_scheme_c_usod/full_master.log
```

USOD10K MAE-route transfer, default GPU `4,5`:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_mae_strategy.sh
```

This follows the old J3 idea: load `mae_pretrain_vit_base.pth`, train S1 on
`DFUI_RUOD_UIIS_USOD_EASY`, extract `backbone_only.pth`, then run RUOD S2 with
the same ViT-Cascade structure.

USOD10K DINO-route transfer, default GPU `4,5`:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_dino_strategy.sh \
  2>&1 | tee logs/j10_dino_usod/full_master.log
```

This uses DINO-pretrained ResNet-50 for S1 on `DFUI_RUOD_UIIS_USOD_EASY`,
extracts the ResNet backbone, then runs the existing Cascade R-CNN RUOD S2.

USOD10K combined dual strategy:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_easy_dual_strategy.sh \
  2>&1 | tee logs/j10_usod_dual_strategy/full_master.log
```

This first uses GPU `2,3` for USOD A/B filtering and merge, then runs the
original RCNN route on GPU `2,3` and the MAE ViT route on GPU `4,5` in parallel.

USOD10K RCNN auto tuning, three experiments per round:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_rcnn_auto_tuning.sh \
  2>&1 | tee logs/j10_usod_auto_tuning/auto_tuning_master.log
```

It runs Round 1 epoch/milestone sweep, Round 2 LR sweep from the Round 1 best,
and Round 3 frozen-stage/weight-decay sweep from the Round 2 best. Results are
written to `logs/j10_usod_auto_tuning/summary.tsv`.

USOD10K easy Faster R-CNN self-eval:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_easy_faster_self_eval.sh
```

This trains Faster R-CNN on `USOD10K/annotations/cross_split_det/easy_merged.json`
and evaluates on the same easy set to estimate how easy the selected subset is.

GPU occupier:

```bash
bash scripts/exp_2/utils/run_exp_2_gpu_occupier.sh "2,3" 16000
```
