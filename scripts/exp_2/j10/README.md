# J10 Experiment Scripts

Run these scripts from the repository root, for example:

```bash
cd ~/xcx/exp_2/mmdetection
```

## Scheme C Tuning

Frozen-stage sweep:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_frozen_lr00375_parallel.sh \
  2>&1 | tee logs/j10_scheme_c_tuning/frozen_lr00375_master.log
```

LR/epoch sweep:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_lr_epoch_sweep_parallel.sh \
  2>&1 | tee logs/j10_scheme_c_tuning/f1_lr_epoch_sweep_master.log
```

Weight-decay sweep:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_wd_sweep_parallel.sh \
  2>&1 | tee logs/j10_scheme_c_tuning/f1_wd_sweep_master.log
```

Milestone/scheduler sweep:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_milestone_sweep_parallel.sh \
  2>&1 | tee logs/j10_scheme_c_tuning/f1_milestone_sweep_master.log
```

Skip GPU idle waiting when needed:

```bash
WAIT_FOR_GPUS=0 bash scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_wd_sweep_parallel.sh
```

## HDP/RFTM

Three-DFUI comparison:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_hdp_triple.sh
```
