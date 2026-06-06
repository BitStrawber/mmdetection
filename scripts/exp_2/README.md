# exp_2 Scripts

Run scripts from the repository root:

```bash
cd ~/xcx/exp_2/mmdetection
```

## Layout

- `base/`: baseline experiment launchers such as J2/J3/J4.
- `j10/`: J10, J10 HDP/RFTM, and J10 scheme C tuning launchers.
- `uiis/`: UIIS10K easy-data and J10 full pipeline launchers.
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

GPU occupier:

```bash
bash scripts/exp_2/utils/run_exp_2_gpu_occupier.sh "2,3" 16000
```

