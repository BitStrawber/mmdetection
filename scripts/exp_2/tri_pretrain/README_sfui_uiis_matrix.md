# SFUI -> UIIS Matrix Runner

`run_sfui_uiis_matrix.sh` runs independent SFUI -> UIIS10K Mask R-CNN
experiments from a pipe-delimited manifest. Each row can select a different
pretrained checkpoint, SFUI dataset, backbone, annotation pair, image prefix,
and optional base configs.

The runner preserves the two-stage contract:

1. Train the SFUI Mask R-CNN on the row's SFUI dataset.
2. Export only the SFUI `backbone.*` weights.
3. Initialize a new UIIS10K Mask R-CNN from that backbone.
4. Train and test the UIIS stage with its own 10-class heads.

The UIIS stage does not load the full SFUI detector checkpoint. The runner
waits for the selected GPU group to have `MIN_FREE_MIB` free memory before
each experiment and runs manifest rows serially.

Manifest columns:

```text
name|backbone|pretrain_checkpoint|sfui_root|train_ann|val_ann|train_prefix|val_prefix|test_ann|test_prefix|sfui_config|uiis_config|expected_sfui_classes
```

`r50` rows call `run_sfui_to_uiis_r50_mask.sh`; `vits` rows call
`run_sfui_to_uiis_vits_mask.sh`. Empty annotation fields default to
`sfui_root/annotations/instances_*.json`; empty config fields use the runner
defaults.

Example preflight:

```bash
cd ~/xcx/exp_2/mmdetection
conda activate mmdet

MANIFEST=/path/to/sfui_uiis_matrix.tsv \
UIIS_ROOT=/media/HDD0/XCX/exp_2/UIIS10K/coco \
OUTPUT_ROOT=/media/HDD2/XCX/exp_2/sfui_uiis_runs/matrix_check \
R50_GPUS=6,7 VITS_GPUS=6,7 \
CHECK_ONLY=1 \
bash scripts/exp_2/tri_pretrain/run_sfui_uiis_matrix.sh
```

Launch after preflight:

```bash
MANIFEST=/path/to/sfui_uiis_matrix.tsv \
UIIS_ROOT=/media/HDD0/XCX/exp_2/UIIS10K/coco \
OUTPUT_ROOT=/media/HDD2/XCX/exp_2/sfui_uiis_runs/matrix_$(date +%Y%m%d_%H%M%S) \
R50_GPUS=6,7 VITS_GPUS=6,7 MIN_FREE_MIB=8192 \
CHECK_ONLY=0 RUN_TEST=1 \
bash scripts/exp_2/tri_pretrain/run_sfui_uiis_matrix.sh
```
