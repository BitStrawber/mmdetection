# Reusable DFUI and Easy Experiment Scripts

These Bash tools support explicit datasets, pretrained checkpoints, GPU groups,
stage markers, and restartable MMEngine training.

Run `chmod +x tools/auto_scripts/*.sh` on Linux after checkout.

## DFUI to RUOD

`run_dfui_ruod_pipeline.sh` executes:

```text
convert -> dfui_train -> dfui_test -> export -> ruod_train -> ruod_test
```

Example:

```bash
tools/auto_scripts/run_dfui_ruod_pipeline.sh \
  --experiment-id ter_random_imagenet_vits \
  --arch vits \
  --pretrained /path/to/imagenet_dino_vits_100e/checkpoint.pth \
  --dfui-root /path/to/dfui_ter_random \
  --ruod-root /path/to/RUOD/coco \
  --gpus 0,1 \
  --output-root /path/to/run_output
```

The RUOD stage initializes from the best DFUI checkpoint's extracted backbone.
Each completed stage gets a marker under `OUTPUT_ROOT/.state`. Rerun the same
command to resume an unfinished stage from `last_checkpoint`.

Reuse an existing manually completed DFUI run:

```bash
tools/auto_scripts/run_dfui_ruod_pipeline.sh \
  --experiment-id ter_random_imagenet_vits_ruod_resume \
  --arch vits \
  --pretrained /path/to/imagenet_dino_vits_100e/checkpoint.pth \
  --dfui-root /path/to/dfui_ter_random \
  --ruod-root /path/to/RUOD/coco \
  --dfui-work /path/to/existing/dfui_work_dir \
  --dfui-best /path/to/existing/best_coco_bbox_mAP_epoch_25.pth \
  --start-stage export \
  --gpus 0,1 \
  --output-root /path/to/new_ruod_resume_output
```

## Cascade R-CNN Easy AB/BA

`run_cascade_easy_abba.sh` executes:

```text
split -> train_A -> filter_B -> train_B -> filter_A -> merge -> sample
```

Example:

```bash
tools/auto_scripts/run_cascade_easy_abba.sh \
  --name cocosyn_easy \
  --annotation /path/to/annotations/instances_coco5w.json \
  --image-root /path/to/images \
  --gpus 4,5 \
  --output-root /path/to/easy_output
```

Each direction and Easy filter has its own stage marker. Use
`--start-stage train_ba`, `--start-stage filter_b`, and similar values to
continue from a known point. Use `--no-sample` to stop after merging the Easy
pool.

## Faster R-CNN Legacy Self-Evaluation

`run_fasterrcnn_easy_eval.sh` preserves the legacy protocol:

```text
instances_train.json + instances_val.json
-> instances_all.json
-> all images train
-> no per-epoch validation
-> all images test on the same combined annotation
-> final epoch_N.pth
```

Example:

```bash
tools/auto_scripts/run_fasterrcnn_easy_eval.sh \
  --name cocosyn_easy_faster \
  --easy-root /path/to/easy_dataset \
  --gpus 0,1 \
  --output-root /path/to/faster_output
```

Default `--easy-root` layout:

```text
annotations/instances_train.json
annotations/instances_val.json
images/
```

This is self-evaluation, not an independent generalization estimate.
