# Reusable Automation Scripts

These Linux Bash entry points are designed for the MMDetection server
checkout. They also locate shared helpers inside the local `.publish_repo`
snapshot when the top-level checkout is sparse.

Common guarantees:

- Inputs are explicit command-line arguments or a manifest.
- GPU groups are selectable.
- Each stage has a marker under `OUTPUT_ROOT/.state`.
- Finished stages are recovered from their final and best checkpoints.
- Interrupted MMEngine training resumes from `last_checkpoint`.
- Core input fingerprints prevent accidental reuse with different data.
- Outputs include logs, status, checkpoints, and provenance.

Run this once after copying the files to Linux:

```bash
chmod +x tools/auto_scripts/*.sh
```

## 1. SFUI to UIIS

`run_sfui_uiis_pipeline.sh` launches the existing manifest-driven two-stage
pipeline:

```text
pretrained DINO -> SFUI Mask R-CNN -> best backbone -> UIIS Mask R-CNN -> test
```

Single experiment:

```bash
tools/auto_scripts/run_sfui_uiis_pipeline.sh \
  --name sfui_ter_random_imagenet_r50 \
  --arch r50 \
  --pretrained /path/to/imagenet_r50_dino/checkpoint.pth \
  --sfui-root /path/to/sfui_ter_random \
  --uiis-root /path/to/UIIS10K/coco \
  --output-root /path/to/sfui_uiis_run \
  --r50-gpus 6,7 \
  --check-only
```

Run the same command without `--check-only` to train.

Multiple experiments:

```bash
tools/auto_scripts/run_sfui_uiis_pipeline.sh \
  --manifest tools/auto_scripts/sfui_uiis_manifest.example.tsv \
  --uiis-root /path/to/UIIS10K/coco \
  --output-root /path/to/matrix_run \
  --r50-gpus 6,7 \
  --vits-gpus 4,5
```

After an interruption, rerun the exact same command and `OUTPUT_ROOT`.
Completed rows are skipped and the interrupted row resumes from its latest
checkpoint.

## 2. Mask R-CNN AB/BA Easy Filtering

`run_maskrcnn_abba_easy.sh` performs:

```text
split -> train A on A/validate B -> filter B
      -> train B on B/validate A -> filter A
      -> merge Easy pool -> optional train/val sample
```

Example:

```bash
tools/auto_scripts/run_maskrcnn_abba_easy.sh \
  --name coco50k_sfui_task \
  --annotation /path/to/coco50k/instances.json \
  --image-root /path/to/coco50k/images \
  --gpus 4,5 \
  --threshold 0.70 \
  --output-root /path/to/maskrcnn_abba_output
```

The default final sample is SFUI-shaped:

```text
train = 11945
val   = 2107
```

Use `--no-sample` to keep only the complete Easy pool. Use `--start-stage` to
continue at a known stage and `--force-stage` to rebuild one stage.

## 3. Cascade Mask R-CNN Easy Validation

`run_cascade_easy_eval.sh` trains, validates, and tests Cascade Mask R-CNN on
an explicitly supplied Easy dataset.

Default layout:

```text
EASY_ROOT/
  annotations/instances_train.json
  annotations/instances_val.json
  images/train/
  images/val/
```

Example:

```bash
tools/auto_scripts/run_cascade_easy_eval.sh \
  --name sfui_ter_task_cascade \
  --easy-root /path/to/easy_dataset \
  --gpus 6,7 \
  --metric segm \
  --output-root /path/to/cascade_eval_output
```

For arbitrary paths:

```bash
tools/auto_scripts/run_cascade_easy_eval.sh \
  --name independent_test \
  --train-ann /path/to/train.json \
  --val-ann /path/to/val.json \
  --test-ann /path/to/test.json \
  --train-image-root /path/to/train/images \
  --val-image-root /path/to/val/images \
  --test-image-root /path/to/test/images \
  --gpus 6,7 \
  --metric segm
```

To train, validate, and test on the complete Easy pool:

```bash
tools/auto_scripts/run_cascade_easy_eval.sh \
  --name sfui_ter_task_cascade_all_easy \
  --all-easy-ann /path/to/easy/instances_all.json \
  --image-root /path/to/easy/images \
  --gpus 6,7 \
  --metric segm
```

The same annotation is automatically assigned to train, validation, and test.
Every Easy image is therefore used for training, checkpoint validation, and
final testing. This is a self-evaluation protocol, not an independent
generalization result.

The best checkpoint is selected by `coco/segm_mAP` by default. Pass
`--metric bbox` for bbox checkpoint selection.

## Legacy Specialized Scripts

`run_dfui_ruod_pipeline.sh`, `run_cascade_easy_abba.sh`, and
`run_fasterrcnn_easy_eval.sh` remain available for their original specialized
workflows. Use the three entry points above for the current SFUI experiment
matrix.

### DFUI to RUOD pipeline

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

Stages:

```text
convert -> dfui_train -> dfui_test -> export -> ruod_train -> ruod_test
```

The RUOD stage initializes from the best DFUI checkpoint's extracted backbone.
Use `--start-stage` to skip completed stages and rerun the same command to
resume unfinished MMEngine training.

Reuse an existing manually-completed DFUI run:

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

### Cascade R-CNN Easy AB/BA

```bash
tools/auto_scripts/run_cascade_easy_abba.sh \
  --name cocosyn_easy \
  --annotation /path/to/annotations/instances_coco5w.json \
  --image-root /path/to/images \
  --gpus 4,5 \
  --output-root /path/to/easy_output
```

Stages:

```text
split -> train_A -> filter_B -> train_B -> filter_A -> merge -> sample
```

### Faster R-CNN legacy self-evaluation

```bash
tools/auto_scripts/run_fasterrcnn_easy_eval.sh \
  --name cocosyn_easy_faster \
  --easy-root /path/to/easy_dataset \
  --gpus 0,1 \
  --output-root /path/to/faster_output
```

This script preserves the old protocol: train and val annotations are merged
into `instances_all.json`; all images are used for training; the same all-image
set is used for testing; and `epoch_${EPOCHS}.pth` is used instead of a best
validation checkpoint.

### Cocosyn geometry normalization

`normalize_cocosyn_geometry.py` applies the method-specific geometry policy:

```text
uwnr      generated 256x256 -> resize to source size, annotations unchanged
cut       generated 256x256 -> resize to source size, annotations unchanged
syreanet  original size     -> keep unchanged
uwdf      original size     -> keep unchanged
watergan  640x480 crop      -> regenerate bbox/mask in crop coordinates
```

Example:

```bash
python tools/auto_scripts/normalize_cocosyn_geometry.py \
  --method watergan \
  --annotation /path/to/instances_watergan_train.json \
  --generated-manifest /path/to/watergan_generated_manifest.jsonl \
  --generated-root /path/to/watergan/train_10k \
  --source-image-root /media/SSD1/XCX/coco5w/source/train \
  --output-root /path/to/normalized/watergan \
  --target-size 640x480
```

Rebuild the complete five-method pool from transfer metadata:

```bash
python tools/auto_scripts/build_corrected_cocosyn.py \
  --syn-root /media/SSD1/XCX/cocosyn50k_final \
  --metadata-root /media/SSD1/XCX/cocosyn50k_transfer/cocosyn50k_final/metadata \
  --output-root /media/SSD1/XCX/dfui_cocosyn_build/cocosyn50k_final_v3_retry01
```

The builder reads `instances_<method>_train.json` from the metadata directory,
not the removed staging tree. WaterGAN annotations are transformed into the
640x480 center-crop coordinate system.
