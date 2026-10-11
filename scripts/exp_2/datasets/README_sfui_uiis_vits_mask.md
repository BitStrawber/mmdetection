# SFUI to UIIS10K ViT-S Mask R-CNN

`run_sfui_to_uiis_vits_mask.sh` is an independent two-stage pipeline. It does
not run AquaOV255 A/B splitting or pseudo-label filtering. SFUI and UIIS are
separate stages so the SFUI definition can be changed without editing the
downstream script.

1. Convert the full ImageNet DINO ViT-S teacher checkpoint to MMDetection
   backbone format.
2. Train a ViT-S Mask R-CNN on the configured SFUI COCO annotations.
3. Export only `backbone.*` from the SFUI checkpoint.
4. Initialize a separate UIIS10K Mask R-CNN from that backbone, train on
   `instances_train.json`, select by `coco/segm_mAP` on
   `instances_val.json`, and evaluate the selected checkpoint there.

The defaults use the full ImageNet ViT-S checkpoint, 48 SFUI epochs, 24 UIIS
epochs, each base config's learning rate, and GPUs `6,7`. SFUI defaults to no validation split: all
SFUI images are used for training and the end-of-stage evaluation uses the
same annotation as a self-evaluation. That SFUI AP is not an independent
generalization result. Set `SFUI_VAL_ANN` to use a separate validation
annotation and select the SFUI checkpoint by segmentation AP. UIIS always uses
its standard train/val annotations.

## Configurable inputs

The SFUI dataset definition is controlled by `SFUI_DATA_ROOT`,
`SFUI_TRAIN_ANN`, `SFUI_TRAIN_PREFIX`, and optionally `SFUI_VAL_ANN`,
`SFUI_VAL_PREFIX`, `SFUI_TEST_ANN`, and `SFUI_TEST_PREFIX`. Paths to custom
configs can be passed using `SFUI_CONFIG` and `UIIS_CONFIG`; the runner changes
only dataset wiring, class count for SFUI Mask R-CNN heads, initialization,
epochs, learning rate, checkpoint retention, and requested LR milestones.

Other stage-level settings include `SFUI_EPOCHS`, optional `SFUI_LR`,
`SFUI_LR_MILESTONES`, `UIIS_EPOCHS`, `UIIS_LR`, `UIIS_LR_MILESTONES`,
`VITS_RAW`, `UIIS_ROOT`, `SFUI_GPUS`, `UIIS_GPUS`, `MIN_FREE_MIB`, and
`BASE_PORT`. If epoch count changes and explicit milestones are omitted,
epoch-based scheduler milestones are scaled proportionally from the base
config. If an LR override is omitted, the base config's LR is preserved. Use a
fresh `OUTPUT_ROOT` to create a distinct experimental run.

The stages can be run independently. `RUN_SFUI=1 RUN_UIIS=0` runs SFUI only.
To transfer an already-trained SFUI model, set `RUN_SFUI=0`, `RUN_UIIS=1`, and
`SFUI_CHECKPOINT=/path/to/sfui_maskrcnn_checkpoint.pth`; the runner extracts its
backbone before preparing UIIS.

## Example on fcp

This uses the already-created 0.70 AquaOV255 Easy set as SFUI. The data root is
the run's `aquaov255_instance_coco` directory because Easy annotation image
paths use the `train/images/` prefix.

```bash
cd ~/xcx/exp_2/mmdetection
conda activate /media/SSD1/conda_envs/mmdet

SFUI_RUN=/media/HDD2/XCX/exp_2/aquaov255_maskrcnn_runs/aquaov255_reuse_ab_thr070_gpu45_20261010_134254
OUTPUT_ROOT=/media/HDD2/XCX/exp_2/sfui_uiis_runs/aquaov255_thr070_vits_$(date +%Y%m%d_%H%M%S)

SFUI_DATA_ROOT="$SFUI_RUN/aquaov255_instance_coco" \
SFUI_TRAIN_ANN="$SFUI_RUN/aquaov255_easy/annotations/instances_train.json" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SFUI_GPUS=6,7 UIIS_GPUS=6,7 MIN_FREE_MIB=8192 \
CHECK_ONLY=1 \
bash scripts/exp_2/datasets/run_sfui_to_uiis_vits_mask.sh
```

After preflight, launch the same settings with `CHECK_ONLY=0` (or omit it).
By default the script waits until every selected GPU has at least 8 GiB free.
Use `RUN_TEST=0` to skip end-of-stage test commands.
