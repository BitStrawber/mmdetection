# exp_2 Project Structure and Server Paths

This document records the repository layout, experiment scripts, data roots,
logs, checkpoints, and generated files used by the `exp_2` branch. Paths are
written for the server environment unless explicitly marked as local.

## Repository Root

Server repository:

```bash
${REPO_ROOT}
```

Current Git branch:

```bash
exp_2
```

Typical sync command:

```bash
cd ${REPO_ROOT}
git pull origin exp_2
```

## Path Placeholders

This public document uses placeholders instead of real server paths. Set them
on the server before copying commands directly:

```bash
export REPO_ROOT="/path/to/mmdetection"
export EXP2_DATA_ROOT_HDD0="/path/to/main_exp2_data"
export EXP2_DATA_ROOT_HDD1="/path/to/large_exp2_data"
export EXP2_DATA_ROOT_SSD1="/path/to/ssd_exp2_data"
export CONDA_ENV_ROOT="/path/to/conda_envs"
export MMDET_ENV="${CONDA_ENV_ROOT}/mmdetection"
export MMPRETRAIN_ENV="${CONDA_ENV_ROOT}/mmpretrain"
export PRETRAINED_WEIGHTS_ROOT="/path/to/pretrained_weights"
```

Private local copies may replace these placeholders with concrete server paths.

## Main Repository Directories

```text
configs/exp_2/        Custom experiment configs.
scripts/exp_2/        Experiment launchers and utility scripts.
tools/                Dataset conversion, filtering, merging, and analysis tools.
logs/                 Runtime logs on the server. Usually not committed.
work_dirs/            Checkpoints and MMDetection work directories. Usually not committed.
exports/              Generated inspection samples, zips, and temporary exports.
third_party/          Vendored or external training code used by some routes.
mmdet/                MMDetection source code.
```

## Quick Start Commands

All commands below assume the server repository root:

```bash
cd ${REPO_ROOT}
git pull origin exp_2
```

Run one tri-pretrain S1 self-supervised pretraining task on eight GPUs:

```bash
EXP_ID=j11 GPU_IDS=0,1,2,3,4,5,6,7 \
BUILD_REALUW_SSL=0 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

Run J7 DINO ResNet-50 S1 without GPU waiting, useful when GPUs are already
reserved manually:

```bash
EXP_ID=j7 GPU_IDS=0,1,2,3,4,5,6,7 \
BUILD_REALUW_SSL=0 WAIT_FOR_GPUS=0 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

Run downstream RUOD detection and UIIS10K mask training for one finished S1
checkpoint. Detection and mask use two independent 2-GPU groups by default:

```bash
EXP_ID=j11 DET_GPU_IDS=0,1 MASK_GPU_IDS=2,3 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_downstream_pair.sh
```

Run the stage-wise J6/J7/J11/J13 downstream pipeline. S1 must already have
created the converted backbone checkpoints under `../pretrained_weights/`.

```bash
TASKS="j6 j7 j11 j13" \
GPU_GROUPS="0,1 2,3 4,5 6,7" \
BUILD_FIRST=0 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_j6_j7_j11_j13_det_then_mask.sh
```

Run J10 scheme-C on the merged DFUI/RUOD/UIIS easy source:

```bash
bash scripts/exp_2/j10/run_exp_2_j10_scheme_c.sh
```

Run the UIIS easy-data pipeline:

```bash
bash scripts/exp_2/uiis/run_exp_2_uiis_easy_j10_full.sh
```

Run the USOD easy merge pipeline:

```bash
bash scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh
```

Start a static GPU memory occupier only when each selected GPU has less than
15 GB already used. This is normally used to discourage other jobs from
starting on the same GPUs.

```bash
GPU_START_MAX_USED_MB=15000 \
GPU_MAX_UTIL=100 \
GPU_IDLE_CHECKS=1 \
GPU_WAIT_INTERVAL=30 \
OCCUPY_TARGET_UTIL=0 \
bash scripts/exp_2/utils/run_exp_2_gpu_occupier.sh "0,1,2,3,4,5,6,7" 8000
```

Diagnose RealUW imagefolder symlink, disk, decode, and GPU/IO state:

```bash
python tools/diagnose_realuw_io.py --iostat \
  2>&1 | tee logs/tri_pretrain/realuw_io_diagnosis.log
```

Download ImageNet-1K from Kaggle after accepting the competition rules in the
browser:

```bash
mkdir -p ${EXP2_DATA_ROOT_HDD1}/imagenet1k
kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p ${EXP2_DATA_ROOT_HDD1}/imagenet1k \
  2>&1 | tee logs/download_imagenet1k_kaggle.log
```

## RealUW SSL Dataset Build

RealUW SSL is the unlabeled underwater image pool used by tri-pretrain S1.
The builder script is:

```text
tools/build_realuw_ssl_dataset.py
```

Default output root:

```text
${EXP2_DATA_ROOT_HDD1}/REALUW_SSL
```

The standard build command is:

```bash
python tools/build_realuw_ssl_dataset.py \
  --preset exp2_bbox20pct \
  --out-root ${EXP2_DATA_ROOT_HDD1}/REALUW_SSL \
  --val-ratio 0 \
  --write-imagefolder
```

Expected important outputs:

```text
${EXP2_DATA_ROOT_HDD1}/REALUW_SSL/meta/train.txt
${EXP2_DATA_ROOT_HDD1}/REALUW_SSL/imagefolder/train/realuw/
```

By default, the imagefolder uses symlinks instead of copying image bytes. This
keeps storage use low, but real training reads still go back to the original
dataset disks. If those sources are on HDD, DINO/MAE pretraining can become IO
and JPEG-decode sensitive.

To skip rebuilding an existing RealUW SSL dataset:

```bash
BUILD_REALUW_SSL=0 EXP_ID=j7 GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

To move the imagefolder to SSD and materialize real files instead of symlinks:

```bash
mkdir -p ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL

rsync -aL --info=progress2 \
  ${EXP2_DATA_ROOT_HDD1}/REALUW_SSL/imagefolder/ \
  ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL/imagefolder/

rsync -a --info=progress2 \
  ${EXP2_DATA_ROOT_HDD1}/REALUW_SSL/annotations/ \
  ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL/annotations/

rsync -a --info=progress2 \
  ${EXP2_DATA_ROOT_HDD1}/REALUW_SSL/meta/ \
  ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL/meta/
```

Check that SSD imagefolder files are no longer symlinks:

```bash
find ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL/imagefolder/train/realuw \
  -maxdepth 1 -type l | wc -l

find ${EXP2_DATA_ROOT_SSD1}/REALUW_SSL/imagefolder/train/realuw \
  -maxdepth 1 -type f | wc -l
```

Run S1 from the SSD copy:

```bash
REALUW_SSL_ROOT=${EXP2_DATA_ROOT_SSD1}/REALUW_SSL \
BUILD_REALUW_SSL=0 \
EXP_ID=j7 GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/exp_2/tri_pretrain/run_exp_2_tri_pretrain_s1.sh
```

## Tri-pretrain Task Matrix

Tri-pretrain uses a two-stage idea:

```text
S1: self-supervised pretraining on RealUW SSL.
S2/S3: downstream RUOD detection and UIIS10K mask training from the S1 backbone.
```

Task mapping:

| Task | S1 method | Backbone | S1 launcher config | S2 detection config | Mask config |
| --- | --- | --- | --- | --- | --- |
| J6 | SparK | ResNet-50 | `configs/exp_2/tri_pretrain/s1_j6_spark_resnet50_realuw.sh` | `configs/exp_2/cascade-rcnn_r50_realuw-pretrain_fpn_2x_ruod_j6.py` | `configs/exp_2/mask-rcnn_r50_realuw-pretrain_fpn_2x_uiis10k_j6_mask.py` |
| J7 | DINO | ResNet-50 | `configs/exp_2/tri_pretrain/s1_j7_dino_resnet50_realuw.sh` | `configs/exp_2/cascade-rcnn_r50_dino-realuw_fpn_2x_ruod_j7.py` | `configs/exp_2/mask-rcnn_r50_dino-realuw_fpn_2x_uiis10k_j7_mask.py` |
| J11 | MAE | ViT-Base | `configs/exp_2/tri_pretrain/s1_j11_mae_vit_base_realuw.sh` | `configs/exp_2/cascade-rcnn_vit-base_mae-realuw_fpn_2x_ruod_j11.py` | `configs/exp_2/mask-rcnn_vit-base_mae-realuw_fpn_2x_uiis10k_j11_mask.py` |
| J12 | SimMIM/MixMIM | SwinV2/Swin-Base | `configs/exp_2/tri_pretrain/s1_j12_simmim_swin_base_realuw.sh` | `configs/exp_2/cascade-rcnn_swinv2-base_mae-realuw_fpn_2x_ruod_j12.py` | `configs/exp_2/mask-rcnn_swinv2-base_mae-realuw_fpn_2x_uiis10k_j12_mask.py` |
| J13 | SparK | ConvNeXtV2-Tiny | `configs/exp_2/tri_pretrain/s1_j13_spark_convnextv2_tiny_realuw.sh` | `configs/exp_2/cascade-rcnn_convnext-tiny_mae-realuw_fpn_2x_ruod_j13.py` | `configs/exp_2/mask-rcnn_convnext-tiny_mae-realuw_fpn_2x_uiis10k_j13_mask.py` |

S1 work dirs:

```text
work_dirs/tri_pretrain/j6_realuw_spark_resnet50/
work_dirs/tri_pretrain/j7_realuw_dino_resnet50/
work_dirs/tri_pretrain/j11_realuw_mae_vit_base/
work_dirs/tri_pretrain/j12_realuw_simmim_swinv2_base/
work_dirs/tri_pretrain/j13_realuw_spark_convnextv2_tiny/
```

Converted backbone checkpoints expected by downstream scripts:

```text
../pretrained_weights/j6_realuw_spark_resnet50_backbone.pth
../pretrained_weights/j7_realuw_dino_resnet50_backbone.pth
../pretrained_weights/j11_realuw_mae_vit_base_backbone.pth
../pretrained_weights/j12_realuw_simmim_swinv2_base_backbone.pth
../pretrained_weights/j13_realuw_spark_convnextv2_tiny_backbone.pth
```

Main S1 logs:

```text
logs/tri_pretrain/j6_realuw_spark_resnet50_s1.log
logs/tri_pretrain/j7_realuw_dino_resnet50_s1.log
logs/tri_pretrain/j11_realuw_mae_vit_base_s1.log
logs/tri_pretrain/j12_realuw_simmim_swinv2_base_s1.log
logs/tri_pretrain/j13_realuw_spark_convnextv2_tiny_s1.log
```

The DINO S1 log fields mean:

```text
time: wall-clock seconds per iteration, window average.
data: dataloader wait/read/decode/augment seconds per iteration, window average.
Total time: exact wall-clock time for the epoch.
```

## Dataset Statistics

Main manually recorded dataset counts and filtered counts:

| Dataset / split | Source images | Filtered images | Filtered annotations | Notes |
| --- | ---: | ---: | ---: | --- |
| CoralSCOP train | 38,098 | 23,947 | 99,966 | Converted from per-image JSON segmentation. |
| CoralSCOP test | 830 | 357 | 2,222 | Converted from per-image JSON segmentation. |
| UVOT400 train | 191,136 valid frames | 11,942 | 11,942 | Tracking boxes; invalid 0-area frames skipped during conversion. |
| UVOT400 test | 80,987 valid frames | 3,361 | 3,361 | Tracking boxes; manual category resolution later created 12 categories. |
| DUO train | 6,671 | 15 | 154 | Native COCO; very few images pass the 20-percent threshold. |
| DUO test | 1,111 | 0 | 0 | Native COCO; no image passes threshold. |
| USIS16K train | 11,242 | 6,950 | 7,376 | Native COCO instance segmentation/detection style annotations. |
| USIS16K val | 1,539 | 945 | 995 | Native COCO. |
| USIS16K test | 3,370 | 2,108 | 2,233 | Native COCO. |
| UOT100 all | 74,004 | 17,334 | 17,334 | Tracking video frames converted to COCO. |
| UW-COT220 all | 158,949 | 16,318 | 16,318 | Tracking video frames converted to COCO. |
| MUOT_3M train/test | See stats JSON | 704,791 requested | 704,791 requested | Filtered frame extraction wrote 703,242 and skipped 1,549 existing frames in one run. |

Five-dataset large-object filtering summary after adding UOT100:

```text
TOTAL input images:     408,988
TOTAL filtered images:   66,959
TOTAL input anns:       738,088
TOTAL filtered anns:    145,583
keep ratio:              16.37%
```

UVOT400 resolved-category stats:

```text
total_images: 15,303
total_annotations: 15,303
category_count: 12
top categories:
  sea-turtle: 3,626
  fish: 3,011
  ray: 2,144
  diver: 1,837
  turtle: 1,500
  octopus: 714
  dolphin: 666
  jelly-fish: 547
  shark: 491
  sealion: 404
```

Useful statistics files:

```text
logs/exp2_dataset_stats_summary.json
logs/exp2_dataset_stats_summary_resolved.json
logs/exp2_category_name_index.csv
logs/exp2_category_name_index_resolved.csv
${EXP2_DATA_ROOT_HDD1}/UVOT400/annotations/uvot400_bbox20pct_category_stats.json
${EXP2_DATA_ROOT_HDD1}/UVOT400/annotations/uvot400_bbox20pct_category_stats.csv
```

Rebuild broad dataset statistics:

```bash
python tools/count_exp2_dataset_stats.py \
  2>&1 | tee logs/exp2_dataset_stats_summary.log
```

Count categories for one COCO dataset:

```bash
python tools/count_coco_category_stats.py \
  --dataset UVOT400 \
  --kind bbox20pct_resolved_categories \
  --ann \
    ${EXP2_DATA_ROOT_HDD1}/UVOT400/train/instances_train_bbox20pct_resolved_categories.json \
    ${EXP2_DATA_ROOT_HDD1}/UVOT400/test/instances_test_bbox20pct_resolved_categories.json \
  --out-json ${EXP2_DATA_ROOT_HDD1}/UVOT400/annotations/uvot400_bbox20pct_category_stats.json \
  --out-csv ${EXP2_DATA_ROOT_HDD1}/UVOT400/annotations/uvot400_bbox20pct_category_stats.csv
```

## Experiment Results Index

Use this section as the manually maintained index of stable reference results.
For each new finished experiment, record the log path, best epoch, best metric,
and checkpoint path before cleaning work directories.

| Experiment | Dataset / stage | Best epoch | Best metric | Checkpoint / note |
| --- | --- | ---: | ---: | --- |
| J2 repeat1 | RUOD det | - | 0.564 mAP | ImageNet-supervised R50 Cascade baseline. |
| J2 repeat2 | RUOD det | - | 0.562 mAP | ImageNet-supervised R50 Cascade baseline. |
| J2 repeat3 | RUOD det | - | 0.564 mAP | ImageNet-supervised R50 Cascade baseline. |
| J2 mean | RUOD det | - | 0.5633 mAP | Treat 0.001-0.003 mAP as normal run variation. |
| J10-HDP easy_ruod | RUOD S2 | 18 | 0.562 mAP | `work_dirs/j10_hdp_easy_ruod/s2/best_coco_bbox_mAP_epoch_18.pth` |
| J10-HDP dfui | RUOD S2 | 18 | 0.561 mAP | `work_dirs/j10_hdp_dfui/s2/best_coco_bbox_mAP_epoch_18.pth` |
| J10-HDP dfui_new | RUOD S2 | 17 | 0.563 mAP | `work_dirs/j10_hdp_dfui_new/s2/best_coco_bbox_mAP_epoch_17.pth` |

Best mAP extraction command:

```bash
for f in logs/*.log logs/*/*.log; do
  [ -f "$f" ] || continue
  best=$(grep -a "coco/bbox_mAP:" "$f" 2>/dev/null | \
    sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*coco\/bbox_mAP_50: ([0-9.]+).*coco\/bbox_mAP_75: ([0-9.]+).*coco\/bbox_mAP_s: ([0-9.]+).*coco\/bbox_mAP_m: ([0-9.]+).*coco\/bbox_mAP_l: ([0-9.]+).*/epoch_\1 mAP=\2 AP50=\3 AP75=\4 small=\5 medium=\6 large=\7/' | \
    sort -t= -k2 -nr | head -n 1)
  if [ -n "$best" ]; then
    printf "%-70s %s\n" "$f" "$best"
  fi
done
```

Checkpoint discovery:

```bash
find work_dirs -type f \( -name "best*.pth" -o -name "checkpoint*.pth" -o -name "latest.pth" \) \
  | sort
```

## Runtime Environment

Main conda environments observed on the server:

```text
${MMDET_ENV}
${MMPRETRAIN_ENV}
```

Typical activation for MMDetection downstream work:

```bash
conda activate ${MMDET_ENV}
```

Check core packages:

```bash
python - <<'PY'
mods = ["mmdet", "mmcv", "mmengine", "torch", "timm", "tensorboardX", "mmpretrain"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING/ERROR:", e)
PY
```

External code roots used by tri-pretrain:

```text
third_party/dino/       facebookresearch DINO route.
third_party/SparK/      keyu-tian SparK route.
../mmpretrain/          MMPreTrain repo if not vendored in third_party.
```

Important command-line tools:

```text
nvidia-smi    GPU memory/util monitoring.
rclone        Google Drive upload/download through remote syn:.
kaggle        ImageNet-1K Kaggle download.
ffmpeg/cv2    Video-frame extraction for tracking datasets.
rsync         RealUW imagefolder copy from HDD to SSD.
```

Disk notes:

```text
${EXP2_DATA_ROOT_HDD0}   Main exp_2 data root, often nearly full.
${EXP2_DATA_ROOT_HDD1}   Large external datasets and default RealUW SSL root.
${CONDA_ENV_ROOT}        Conda environments and some code dependencies.
${EXP2_DATA_ROOT_SSD1}   Preferred target for materialized RealUW imagefolder when IO matters.
```

When `${CONDA_ENV_ROOT}` is full, remove unused conda environments with:

```bash
du -sh ${CONDA_ENV_ROOT}/* | sort -h
conda env remove -p ${CONDA_ENV_ROOT}/ENV_NAME
```

Kaggle token path:

```text
~/.kaggle/kaggle.json
```

rclone remote:

```text
syn:
```

## Experiment Script Layout

```text
scripts/exp_2/base/
  Basic J2/J3/J4 launchers.

scripts/exp_2/j10/
  J10, J10-HDP, RFTM, scheme-C, and tuning launchers.

scripts/exp_2/uiis/
  UIIS10K easy-data pipeline and UIIS easy Faster R-CNN self-eval.

scripts/exp_2/usod/
  USOD10K conversion/easy/merge/J10 routes and visualization export.

scripts/exp_2/tri_pretrain/
  Real-underwater self-supervised pretraining and downstream pipelines.

scripts/exp_2/utils/
  GPU monitoring, watcher, dynamic guard, and occupier scripts.
```

Important script entrypoints:

```text
scripts/exp_2/uiis/run_exp_2_uiis_easy_j10_full.sh
  UIIS segmentation-to-detection conversion, A/B easy filtering, and
  DFUI+RUOD_easy+UIIS_easy merge pipeline.

scripts/exp_2/uiis/run_exp_2_uiis_easy_faster_self_eval.sh
  Faster R-CNN train=test self-eval on UIIS easy.

scripts/exp_2/usod/run_exp_2_usod_easy_merge.sh
  USOD objectness easy filtering and merge into expanded DFUI source.

scripts/exp_2/usod/run_exp_2_usod_easy_faster_self_eval.sh
  Faster R-CNN train=test self-eval on USOD easy.

scripts/exp_2/usod/export_usod_easy_visual_sample.sh
  Randomly sample USOD easy images, draw converted bboxes, zip, and upload with
  rclone.

scripts/exp_2/usod/run_exp_2_usod_easy_j10_scheme_c_full.sh
  USOD-expanded DFUI source + scheme-C J10 RCNN route.

scripts/exp_2/usod/run_exp_2_usod_mae_strategy.sh
  MAE ViT route. Uses MAE ViT S1 and ViT Cascade RUOD S2.

scripts/exp_2/usod/run_exp_2_usod_dino_strategy.sh
  DINO ResNet-50 route. Uses DINO R50 S1 and Cascade R50 RUOD S2.

scripts/exp_2/usod/run_exp_2_usod_rcnn_auto_tuning.sh
  Multi-round RCNN tuning. Three jobs per round.

scripts/exp_2/j10/run_exp_2_j10_scheme_c.sh
  Single scheme-C J10 run.

scripts/exp_2/j10/run_exp_2_j10_scheme_c_frozen_lr00375_parallel.sh
  Frozen-stage sweep at lr=0.00375.

scripts/exp_2/j10/run_exp_2_j10_scheme_c_f1_lr_epoch_sweep_parallel.sh
  LR/epoch sweep for scheme-C f1.

scripts/exp_2/utils/run_exp_2_gpu_occupier.sh
  GPU memory and compute occupier.
```

## Key Configs

RUOD baselines and downstream:

```text
configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py
  ImageNet-supervised ResNet-50 + Cascade R-CNN RUOD baseline.
  Uses ${EXP2_DATA_ROOT_HDD0}/RUOD/coco/.

configs/exp_2/cascade-rcnn_r50_dino_fpn_2x_ruod_j4.py
  DINO ResNet-50 RUOD baseline.

configs/exp_2/cascade-rcnn_vit-base_mae_fpn_2x_ruod_j3.py
  MAE ViT-Base RUOD baseline.

configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j10_v2_s2.py
  J10 S2 RUOD config used by many J10 routes.
```

J10 supervised source adaptation:

```text
configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_j10_s1.py
configs/exp_2/cascade-rcnn_r50_fpn_2x_merged_j10_s1.py
configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_s1.py
configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_easy_j10_scheme_c_s1.py
configs/exp_2/cascade-rcnn_r50_fpn_2x_dfui_ruod_uiis_usod_easy_j10_scheme_c_s1.py
```

UIIS/USOD easy filtering:

```text
configs/exp_2/cascade-rcnn_r50_fpn_2x_uiis10k_det.py
configs/exp_2/cascade-rcnn_r50_fpn_2x_usod10k_det.py
```

RFTM/HDP:

```text
configs/exp_2/cascade-rcnn_r50-rftm_fpn_2x_dfui_new_s1.py
configs/exp_2/cascade-rcnn_r50-rftm_fpn_2x_ruod_s2.py
configs/exp_2/cascade-rcnn_r50-rftm-hdp_fpn_2x_ruod_j10_s2.py
```

Tri-pretrain downstream:

```text
configs/exp_2/cascade-rcnn_r50_realuw-pretrain_fpn_2x_ruod_j6.py
configs/exp_2/cascade-rcnn_r50_dino-realuw_fpn_2x_ruod_j7.py
configs/exp_2/cascade-rcnn_vit-base_mae-realuw_fpn_2x_ruod_j11.py
configs/exp_2/cascade-rcnn_swinv2-base_mae-realuw_fpn_2x_ruod_j12.py
configs/exp_2/cascade-rcnn_convnext-tiny_mae-realuw_fpn_2x_ruod_j13.py
```

## Server Data Roots

Main exp_2 data root:

```text
${EXP2_DATA_ROOT_HDD0}/
```

Additional data root for large external datasets:

```text
${EXP2_DATA_ROOT_HDD1}/
```

Common environment and code root:

```text
${MMDET_ENV}/
${REPO_ROOT}/
```

## Core Datasets

RUOD:

```text
${EXP2_DATA_ROOT_HDD0}/RUOD/
${EXP2_DATA_ROOT_HDD0}/RUOD/coco/annotations/instances_train.json
${EXP2_DATA_ROOT_HDD0}/RUOD/coco/annotations/instances_val.json
${EXP2_DATA_ROOT_HDD0}/RUOD/coco/annotations/easy_merged.json
${EXP2_DATA_ROOT_HDD0}/RUOD/coco/train/
${EXP2_DATA_ROOT_HDD0}/RUOD/coco/val/
```

DFUI original:

```text
${EXP2_DATA_ROOT_HDD0}/dfui/
${EXP2_DATA_ROOT_HDD0}/dfui/annotations/instances_train2017.json
${EXP2_DATA_ROOT_HDD0}/dfui/annotations/instances_val2017.json
${EXP2_DATA_ROOT_HDD0}/dfui/annotations/instances_test2017.json
```

DFUI_NEW:

```text
${EXP2_DATA_ROOT_HDD0}/DFUI_NEW/
```

UIIS10K:

```text
${EXP2_DATA_ROOT_HDD0}/UIIS10K/
${EXP2_DATA_ROOT_HDD0}/UIIS10K/img/
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/instances_train.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/instances_train_det.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/cross_split_det/train_A.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/cross_split_det/train_B.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/cross_split_det/A_easy.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/cross_split_det/B_easy.json
${EXP2_DATA_ROOT_HDD0}/UIIS10K/coco/annotations/cross_split_det/easy_merged.json
```

USOD10K:

```text
${EXP2_DATA_ROOT_HDD0}/USOD10K/
${EXP2_DATA_ROOT_HDD0}/USOD10K/images/
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/instances_trainval.json
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/cross_split_det/train_A.json
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/cross_split_det/train_B.json
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/cross_split_det/A_easy.json
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/cross_split_det/B_easy.json
${EXP2_DATA_ROOT_HDD0}/USOD10K/annotations/cross_split_det/easy_merged.json
```

Merged detection sources:

```text
${EXP2_DATA_ROOT_HDD0}/DFUI_RUOD_UIIS_EASY/
${EXP2_DATA_ROOT_HDD0}/DFUI_RUOD_UIIS_USOD_EASY/
```

Large external datasets on HDD1:

```text
${EXP2_DATA_ROOT_HDD1}/CoralSCOP/
${EXP2_DATA_ROOT_HDD1}/DUO/
${EXP2_DATA_ROOT_HDD1}/MARIS/
${EXP2_DATA_ROOT_HDD1}/UOT100/
${EXP2_DATA_ROOT_HDD1}/USIS16K/
${EXP2_DATA_ROOT_HDD1}/UVOT400/
${EXP2_DATA_ROOT_HDD1}/UW-COT220/
${EXP2_DATA_ROOT_HDD1}/MUOT_3M/
```

## Generated Large-Object Filter Files

The 20-percent large-object filtering outputs normally use this suffix:

```text
*_bbox20pct.json
```

Examples:

```text
${EXP2_DATA_ROOT_HDD1}/CoralSCOP/annotations/instances_train_bbox20pct.json
${EXP2_DATA_ROOT_HDD1}/UVOT400/train/instances_train_bbox20pct.json
${EXP2_DATA_ROOT_HDD1}/DUO/annotations/instances_train_bbox20pct.json
${EXP2_DATA_ROOT_HDD1}/USIS16K/USIS16K/annotations/instances_train_bbox20pct.json
${EXP2_DATA_ROOT_HDD1}/UOT100/annotations/instances_all_bbox20pct.json
```

## Important Tools

Dataset inspection and conversion:

```text
tools/inspect_annotation_format.py
tools/convert_uiis10k_seg_to_det.py
tools/convert_usod10k_mask_to_det.py
tools/convert_coralscop.py
tools/convert_uvot400_v2.py
tools/convert_uot100_to_coco.py
tools/convert_webuot_coco.py
```

Filtering and merging:

```text
tools/filter_coco_large_objects.py
tools/uiis10k_cross_easy.py
tools/usod10k_cross_easy.py
tools/merge_dfui_ruod_uiis_easy.py
tools/merge_dfui_ruod_uiis_usod_easy.py
```

Visualization and checking:

```text
tools/visualize_coco_bbox_samples.py
tools/count_coco_category_stats.py
tools/sample_coco.py
```

Backbone conversion:

```text
tools/convert_ssl_backbone_to_mmdet.py
```

## Logs and Checkpoints

Default log root:

```text
${REPO_ROOT}/logs/
```

Default checkpoint root:

```text
${REPO_ROOT}/work_dirs/
```

Common log folders:

```text
logs/j10_scheme_c_tuning/
logs/j10_usod_auto_tuning/
logs/j10_usod_dual_strategy/
logs/j10_dino_usod/
logs/j1_repeat_parallel_j2config/
```

Common work dirs:

```text
work_dirs/j10_scheme_c_tuning/
work_dirs/j10_usod_auto_tuning/
work_dirs/j10_scheme_c_usod/
work_dirs/j10_dino_usod/
work_dirs/uiis_easy_faster_rcnn_self_eval_e96/
work_dirs/usod_easy_faster_rcnn_self_eval/
work_dirs/usod_easy_faster_rcnn_self_eval_e96/
```

Most scripts set:

```text
default_hooks.checkpoint.save_best=coco/bbox_mAP
default_hooks.checkpoint.max_keep_ckpts=5
```

This keeps the best mAP checkpoint and the latest few regular checkpoints.

## Pretrained Weights

Common pretrained weight root:

```text
${PRETRAINED_WEIGHTS_ROOT}/
```

Expected examples:

```text
${PRETRAINED_WEIGHTS_ROOT}/mae_pretrain_vit_base.pth
${PRETRAINED_WEIGHTS_ROOT}/dino_resnet50_pretrain.pth
```

MMDetection configs may also use torchvision/ImageNet initialization through
the base model config.

## rclone

Known rclone remote on the server:

```text
syn:
```

Check remotes:

```bash
rclone listremotes
```

Upload a zip:

```bash
rclone copy -P path/to/file.zip syn:exp_2/some_folder/
```

List uploaded files:

```bash
rclone lsf syn:exp_2/some_folder/
```

USOD easy visual samples default upload destination:

```text
syn:exp_2/usod_easy_visual_samples/
```

## Common Result Extraction Commands

Best COCO bbox mAP from one log:

```bash
grep -a "coco/bbox_mAP:" LOG_FILE.log | \
sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*coco\/bbox_mAP_50: ([0-9.]+).*coco\/bbox_mAP_75: ([0-9.]+).*coco\/bbox_mAP_s: ([0-9.]+).*coco\/bbox_mAP_m: ([0-9.]+).*coco\/bbox_mAP_l: ([0-9.]+).*/epoch_\1 mAP=\2 AP50=\3 AP75=\4 small=\5 medium=\6 large=\7/' | \
sort -t= -k2 -nr | head -n 10
```

Summarize repeat logs:

```bash
for f in logs/j1_repeat_parallel_j2config/*.log; do
  echo "========================================"
  echo "$(basename "$f")"
  grep -a "coco/bbox_mAP:" "$f" | \
  sed -E 's/.*Epoch\(val\) \[([0-9]+)\].*coco\/bbox_mAP: ([0-9.]+).*coco\/bbox_mAP_50: ([0-9.]+).*coco\/bbox_mAP_75: ([0-9.]+).*coco\/bbox_mAP_s: ([0-9.]+).*coco\/bbox_mAP_m: ([0-9.]+).*coco\/bbox_mAP_l: ([0-9.]+).*/epoch_\1 mAP=\2 AP50=\3 AP75=\4 small=\5 medium=\6 large=\7/' | \
  sort -t= -k2 -nr | head -n 5
done
```

## Common Dataset Count Checks

Count COCO images and annotations:

```bash
python - <<'PY'
import json
p = "PATH/TO/instances.json"
d = json.load(open(p))
print("images:", len(d.get("images", [])))
print("annotations:", len(d.get("annotations", [])))
print("categories:", [c.get("name") for c in d.get("categories", [])])
PY
```

Check missing image references:

```bash
python - <<'PY'
import json, os
ann = "PATH/TO/instances.json"
root = "PATH/TO/images"
d = json.load(open(ann))
missing = []
for im in d.get("images", []):
    p = os.path.join(root, im["file_name"])
    if not os.path.exists(p):
        missing.append(im["file_name"])
        if len(missing) >= 10:
            break
print("images:", len(d.get("images", [])))
print("annotations:", len(d.get("annotations", [])))
print("missing sample:", missing)
PY
```

## Notes on Dataset Roles

RUOD, DFUI, and UIIS easy are detection-supervised sources:

```text
RUOD: detection dataset.
DFUI: detection dataset.
UIIS10K: instance segmentation dataset converted to bbox detection; still has
         semantic instance categories.
```

USOD10K is different:

```text
USOD10K: saliency/objectness dataset converted to single-class object bboxes.
```

USOD is useful for checking water-domain image features, but its converted bbox
labels are less reliable as detection supervision. When using USOD, keep
Faster R-CNN self-eval and visualized bbox samples as quality checks before
merging it into detection-supervised J10 experiments.

## Current Baseline Reference

Recent repeated ImageNet/Cascade R50 RUOD baseline using
`configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py`:

```text
repeat1 best mAP: 0.564
repeat2 best mAP: 0.562
repeat3 best mAP: 0.564
mean: 0.5633
```

Small differences around 0.001 to 0.003 mAP should be treated as normal random
training variation unless repeated runs show a stable trend.
