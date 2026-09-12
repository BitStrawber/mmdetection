# Nested DINO scale experiments

This workflow evaluates three DINO source pools (`imagenet`, `realuw`, and
`synthetic5`) at 100K, 300K, 500K, 800K, and 1M images.  It is deliberately
split across hosts:

1. **fuping:** create compact nested path indexes and run 100-epoch Facebook
   DINO pretraining for ResNet-50 and ViT-S directly from the original SSD1
   image pools.
2. **Manual transfer:** copy the two raw `checkpoint.pth` files for one source
   and scale to the `fcp` transfer root.
3. **fcp:** convert teacher backbones, run direct RUOD Cascade R-CNN and
   UIIS10K Mask R-CNN, then run the two DFUI Cascade R-CNN branches.  Each DFUI
   branch can additionally run RUOD and UIIS10K downstream transfer.

## Dataset construction on fuping

The builder requires the original training pool and the already-used immutable
100K ImageFolder root. It uses the 100K relative paths exactly, then appends a
deterministic shuffled ordering of unused source images. It stores only two
text indexes per source: `base_100k.txt` and `remaining_permutation.txt`.
Each scale manifest records a prefix length and SHA256 selection digest. No
image bytes, symlinks, hardlinks, or duplicate ImageFolder trees are created.

```bash
python tools/exp_2/build_nested_dino_subsets.py \
  --source imagenet=/PATH/TO/IMAGENET/imagefolder/train \
  --source realuw=/PATH/TO/REALUW_SSL/imagefolder/train \
  --source synthetic5=/media/SSD1/XCX/exp_2/synthetic_imagenet/merged_5methods/imagefolder/train \
  --base-100k imagenet=/media/SSD1/XCX/exp_2/dino_100k_control/imagenet100k/imagefolder/train \
  --base-100k realuw=/media/SSD1/XCX/exp_2/dino_100k_control/realuw100k/imagefolder/train \
  --base-100k synthetic5=/media/SSD1/XCX/exp_2/dino_100k_control/synthetic5_100k/imagefolder/train \
  --out-root /media/SSD1/XCX/exp_2/dino_nested_control \
  --seed 20260831
```

Before committing metadata or compute, append `--dry-run`. The source and its
existing 100K root must have matching ImageFolder-relative file paths. This
intentionally fails rather than silently creating a different 100K base.

The DINO runner invokes `tools/exp_2/run_dino_with_index.py` when an index
manifest is supplied. That adapter replaces only `torchvision.datasets.ImageFolder`
for the current process; Facebook DINO's model, multi-crop augmentation, loss,
optimizer, EMA teacher, and schedule remain unchanged. The original image files
are read directly from the SSD1 source root.

## Pretraining on fuping

`run_nested_dino_scale_pretraining.sh` iterates in this exact order:

```text
100K: ImageNet R50 -> ViT-S -> RealUW R50 -> ViT-S -> Synthetic5 R50 -> ViT-S
300K: same six jobs
500K: same six jobs
800K: same six jobs
1M:   same six jobs
```

Every job uses the current Facebook DINO runner, 100 epochs, the fixed DINO
recipes, and the complete `GPU_IDS` group.  Therefore no larger-scale job
starts before all smaller-scale jobs finish.

The nested launcher waits for each GPU to have no more than `9000 MiB` used
memory and no more than 5% utilization for one check. This default
allows an 8 GiB low-utilization memory guard while preserving approximately 3
GiB of headroom for the measured 13 GiB DINO workload on 24 GiB RTX 3090 GPUs.
Override `GPU_MAX_MEM_MB`, `GPU_MAX_UTIL`, `GPU_IDLE_CHECKS`, or
`GPU_WAIT_INTERVAL` in the launch environment when running without that guard.

## Manual transfer convention

For `SOURCE=realuw`, `SCALE=300k`, place raw checkpoints on fcp as:

```text
/media/SSD1/XCX/exp_2/manual_dino_nested_transfer/
  scale300k_realuw_dino_resnet50_100e/checkpoint.pth
  scale300k_realuw_dino_vits_100e/checkpoint.pth
```

The downstream script validates `epoch=100`, architecture, and teacher state
before conversion.  It never assumes that a checkpoint filename alone proves
its provenance.

## Downstream coverage on fcp

For one source/scale pair, the downstream script runs on two independent
two-GPU groups:

```text
R50, GPUs 4,5: direct RUOD Cascade + direct UIIS Mask
               DFUI+RUOD Cascade -> RUOD Cascade + UIIS Mask
               DFUI+RUOD+UIIS Cascade -> RUOD Cascade + UIIS Mask

ViT-S, GPUs 6,7: same sequence
```

Set `DFUI_FOLLOWUPS=ruod` when only the historical DFUI-to-RUOD protocol is
needed.  The default `ruod,mask` produces the complete detection and instance
segmentation transfer set for both DFUI branches.

## Full-source DFUI-to-RUOD detection transfer on fcp

`run_full_realuw_then_synthetic5_dfui_ruod_det.sh` runs the historical
DFUI+RUOD+UIIS detector stage for 48 epochs and then the standard RUOD detector
stage for 24 epochs. It uses the transferred full-source DINO checkpoints by
default. The chains are independent: GPUs `4,5` process RealUW R50 followed by
corrected Synthetic5 R50, while GPUs `6,7` process RealUW ViT-S followed by
corrected Synthetic5 ViT-S. This lets one architecture begin its Synthetic5
run without waiting for the other architecture's RealUW run to complete.

Run a no-training validation first:

```bash
CHECK_ONLY=1 bash scripts/exp_2/tri_pretrain/run_full_realuw_then_synthetic5_dfui_ruod_det.sh
```

Then start the two chains:

```bash
nohup bash scripts/exp_2/tri_pretrain/run_full_realuw_then_synthetic5_dfui_ruod_det.sh \
  > /media/SSD1/XCX/exp_2/dfui_training_runs/full_transfer_launcher.log 2>&1 &
```
