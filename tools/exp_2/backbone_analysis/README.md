# Backbone Analysis Data Pipeline

This directory provides reusable, parameterized stages for comparing an
ImageNet backbone with the backbone extracted from its RUOD Cascade R-CNN.
The stages do not hard-code experiment paths and may be run independently.

## Terminology

`render_feature_activation.py` computes `mean(abs(feature), channel)`. Its
outputs are feature activation maps, not Grad-CAM. Detection Grad-CAM requires
a detection score and gradients and is intentionally outside this paired
backbone comparison.

## Requirements

Run commands from the MMDetection repository root in the existing MMDetection
environment. The feature extractor requires MMEngine, MMDetection, PyTorch,
NumPy and Pillow. CKA and frequency summaries require NumPy. t-SNE additionally
requires scikit-learn. PNG rendering optionally uses matplotlib for the CKA
heatmap.

## 1. Deterministic RUOD sample

```bash
python -m tools.exp_2.backbone_analysis.sample_ruod \
  --annotation-file /path/to/instances_val.json \
  --image-root /path/to/RUOD/val \
  --out-dir /path/to/analysis/sample \
  --samples 50 \
  --seed 2026 \
  --materialize copy
```

Outputs include `manifest.jsonl`, a COCO annotation subset, copied/symlinked
images and sampling metadata. Set `--materialize none` to reference source
images without copying them.

## 2. Frequency-controlled images

```bash
python -m tools.exp_2.backbone_analysis.generate_frequency_bands \
  --manifest /path/to/analysis/sample/manifest.jsonl \
  --out-dir /path/to/analysis/frequency_inputs \
  --bands 'low:0.0:0.15,mid:0.15:0.40,high:0.40:1.0' \
  --reconstruction mean-preserve
```

The output manifest contains clean/low/mid/high paths and exact filter
parameters. `mean-preserve` adds the original channel DC level to non-low
bands, making band-isolated inputs usable by standard image pipelines. Use
`--save-float` when exact pre-quantization arrays are required.

## 3. Model configuration

Copy and edit `models.example.json`, or let `run_pipeline.sh` generate a runtime
configuration. A `backbone` entry builds a detector shell solely to reuse the
same MMDetection preprocessing and loads the checkpoint into `model.backbone`.
A `detector` entry loads a complete checkpoint but exports only configured
backbone hooks.

Raw DINO checkpoints commonly need:

```json
"state_dict_key": "teacher",
"checkpoint_prefix": "module.backbone."
```

Converted MMDetection backbone checkpoints usually need neither. The loader
also tries common module/backbone/student/teacher prefixes automatically and
writes `model_load_reports.json`. Always inspect the matched tensor ratio.

## 4. Extract clean and frequency features

```bash
python -m tools.exp_2.backbone_analysis.extract_backbone_features \
  --manifest /path/to/analysis/frequency_inputs/frequency_manifest.jsonl \
  --models-config /path/to/models.json \
  --out-dir /path/to/analysis/feature_store \
  --variants clean,low,mid,high \
  --layers res2,res3,res4,res5 \
  --device cuda:0 \
  --save-spatial \
  --spatial-samples 50
```

Pooled features are stored as
`features/MODEL/VARIANT/LAYER.npy` with shape `N x D`. Full valid-region norms
are stored as `LAYER.spatial_norm.npy`. Optional spatial tensors are compressed
under `spatial/MODEL/VARIANT/SAMPLE/LAYER.npz`.

Both models use the detector config's test pipeline. This prevents ImageNet
center-crop geometry from being compared against detector keep-ratio geometry.

## 5. CKA

```bash
python -m tools.exp_2.backbone_analysis.compute_cka \
  --feature-root /path/to/analysis/feature_store \
  --model-a imagenet_backbone \
  --model-b cascade_ruod_backbone \
  --variant clean \
  --layers-a res2,res3,res4,res5 \
  --layers-b res2,res3,res4,res5 \
  --out-dir /path/to/analysis/results/cka
```

The command writes the numeric matrix as NPY/TSV, metadata and an optional PNG.

## 6. Frequency response

```bash
python -m tools.exp_2.backbone_analysis.compute_frequency_response \
  --feature-root /path/to/analysis/feature_store \
  --models imagenet_backbone,cascade_ruod_backbone \
  --layers res2,res3,res4,res5 \
  --frequency-variants low,mid,high \
  --out-dir /path/to/analysis/results/frequency_response
```

Outputs contain per-sample and aggregate pooled norm ratio, normalized feature
shift and full-spatial Frobenius norm ratio.

## 7. Blue-yellow activation maps and FG/BG data

Feature extraction must have used `--save-spatial`.

```bash
python -m tools.exp_2.backbone_analysis.render_feature_activation \
  --feature-root /path/to/analysis/feature_store \
  --manifest /path/to/analysis/frequency_inputs/frequency_manifest.jsonl \
  --models imagenet_backbone,cascade_ruod_backbone \
  --layers res2,res3,res4,res5 \
  --out-dir /path/to/analysis/results/activation
```

The script saves raw float activation arrays, shared-normalization metadata,
blue-yellow PNGs with/without GT boxes, comparison panels and FG/BG statistics.

## 8. Optional joint t-SNE

```bash
python -m tools.exp_2.backbone_analysis.compute_tsne \
  --feature-root /path/to/analysis/feature_store \
  --manifest /path/to/analysis/sample/manifest.jsonl \
  --models imagenet_backbone,cascade_ruod_backbone \
  --layer res5 \
  --out-dir /path/to/analysis/results/tsne_res5
```

This performs one joint PCA/t-SNE fit and writes coordinates. It does not fit
the models separately because independently fitted t-SNE coordinates are not
comparable.

## 9. Optional orchestration

`run_pipeline.sh` calls the independent stages. All experiment values are
environment variables:

```bash
RUOD_ANN=/path/to/instances_val.json \
RUOD_IMAGE_ROOT=/path/to/val \
BACKBONE_CONFIG=configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_ruod.py \
BACKBONE_CHECKPOINT=/path/to/imagenet/checkpoint.pth \
BACKBONE_STATE_KEY=teacher \
BACKBONE_PREFIX=module.backbone. \
CASCADE_CONFIG=configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_ruod.py \
CASCADE_CHECKPOINT=/path/to/ruod/best.pth \
OUT_ROOT=/path/to/analysis/run01 \
SAMPLES=50 DEVICE=cuda:0 \
bash tools/exp_2/backbone_analysis/run_pipeline.sh
```

Set any `RUN_*` flag to zero to skip a stage. For example, recompute only CKA:

```bash
RUN_SAMPLE=0 RUN_FREQUENCY_IMAGES=0 RUN_FEATURES=0 \
RUN_FREQUENCY_RESPONSE=0 RUN_ACTIVATION=0 RUN_CKA=1 \
OVERWRITE=1 ... bash tools/exp_2/backbone_analysis/run_pipeline.sh
```

The orchestrator defaults to `RUN_TSNE=0` because t-SNE is optional in the
current experiment.
