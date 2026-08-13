# RUOD CAM and Backbone Activation Suite

This directory now exposes three deliberately separate analysis branches. They
share one deterministic RUOD sample but answer different questions:

| Switch | Target | Output meaning | Recommended role |
|---|---|---|---|
| `RUN_FIXED_GT_CAM` | fixed GT ROI and GT class logit | evidence used for the same controlled object decision | primary cross-model CAM experiment |
| `RUN_PREDICTION_CAM` | each detector's post-NMS box and predicted class | evidence behind actual detector behavior, including false positives | qualitative/error-analysis supplement |
| `RUN_PRETRAINED_BACKBONE_ACTIVATION` | no detection target or gradient | channel-mean absolute `res2-res5` feature activation | upstream representation evidence |

The first two branches compute XGradCAM. The pretrained-backbone branch is a
feature-activation visualization and must not be called CAM in the paper.

## Switch-based 10-image validation

Run all three branches on exactly the same ten RUOD images:

```bash
cd ~/xcx/exp_2/mmdetection
conda activate /media/SSD1/conda_envs/mmdet
unset LD_LIBRARY_PATH LD_PRELOAD

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=cam_suite_ruod10_${STAMP}
OUT_ROOT=/media/HDD2/XCX/exp_2/further_features/${RUN_NAME}
mkdir -p "$OUT_ROOT"

nohup env \
  RUN_NAME="$RUN_NAME" \
  OUT_ROOT="$OUT_ROOT" \
  SAMPLES=10 \
  SEED=2026 \
  RUN_FIXED_GT_CAM=1 \
  RUN_PREDICTION_CAM=1 \
  RUN_PRETRAINED_BACKBONE_ACTIVATION=1 \
  DETECTOR_DEVICES=cuda:4,cuda:5,cuda:6 \
  DETECTOR_PARALLEL_MODELS=3 \
  PRETRAINED_DEVICE=cuda:6 \
  PREDICTION_SCORE_THRESHOLD=0.05 \
  MAX_PREDICTIONS_PER_IMAGE=10 \
  PREDICTION_MATCH_IOU=0.50 \
  RESUME=1 \
  bash scripts/exp_2/features/further_features/run_cam_analysis_suite.sh \
  > "$OUT_ROOT/launcher.log" 2>&1 &

echo $! | tee "$OUT_ROOT/launcher.pid"
echo "OUT_ROOT=$OUT_ROOT"
```

To run only selected branches, set the other branch switches to `0`. Examples:

```bash
# Prediction-conditioned CAM only, reusing an existing sample manifest.
RUN_FIXED_GT_CAM=0 \
RUN_PREDICTION_CAM=1 \
RUN_PRETRAINED_BACKBONE_ACTIVATION=0 \
RUN_SAMPLE=0 \
SAMPLE_ROOT=/path/to/existing/run/sample \
OUT_ROOT=/path/to/new/run \
bash scripts/exp_2/features/further_features/run_cam_analysis_suite.sh

# Bare pretrained backbones only.
RUN_FIXED_GT_CAM=0 \
RUN_PREDICTION_CAM=0 \
RUN_PRETRAINED_BACKBONE_ACTIVATION=1 \
OUT_ROOT=/path/to/new/run \
bash scripts/exp_2/features/further_features/run_cam_analysis_suite.sh
```

Monitor all branches:

```bash
OUT_ROOT=/media/HDD2/XCX/exp_2/further_features/<run-name> \
INTERVAL=15 \
bash scripts/exp_2/features/further_features/monitor_cam_analysis_suite.sh
```

`Ctrl+C` stops only the monitor.

## Prediction-conditioned branch

For every detector and sampled image, normal inference first supplies
post-NMS boxes, labels and scores. Predictions are sorted by score, filtered by
`PREDICTION_SCORE_THRESHOLD`, and truncated by
`MAX_PREDICTIONS_PER_IMAGE`. Each selected predicted ROI is then evaluated at
the configured Cascade stage, and XGradCAM targets its predicted-class logit.

The branch saves, per prediction:

- raw, unnormalized CAM for every selected layer;
- post-NMS prediction box, class and score;
- recomputed ROI-head target logit and probability;
- best-IoU GT annotation, category, IoU and class-correct flag;
- a TP-like flag using `PREDICTION_MATCH_IOU`;
- spatial CAM metrics relative to the predicted box and, when matched, the GT
  box;
- individual heatmaps/overlays and a pixelwise-max prediction-union panel.

Prediction-union panels show where the detector's selected decisions focus.
They do not make prediction instances identical across models. Consequently,
the prediction branch should explain behavior or failure cases, while fixed-GT
CAM remains the controlled cross-model comparison.

## Bare pretrained-backbone branch

`models.pretrained_backbone_activation.json` loads the ImageNet DINO 100e,
RealUW DINO 100e, Synthetic5 DINO 100e and ImageNet-DINO-to-DFUI adapted
backbones without downstream RUOD detector weights.
It hooks `res2-res5`, saves aligned spatial features, and aggregates each CHW
feature as `mean(abs(feature), channel)` for display.

Two ImageNet-reference visual scales are produced:

1. `imagenet_reference_dataset_p1_p99`: one ImageNet-derived P1/P99 range per
   layer over the complete sampled dataset;
2. `imagenet_reference_per_sample_p1_p99`: one ImageNet-derived P1/P99 range
   for each image and layer.

These products show generic representation response. Because there is no RUOD
ROI class logit, they are not class-conditioned and are not XGradCAM.

## Branch output layout

```text
RUN_ROOT/
  sample/                              # one manifest shared by every branch
  fixed_gt/
    raw/raw_cam/MODEL/image_*/ann_*/
    rendered/
  prediction/
    raw/raw_cam/MODEL/image_*/pred_*/
    rendered/
      independent_p1_p99/
      imagenet_reference_dataset_p1_p99/
      panels/
      metrics/prediction_layer_metrics.tsv
  pretrained_backbone/
    feature_store/spatial/MODEL/clean/
    rendered/
      imagenet_reference_dataset_p1_p99/
      imagenet_reference_per_sample_p1_p99/
  logs/
  COMPLETE.env
```

The older fixed-GT-only commands below remain supported.

This directory implements task-conditioned, fixed-ground-truth XGradCAM for
comparing RUOD-trained Cascade R-CNN detectors initialized from different
pretraining datasets.

It is deliberately separate from the existing backbone activation pipeline.
The existing activation renderer computes a channel aggregation such as
`mean(abs(feature), channel)` and does not use a detection score or gradients.
This module instead uses the unnormalized RUOD class logit of one fixed GT ROI
as its gradient target.

## Research question

For the same RUOD image, annotation, GT box, GT class, Cascade stage and target
layer, the only intended variable is the detector checkpoint. This answers:

> Given the correct target ROI, which image regions support the detector's
> RUOD-category decision, and how does that spatial evidence differ between
> pretraining strategies?

A bare DINO/ImageNet backbone cannot provide this target because it has no RUOD
ROI classifier. Use bare pretrained checkpoints for CKA, feature activation,
t-SNE and frequency response. Use the downstream RUOD detector checkpoints for
fixed-GT category-conditioned XGradCAM.

## Core definition

For target layer activations `A`, gradients `G = d logit(gt_class) / dA`, and
channel `k`, the extractor computes the XGradCAM weight

```text
alpha_k = sum_xy(G_kxy * A_kxy) / (sum_xy(A_kxy) + eps)
raw_cam = ReLU(sum_k(alpha_k * A_k))
```

The extractor saves this nonnegative `raw_cam` before any min-max, percentile,
gamma or color operation. Spatial metrics are computed from this raw array.

## Two rendering strategies

The renderer always produces two complementary views for every
`(image_id, annotation_id, layer)` unit:

1. `independent_p1_p99`: each model/instance/layer uses its own P1 and P99.
   This is the clearest view of response location and shape, but it cannot show
   absolute response-strength differences.
2. `imagenet_reference_p1_p99`: all models use the P1/P99 values from the
   ImageNet-pretrained RUOD Cascade detector for the same instance and layer.
   This preserves relative intensity under one displayed reference scale.

The exact low/high values are saved in `normalization_scales.tsv`. Cross-model
raw CAM magnitude is affected by classifier, BN and logit scale, so shared-scale
color intensity is supporting evidence only. Primary quantitative evidence
should use scale-invariant spatial metrics.

An additional compact old-style view is written to `image_aggregate/`. It
combines the separate fixed-GT CAMs belonging to the same image and writes
exactly one PNG for every image, model, layer and normalization strategy. The
default aggregation is pixelwise `max`; `sum` and `mean` are also supported via
`FIXED_GT_IMAGE_AGGREGATION`. The one saved view defaults to `pure` and can be
changed to `overlay` or `with_gt` with `FIXED_GT_IMAGE_AGGREGATE_VIEW`. This
compact output is useful for comparison with older top-k prediction aggregation
scripts, while per-instance fixed-GT CAM remains the primary controlled result.
It also writes final cross-model panels to
`image_aggregate/panels/STRATEGY/LAYER/image_*.png`. Each panel fixes the
strategy, source image and layer, then places the input image followed by the
four detector-backbone results in columns. Incomplete model sets are skipped
and counted in `render_summary.json`.
For reading depth evolution within one detector, it additionally writes
`image_aggregate/panels_by_model/STRATEGY/MODEL/image_*.png`. These panels fix
the strategy, image and model and use the columns `Input | res2 | res3 | res4 |
res5`.

For a direct reproduction of the three legacy prediction-CAM scripts, the
prediction branch also writes `legacy_image_aggregate/`. It aggregates the
top-scoring prediction CAMs for each image and layer (`sum` by default), applies
independent post-aggregation min-max normalization, and saves one pure heatmap
per image, model, layer and style. `legacy_jet` matches the older OpenCV JET
renderer; `legacy_turbo_gamma05` matches the newer Turbo renderer with gamma
0.5. These high-contrast images are for qualitative display only and must not
be used to compare absolute response magnitude across models.
Final legacy comparison panels are written to
`legacy_image_aggregate/panels/STYLE/LAYER/image_*.png`, with the input image
in the first column and the four detector results in the remaining columns.
The style, image and layer are therefore held constant inside each panel.
The companion layer-evolution panel is
`legacy_image_aggregate/panels_by_model/STYLE/MODEL/image_*.png`, which fixes
the style, image and model and uses `Input | res2 | res3 | res4 | res5`.

The bare pretrained-backbone branch already uses the same comparison contract:
each normalization output contains `panels/LAYER/*.png`, where one panel fixes
the normalization strategy, image and layer and compares the input against all
four pretrained backbones. No new feature extraction or activation calculation
is required to obtain these panels. It also emits
`panels_by_model/MODEL/*.png`, which fixes the normalization strategy, image
and backbone and compares that backbone's `res2-res5` activations beside the
input.

## Quantitative outputs

The renderer computes, per model/instance/layer:

- fraction of CAM energy inside the target GT box;
- fraction of CAM energy inside any GT box;
- target/background mean-response ratio, where all valid GT boxes are excluded
  from the background;
- pointing-game hit;
- IoU between the target box and the top 20% CAM pixels;
- normalized spatial entropy;
- CAM peak distance from the GT center, divided by the GT-box diagonal;
- target class logit and probability.

Summaries are grouped by model/layer, category and COCO pixel-size bucket. PNG
and PDF bar charts and heatmaps are produced from the metric table.

## Model configuration

Edit `models.fixed_gt_cam.json`. Every entry must be a complete RUOD-trained
detector:

```json
{
  "id": "imagenet_dino100e_ruod_cascade",
  "kind": "detector",
  "config": "configs/exp_2/cascade-rcnn_r50_dino-official_fpn_2x_ruod.py",
  "checkpoint": "/path/to/RUOD_CascadeRCNN/best.pth",
  "layers": {
    "res2": "layer1",
    "res3": "layer2",
    "res4": "layer3",
    "res5": "layer4"
  }
}
```

The default detector file contains four verified downstream RUOD Cascade
R-CNNs initialized from ImageNet DINO 100e, RealUW DINO 100e, Synthetic5 DINO
100e, and the ImageNet-DINO-to-DFUI adapted backbone. The DFUI entry uses the
joint DFUI + RUOD-Easy + UIIS10K-Easy S1 trajectory and its epoch-18 S2 RUOD
checkpoint. The companion pretrained-backbone file contains the corresponding
four upstream backbones, including the joint DFUI S1 `backbone_only.pth`.

## Check paths

Run from the MMDetection repository root:

```bash
conda activate /media/SSD1/conda_envs/mmdet
unset LD_LIBRARY_PATH LD_PRELOAD

python scripts/exp_2/features/further_features/check_fixed_gt_cam_inputs.py \
  --annotation-file /media/HDD0/XCX/exp_2/RUOD/coco/annotations/instances_val.json \
  --image-root /media/HDD0/XCX/exp_2/RUOD/coco/val \
  --models-config scripts/exp_2/features/further_features/models.fixed_gt_cam.json
```

This checks files and the annotation dataset. The extractor performs the final
runtime checks for ROI-head compatibility, class count, layer hooks and
checkpoint loading.

## Full run

The example below uses one detector per GPU and processes at most five largest
GT instances per selected image. Selection is deterministic and shared by all
models.

```bash
cd ~/xcx/exp_2/mmdetection
conda activate /media/SSD1/conda_envs/mmdet
unset LD_LIBRARY_PATH LD_PRELOAD

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=fixed_gt_xgradcam_ruod50_${STAMP}
OUT_ROOT=/media/HDD2/XCX/exp_2/further_features/${RUN_NAME}

nohup env \
  RUN_NAME="$RUN_NAME" \
  OUT_ROOT="$OUT_ROOT" \
  RUOD_ROOT=/media/HDD0/XCX/exp_2/RUOD/coco \
  MODELS_CONFIG=$PWD/scripts/exp_2/features/further_features/models.fixed_gt_cam.json \
  SAMPLES=50 \
  SEED=2026 \
  MAX_INSTANCES_PER_IMAGE=5 \
  INSTANCE_ORDER=area-desc \
  LAYERS=res2,res3,res4,res5 \
  CASCADE_STAGE=-1 \
  REFERENCE_MODEL=imagenet_dino100e_ruod_cascade \
  LOW_PERCENTILE=1 \
  HIGH_PERCENTILE=99 \
  DISPLAY_GAMMA=1.0 \
  DEVICES=cuda:4,cuda:5 \
  PARALLEL_MODELS=2 \
  RESUME=1 \
  bash scripts/exp_2/features/further_features/run_fixed_gt_xgradcam_analysis.sh \
  > "$OUT_ROOT/launcher.log" 2>&1 &

echo $! | tee "$OUT_ROOT/launcher.pid"
echo "OUT_ROOT=$OUT_ROOT"
```

Use `DISPLAY_GAMMA=0.5` only as a separate brighter display variant. Gamma is
never used for metric calculation.

## Monitor

```bash
OUT_ROOT=/media/HDD2/XCX/exp_2/further_features/<run-name> \
INTERVAL=15 \
bash scripts/exp_2/features/further_features/monitor_fixed_gt_xgradcam.sh
```

`Ctrl+C` stops only the monitor.

## Output layout

```text
RUN_ROOT/
  sample/
    manifest.jsonl
    annotations.coco.json
    sampling.json
  fixed_gt_cam/
    raw_cam/MODEL/image_*/ann_*/LAYER.npz
    raw_cam/MODEL/image_*/ann_*/instance.json
    raw_cam_index.jsonl
    raw_cam_index_summary.json
    model_load_reports/*.json
    extraction_summaries/*.json
  rendered/
    independent_p1_p99/MODEL/LAYER/*.png
    imagenet_reference_p1_p99/MODEL/LAYER/*.png
    panels/independent_p1_p99/*.png
    panels/imagenet_reference_p1_p99/*.png
    normalization_scales.tsv
    metrics/instance_layer_metrics.tsv
    metrics/model_layer_summary.tsv
    metrics/model_layer_category_summary.tsv
    metrics/model_layer_size_summary.tsv
    figures/*.png
    figures/*.pdf
    render_summary.json
    COMPLETE.json
  logs/
  COMPLETE.env
```

Every rendered strategy contains pure heatmaps, overlays and overlays with the
fixed GT box. Panels put models in columns and backbone stages in rows.

## Resume and rerender

Raw extraction is resumable at the model/instance/layer level. To rerender with
different percentiles without recomputing CAM, set `RUN_SAMPLE=0 RUN_EXTRACT=0`
and reuse the same `SAMPLE_ROOT` and `CAM_ROOT`.

## Interpretation limits

- Fixed-GT CAM conditions the model on a correct ROI. It does not measure RPN,
  regression or NMS behavior and does not claim that a missed detector proposed
  this ROI during normal inference.
- Do not calculate metrics from blue-yellow PNG files.
- Do not compare `res2` color strength directly with `res5`; normalization is
  layer-specific.
- Keep one common GT annotation, class, layer, Cascade stage, preprocessing and
  normalization rule across models.
- Prediction-conditioned CAM is useful as a separate error-analysis branch but
  must not replace fixed-GT CAM for the controlled cross-model comparison.
