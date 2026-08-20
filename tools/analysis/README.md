# RUOD CKA, Frequency, and Prediction-CAM Analysis

This directory is a compact reproducible workflow for four pretrained
ResNet-50 DINO backbones and the four corresponding RUOD Cascade R-CNN
detectors. It intentionally avoids the old workflow's large all-frequency
activation-image exports.

## Included analyses

| Analysis | Models | Inputs | Main output |
| --- | --- | --- | --- |
| Same-layer CKA | 4 pretrained, then 4 RUOD detectors | Clean RUOD images | One heatmap per group; both use the ImageNet-to-RUOD Cascade backbone as the horizontal reference |
| Frequency response | 4 pretrained and 4 RUOD detectors | clean, low, mid, high, remove_low, remove_mid, remove_high | Feature RMS/clean-feature RMS and log foreground/background-ratio figures |
| Prediction CAM | 4 RUOD Cascade R-CNN detectors | Clean RUOD images | Prediction-conditioned XGradCAM, max over predictions, JET, per-image normalization, 5 x 4 panels |

`models.features.example.json` provides all eight models for feature
extraction. `models.cam.example.json` has only the four complete detectors.
Confirm checkpoint paths before starting a server job.

## Start a 20-image run

Run from `~/xcx/exp_2/mmdetection` after activating
`/media/SSD1/conda_envs/mmdet` and unsetting `LD_LIBRARY_PATH` and
`LD_PRELOAD`:

```bash
OUT_ROOT=/media/HDD2/XCX/exp_2/analysis/ruod100_energy_cam_cka SAMPLES=100 CAM_SAMPLES=30 SEED=2026 FEATURE_GPUS=cuda:2,cuda:3,cuda:6,cuda:7 CPU_WORKERS=16 FREQUENCY_MODEL_WORKERS=4 CAM_DEVICES=cuda:4,cuda:5,cuda:6,cuda:7 CAM_PARALLEL_MODELS=4 bash tools/analysis/run_ruod_analysis_suite.sh
```

Default RUOD paths are:

```text
/media/HDD0/XCX/exp_2/RUOD/coco
/media/HDD0/XCX/exp_2/RUOD/coco/val
/media/HDD0/XCX/exp_2/RUOD/coco/annotations/instances_val.json
```

The frequency bands use the paper-ready `soft-cpp` filter and the
`dataset-energy` policy. Its two cutoffs are obtained from the sampled RUOD
set's mean RGB power spectrum at cumulative energy quantiles `1/3` and `2/3`.
For a paper run, provide `ENERGY_CALIBRATION_MANIFEST=/path/to/larger_manifest.jsonl`
to estimate the cutoffs from a larger fixed RUOD calibration subset while still
rendering features only for the selected analysis images.

`SAMPLES` controls the shared statistical set for frequency analysis and CKA.
`CAM_SAMPLES` controls a deterministic random subset drawn only from that
shared set for Prediction-CAM. For example, `SAMPLES=100 CAM_SAMPLES=30`
extracts features for all 100 images but renders CAM only for 30 of those 100.
The CAM subset uses `CAM_SEED=SEED+1009` by default and is saved in
`OUT_ROOT/cam_sample/selection.json` for reproducibility.

Feature extraction runs in two GPU waves. The four bare pretrained backbones
are assigned to `FEATURE_GPUS` first, then the four corresponding RUOD Cascade
backbones reuse the same GPUs. Each worker writes to a private staging directory
before its completed feature tree is merged into the shared `feature_store`.
`FREQUENCY_MODEL_WORKERS` parallelizes CPU frequency metrics by model, while
`CPU_WORKERS` sets the default number of independent FFT/calibration image
workers. Each such process is restricted to one native BLAS/FFT thread to
avoid CPU oversubscription.

## Resume selected stages

Do not rerun completed extraction just to produce figures. Disable completed
stages and point `OUT_ROOT` to the same run:

```bash
OUT_ROOT=/media/HDD2/XCX/exp_2/analysis/ruod20_energy_cam_cka RUN_SAMPLE=0 RUN_BANDS=0 RUN_FEATURES=0 RUN_CKA=1 RUN_FREQUENCY=1 RUN_CAM=0 bash tools/analysis/run_ruod_analysis_suite.sh
```

Set `OVERWRITE=1` only when regenerating the selected stage's output.

## Detector prediction overlays

`visualize_detector_predictions.py` renders scored detector predictions over a
shared RUOD manifest. It writes both a uniform yellow-box version and a
detector-specific color version from the same inference pass. Box width and
label font size are adaptive to each source image's short side, with clamped
defaults that avoid oversized labels on small images and unreadable labels on
large images. `run_detector_prediction_visualization.sh` runs one detector per
requested GPU and composes one 2 x 2 detector-comparison panel per image for
each color mode.

```bash
SAMPLE_ROOT=/media/HDD2/XCX/exp_2/analysis/ruod100_energy_cam30_parallel_20260816_010422/sample \
OUT_ROOT=/media/HDD2/XCX/exp_2/analysis/ruod100_detector_predictions \
DEVICES=cuda:2,cuda:3,cuda:6,cuda:7 \
PARALLEL_MODELS=4 \
SCORE_THRESHOLD=0.30 \
MAX_DETECTIONS=30 \
bash tools/analysis/run_detector_prediction_visualization.sh
```

The main output paths are:

```text
OUT_ROOT/
  uniform/images/<model>/image_XXXXXXXX.png
  uniform/panels_2x2/image_XXXXXXXX.png
  model/images/<model>/image_XXXXXXXX.png
  model/panels_2x2/image_XXXXXXXX.png
  metadata/<model>.jsonl
  COMPLETE.json
```

## Output layout

```text
OUT_ROOT/
  sample/                         # sampled RUOD images, GT boxes, manifest
  cam_sample/                     # deterministic CAM-only subset of sample/
  frequency_inputs/               # dataset-energy frequency variants
  feature_store/                  # pooled + spatial features for eight models
  cka/
    pretrained/same_layer_cka.{png,pdf,tsv}
    detector/same_layer_cka.{png,pdf,tsv}
  frequency/
    feature_norm_over_clean_pretrained.{png,pdf}
    feature_norm_over_clean_detector.{png,pdf}
    fg_bg_ratio_pretrained.{png,pdf}
    fg_bg_ratio_detector.{png,pdf}
    frequency_per_sample.tsv
    frequency_summary.tsv
  cam_prediction/
    raw/                          # raw prediction-conditioned CAM tensors
    jet_per_image_max/
      panels_5x4/legacy_jet/      # one final panel per image
```

## Definitions and constraints

`feature_norm_over_clean` is RMS(raw CHW feature for a frequency variant)
divided by RMS(raw CHW clean feature), matched by model, sample, and layer.
Thus the clean curve is exactly 1.0 and every other point reports the
representation magnitude relative to the corresponding clean-input baseline.
FG/BG is the mean absolute channel activation in the union of
RUOD GT boxes divided by the background complement. The frequency figures are
split into pretrained and detector groups. Pretrained plots append the
ImageNet-to-RUOD Cascade backbone as a distinct reference curve.

Prediction-CAM aggregates a model's valid predicted-box CAMs using the
pixelwise maximum. It is independently min-max normalized per
`image x model x layer` and rendered with OpenCV JET. It is therefore a
spatial-attention visualization, not an absolute activation-strength
comparison across models.

CKA includes only semantically corresponding layers. Both heatmaps use the
ImageNet-to-RUOD Cascade backbone as the horizontal reference. Its detector
self-comparison is excluded because linear `CKA(X, X) = 1` by definition.
