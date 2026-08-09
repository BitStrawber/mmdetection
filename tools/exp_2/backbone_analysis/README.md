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
  --method soft-cpp \
  --bands 'low:0:1/32,mid:1/32:1/8,high:1/8:max' \
  --transition-ratio 0.25 \
  --resize 1333x800 \
  --pad-fraction 0.05 \
  --model-input-mode natural-energy \
  --save-raw \
  --save-band-stop \
  --save-visualizations
```

The paper-ready default uses cycles/pixel rather than corner-normalized FFT
radius. Images are first resized with the detector's `(1333, 800)` keep-ratio
rule, then reflect-padded before FFT. The default bands correspond to spatial
wavelengths larger than 32 pixels, 8--32 pixels, and smaller than 8 pixels.
Raised-cosine transitions form a partition of unity, so the signed raw bands
reconstruct the resized clean image within floating-point tolerance.

Two frequency-band policies are supported and must be stored in different
experiment roots:

1. `--band-policy fixed` is the primary, architecture-independent experiment.
   It uses the explicit `--bands` cutoffs shown above, so the physical spatial
   scales have the same meaning across datasets and papers.
2. `--band-policy dataset-energy` is a data-driven sensitivity experiment. It
   estimates an equal-sample-weight mean radial power spectrum from a
   calibration manifest and selects two cutoffs at configurable cumulative
   energy quantiles (default one-third and two-thirds). It writes
   `dataset_energy_profile.tsv` and `dataset_energy_calibration.json`.

Example adaptive run:

```bash
python -m tools.exp_2.backbone_analysis.generate_frequency_bands \
  --manifest /path/to/analysis/sample/manifest.jsonl \
  --calibration-manifest /path/to/independent/calibration_manifest.jsonl \
  --out-dir /path/to/analysis/frequency_inputs_energy_adaptive \
  --method soft-cpp \
  --band-policy dataset-energy \
  --energy-quantiles '1/3,2/3' \
  --energy-bins 1024 \
  --energy-color-space rgb \
  --resize 1333x800 \
  --pad-fraction 0.05
```

Calibration removes the per-channel DC component, applies a 2D Hann window,
computes radial power in cycles/pixel, normalizes every image histogram to unit
energy, and only then averages across images. Thus large images do not dominate
the learned cutoffs. Prefer a fixed, independent calibration subset when one is
available. The adaptive bands are energy tertiles, not universal semantic
definitions of low/mid/high frequency, so report their numerical cutoffs and
wavelengths whenever they are used.

Artifacts are separated by purpose:

- `arrays/raw/{low,mid,high}/*.npy` contains signed, unclipped float32 bands.
- `images/{clean,low,mid,high}/*.png` contains model-ready band-pass inputs.
- `images/remove_{low,mid,high}/*.png` contains mean-preserved band-stop inputs.
- `visualizations/magnitude/...` contains percentile-stretched display images
  that must not be fed back into a model.
- `frequency_qa.tsv` records reconstruction error, raw energy, RMS scaling and
  clipping rates for every sample and band.

`natural-energy` preserves each band's natural amplitude. Use
`--model-input-mode equal-rms` as a separate control when comparing response
magnitudes across bands. Do not replace the natural-energy result with only an
equal-RMS result, because equalizing high-frequency energy creates an artificial
input distribution.

The old implementation remains available only for reproducing prior runs:

```bash
python -m tools.exp_2.backbone_analysis.generate_frequency_bands \
  --manifest /path/to/manifest.jsonl \
  --out-dir /path/to/legacy_frequency_inputs \
  --method legacy-hard-corner \
  --resize none \
  --bands 'low:0:0.15,mid:0.15:0.40,high:0.40:1.0' \
  --reconstruction mean-preserve
```

Legacy PNGs are clipped mean-preserved visual inputs and are not exact signed
frequency components.

## 3. Model configuration

Copy and edit `models.example.json`, or let `run_pipeline.sh` generate a runtime
configuration. A `backbone` entry builds a detector shell solely to reuse the
same MMDetection preprocessing and loads the checkpoint into `model.backbone`.
A `detector` entry loads a complete checkpoint but exports only configured
backbone hooks. When a `backbone` entry omits `checkpoint`, the loader preserves
and executes `model.backbone.init_cfg` from the config. This is the intended
mode for the supervised torchvision baseline in this experiment:

```json
{
  "id": "imagenet_backbone",
  "kind": "backbone",
  "config": "configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py"
}
```

That J2 config inherits `checkpoint='torchvision://resnet50'`. It must not be
replaced by a DINO 100e checkpoint. The paired detector uses the same J2 config
and the RUOD-trained J2 Cascade R-CNN checkpoint.

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
  --y-model imagenet_backbone \
  --x-model cascade_ruod_backbone \
  --variant clean \
  --layers-a res2,res3,res4,res5 \
  --layers-b res2,res3,res4,res5 \
  --out-dir /path/to/analysis/results/cka
```

The command writes the numeric matrix as NPY/TSV, metadata and an optional PNG.
The RUOD Cascade R-CNN is deliberately passed as `--x-model`, so its `res2`--
`res5` layers are the heatmap columns. `--model-a` and `--model-b` remain
backward-compatible aliases for `--y-model` and `--x-model`, respectively.

## 6. Frequency response

```bash
python -m tools.exp_2.backbone_analysis.compute_frequency_response \
  --feature-root /path/to/analysis/feature_store \
  --frequency-manifest /path/to/analysis/frequency_inputs/frequency_manifest.jsonl \
  --models imagenet_backbone,cascade_ruod_backbone \
  --layers res2,res3,res4,res5 \
  --frequency-variants low,mid,high,remove_low,remove_mid,remove_high \
  --out-dir /path/to/analysis/results/frequency_response
```

The analysis deliberately separates two complementary experiments:

- `low`, `mid` and `high` are band-pass inputs. Use `pooled_norm_ratio` to
  report retained feature response and `input_normalized_response_gain` to
  account for the very different natural image energy in each band.
- `remove_low`, `remove_mid` and `remove_high` are band-stop inputs. Use
  `pooled_relative_shift` and `pooled_cosine_distance` to report how much a
  clean-image representation changes when one band is removed. The
  `input_normalized_shift_gain` divides that change by the actual input-space
  perturbation.

Input RMS and shift are measured from the exact quantized PNGs passed through
the models, after removing each image's per-channel spatial mean. This captures
clipping and quantization rather than assuming the ideal float FFT component
was the model input. `frequency_input_summary.tsv` stores the input-only
statistics once per variant, while `frequency_response_summary.tsv` stores the
model/layer response statistics. `frequency_response_per_sample.tsv` preserves
the aligned sample and image IDs plus all input and feature values. The tables
include:

- `input_centered_rms_ratio` and `input_relative_shift`;
- `pooled_norm_ratio` and full-spatial `spatial_norm_ratio`;
- `pooled_relative_shift` (`normalized_feature_shift` is retained as an alias);
- pooled cosine similarity/distance;
- input-normalized response and shift gains.

Do not interpret raw band-pass feature norm alone as frequency preference:
natural underwater images contain much more low-frequency than high-frequency
energy. Report the natural-energy branch as the primary result, the band-stop
branch as causal sensitivity evidence, and equal-RMS band-pass inputs only as
a controlled supplementary experiment.

## 6.1 Frequency-analysis figures

Input-only figures are rendered independently from model results:

```bash
python -m tools.exp_2.backbone_analysis.render_frequency_inputs \
  --frequency-root /path/to/run/frequency_inputs \
  --out-dir /path/to/run/analysis/frequency_inputs \
  --spectrum-variants remove_low,remove_mid,remove_high \
  --panel-samples 6 \
  --overwrite
```

This writes the clean/difference radial spectra, input band-energy chart,
chart-ready TSV data, and clean/low/mid/high image panels. These products only
validate and describe the input decomposition; they are not evidence of model
frequency dependence.

Model-response figures are rendered from the response tables:

```bash
python -m tools.exp_2.backbone_analysis.render_frequency_response \
  --response-dir /path/to/run/analysis/frequency_response \
  --activation-root /path/to/run/analysis/activation_by_frequency \
  --detection-metrics /path/to/run/analysis/frequency_detection/frequency_detection_metrics.tsv \
  --out-dir /path/to/run/analysis/frequency_figures \
  --overwrite
```

It produces PNG/PDF and backing TSV files for:

- band--Feature Norm curves with P05--P95 intervals;
- band-pass and band-stop Feature Distance curves relative to clean;
- input-normalized Feature Distance curves that divide by actual input shift;
- layer--band heatmaps with one shared scale across compared models;
- frequency-conditioned foreground/background response curves;
- band-stop bbox AP retention;
- feature-shift versus AP-drop scatter plots.

## 6.2 Band-stop detection AP

AP requires complete detector checkpoints. A standalone backbone cannot produce
an AP value and is intentionally skipped. Configure each evaluable detector as
`kind: detector` in `models.json`, then run:

```bash
python -m tools.exp_2.backbone_analysis.evaluate_frequency_detection \
  --frequency-manifest /path/to/run/frequency_inputs/frequency_manifest.jsonl \
  --annotation-file /path/to/ruod_instances.json \
  --models-config /path/to/four_models.json \
  --variants clean,remove_low,remove_mid,remove_high \
  --device cuda:0 \
  --out-dir /path/to/run/analysis/frequency_detection \
  --overwrite
```

The command saves COCO-format predictions, AP/AP50/AP75/APs/APm/APl, AP drop,
and AP retention. AP on 50 sampled images is exploratory and statistically
noisy. Generate a frequency manifest for the complete RUOD validation set for
the primary paper result.

## 6.3 Fourier basis sensitivity

The three broad bands cannot produce a two-dimensional Fourier sensitivity
map. This separate experiment adds phase-averaged sinusoidal perturbations at a
configurable `(fx, fy)` grid and measures normalized feature shift:

```bash
python -m tools.exp_2.backbone_analysis.compute_fourier_basis_sensitivity \
  --manifest /path/to/run/frequency_inputs/frequency_manifest.jsonl \
  --models-config /path/to/four_models.json \
  --models model_a,model_b,model_c,model_d \
  --layers res2,res3,res4,res5 \
  --filter-config /path/to/run/frequency_inputs/filter_config.json \
  --samples 10 \
  --grid-size 15 \
  --amplitude 0.031372549 \
  --device cuda:0 \
  --out-dir /path/to/run/analysis/fourier_basis_sensitivity \
  --overwrite
```

The heatmaps overlay the active fixed or dataset-energy radial cutoffs. Runtime
scales as `models x samples x grid_size^2 x phases`, so this stage is disabled
by default. It reports feature sensitivity, not detection-loss sensitivity.

## 6.4 Four or more models

Pass an external model configuration and explicit model IDs instead of using
the generated two-model default:

```bash
MODELS_CONFIG_INPUT=/path/to/four_models.json
ANALYSIS_MODELS=model_a,model_b,model_c,model_d
CKA_Y_MODELS=model_a,model_b,model_c
CKA_X_MODEL=imagenet_dino100e_ruod_cascade
```

Keep the RUOD detector in `CKA_X_MODEL` and put every comparison backbone in
`CKA_Y_MODELS`. The resulting matrix has the RUOD `res2`--`res5` layers as its
four columns and stacked `MODEL/LAYER` rows for all comparison models. For five
models (one RUOD reference plus four comparison models), this is a `16 x 4`
heatmap. A single `CKA_Y_MODEL` and the older `CKA_MODEL_A/B` variables remain
supported for pairwise runs.

All backbone entries can produce feature-response and FG/BG figures. Only
`kind: detector` entries produce band-stop AP metrics.

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
BACKBONE_CONFIG=configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py \
CASCADE_CONFIG=configs/exp_2/cascade-rcnn_r50_fpn_2x_ruod_j2.py \
CASCADE_CHECKPOINT=/media/SSD1/XCX/exp_2/BitStrawber_Output/J2/det/checkpoint/best_coco_bbox_mAP_epoch_18.pth \
OUT_ROOT=/path/to/analysis/run01 \
SAMPLES=50 DEVICE=cuda:0 \
bash tools/exp_2/backbone_analysis/run_pipeline.sh
```

The orchestrator defaults to the paper-ready frequency settings below:

```bash
FREQUENCY_METHOD=soft-cpp
FREQUENCY_BAND_POLICY=fixed
BANDS='low:0:1/32,mid:1/32:1/8,high:1/8:max'
FREQUENCY_TRANSITION_RATIO=0.25
FREQUENCY_RESIZE=1333x800
FREQUENCY_PAD_FRACTION=0.05
FREQUENCY_MODEL_INPUT_MODE=natural-energy
```

For a second dataset-energy experiment root, use:

```bash
FREQUENCY_BAND_POLICY=dataset-energy
FREQUENCY_ENERGY_QUANTILES='1/3,2/3'
FREQUENCY_ENERGY_BINS=1024
FREQUENCY_ENERGY_COLOR_SPACE=rgb
FREQUENCY_CALIBRATION_MANIFEST=/path/to/calibration_manifest.jsonl
```

Run a second output root with `FREQUENCY_MODEL_INPUT_MODE=equal-rms` for the
energy-controlled feature-response comparison. Include
`remove_low,remove_mid,remove_high` in `VARIANTS` when extracting features for
the band-stop branch, and set `FREQUENCY_RESPONSE_VARIANTS` to the same list
when computing its response table. Keep the natural-energy and equal-RMS
outputs in separate directories so their manifests and model inputs cannot be
mixed.

Set any `RUN_*` flag to zero to skip a stage. For example, recompute only CKA:

```bash
RUN_SAMPLE=0 RUN_FREQUENCY_IMAGES=0 RUN_FEATURES=0 \
RUN_FREQUENCY_RESPONSE=0 RUN_ACTIVATION=0 RUN_CKA=1 \
OVERWRITE=1 ... bash tools/exp_2/backbone_analysis/run_pipeline.sh
```

The orchestrator defaults to `RUN_TSNE=0` because t-SNE is optional in the
current experiment.

## 10. Fixed and adaptive runs together

`run_dual_frequency_analysis.sh` executes the same product contract under two
separate roots, preventing fixed and adaptive features from being mixed:

```bash
BASE_OUT_ROOT=/media/SSD2/XCX/exp_2/backbone_analysis/frequency_dual \
RUOD_ANN=/path/to/instances.json \
RUOD_IMAGE_ROOT=/path/to/images \
MODELS_CONFIG_INPUT=/path/to/four_models.json \
ANALYSIS_MODELS=model_a,model_b,model_c,model_d \
FREQUENCY_CALIBRATION_MANIFEST=/path/to/calibration_manifest.jsonl \
SAMPLES=50 DEVICE=cuda:0 \
RUN_DETECTION_FREQUENCY_EVAL=1 \
RUN_FOURIER_SENSITIVITY=1 \
bash tools/exp_2/backbone_analysis/run_dual_frequency_analysis.sh
```

The resulting roots are:

```text
frequency_dual/fixed/
frequency_dual/dataset_energy/
```

Each contains input QA, model features, response tables, activation statistics,
optional detector AP, Fourier sensitivity matrices, and PNG/PDF figures. For
expensive Fourier analysis the clean sensitivity matrix is mathematically
independent of the broad-band policy; running it twice only changes the cutoff
overlays. It may be computed once and re-rendered with the two filter configs
when compute time is limited.

### 10.1 One model per GPU, policies in sequence

For several models, use the parallel-model orchestrator. It runs the fixed
policy first, assigns one model to one GPU, validates every model/variant, runs
the unified analysis stages, and only then starts the dataset-energy policy:

```bash
BASE_OUT_ROOT=/path/to/run \
RUOD_ANN=/path/to/instances_val.json \
RUOD_IMAGE_ROOT=/path/to/val \
MODELS_CONFIG_INPUT=/path/to/five_models.json \
GPUS=2,3,4,5,6,7 ANALYSIS_GPU=7 \
POLICIES=fixed,dataset-energy \
SAMPLES=100 SPATIAL_SAMPLES=100 \
RUN_TSNE=1 \
ACTIVATION_JOBS=7 ACTIVATION_PNG_COMPRESS_LEVEL=1 \
ACTIVATION_REUSE_COMPLETE=1 \
bash tools/exp_2/backbone_analysis/run_dual_frequency_analysis_parallel_models.sh
```

The recovery unit is one `model/variant`. A complete task is reused from the
official feature store; an incomplete task is regenerated in
`.parallel_feature_stage`, validated, and then merged. Workers write different
model directories, so they never concurrently write the same feature files.
Existing fixed-policy results can therefore be resumed without deleting them.
Per-model worker logs are stored under
`POLICY_ROOT/logs/parallel_features/`.

Frequency activation variants can be rendered concurrently because each uses
an independent output directory. `ACTIVATION_JOBS=7` runs all seven variants
in parallel while preserving shared normalization across models inside each
variant. `ACTIVATION_PNG_COMPRESS_LEVEL=1` uses faster lossless PNG encoding;
it changes encoded file size, not image pixels or analysis values. Variant logs
are stored under `analysis/activation_by_frequency_logs/`.
Complete variant outputs are reused by default; a partial variant is rendered
again so interrupted analysis runs can be resumed safely.
