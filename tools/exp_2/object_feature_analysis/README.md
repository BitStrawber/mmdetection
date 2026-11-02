# Object Feature Analysis

This package implements the object-level feature analysis pipeline for six
frozen backbones on RUOD.

## Pipeline

1. Build a deterministic, class-balanced, scale-stratified instance manifest.
2. Load six pretrained backbones through the existing backbone analysis adapter.
3. Apply the MMDetection test pipeline to images and GT boxes.
4. Capture intermediate features and apply torchvision RoIAlign.
5. Save aligned feature bundles with provenance metadata.
6. Compute Fisher discriminant ratio, linear-probe metrics, and paired bootstrap CIs.
7. Render independent PCA/t-SNE panels for each backbone/source pair.

## Commands

Run from the repository root:

```bash
python -m tools.exp_2.object_feature_analysis run-all \
  --config tools/exp_2/object_feature_analysis/configs/object_analysis.example.json \
  --out-dir outputs/object_feature_analysis
```

The individual stages are:

```bash
python -m tools.exp_2.object_feature_analysis sample-instances \
  --config tools/exp_2/object_feature_analysis/configs/object_analysis.example.json \
  --out-dir outputs/object_feature_analysis/manifest
```

```bash
python -m tools.exp_2.object_feature_analysis extract-roi \
  --config tools/exp_2/object_feature_analysis/configs/object_analysis.example.json \
  --manifest outputs/object_feature_analysis/manifest/instances.jsonl \
  --out-dir outputs/object_feature_analysis/features
```

```bash
python -m tools.exp_2.object_feature_analysis compute-metrics \
  --config tools/exp_2/object_feature_analysis/configs/object_analysis.example.json \
  --features-dir outputs/object_feature_analysis/features/bundles \
  --out-dir outputs/object_feature_analysis/metrics
```

```bash
python -m tools.exp_2.object_feature_analysis render-figures \
  --config tools/exp_2/object_feature_analysis/configs/object_analysis.example.json \
  --features-dir outputs/object_feature_analysis/features/bundles \
  --metrics-dir outputs/object_feature_analysis/metrics \
  --out-dir outputs/object_feature_analysis/figures
```

## Data Contracts

The manifest contains one selected instance per JSONL row. Required fields:

```text
instance_id, annotation_id, image_id, category_id, split, file_name,
image_path, image_width, image_height, bbox_xyxy, normalized_scale,
size_bin, annotation_file
```

Each feature bundle is a compressed NPZ containing:

```text
features, labels, instance_ids, image_ids, group_ids, bboxes, split
```

The corresponding `.metadata.json` stores provenance, RoIAlign options,
feature shape, manifest hash, and a SHA256 of the NPZ.

## Box Mapping Validation

The extractor writes mapping QA tables and optional image/feature overlays:

```text
qa/<model_id>.mapping.tsv
qa/<model_id>.audit.json
qa/<model_id>/<layer>/<instance_index>.png
```

Every mapping record includes the resized-image box, projected feature box,
round-trip error, feature dimensions, and ROI statistics.

The deterministic mapping smoke test runs without torch:

```bash
python -m tools.exp_2.object_feature_analysis self-check --self-test
```

For a complete coordinate-feature test, run the same RoIAlign call on a
two-channel coordinate map. The ROI mean should match the normalized box
center. This test should be run in the Linux GPU environment with torchvision.

## Unit Tests

```bash
python -m unittest discover \
  -s tools/exp_2/object_feature_analysis/tests \
  -t .
```

The tests cover deterministic sampling, box projection round trips, feature
bundle validation, and Fisher ratio behavior.
