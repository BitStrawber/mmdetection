# Detection-box visualizations

## Controlled comparison

Use fixed-GT XGradCAM when the same object, box, class, Cascade stage, and layer must be held constant across models:

```bash
CAM_ROOT=/path/to/fixed_gt/raw \
OUT_DIR=/path/to/fixed_gt/rendered \
bash tools/analysis/final_visualizations/detection_boxes/render_fixed_gt_boxes.sh
```

The publication-ready files end in `_with_gt.png`; comparison panels are under `panels/`.

## Detector behavior/error analysis

Use prediction-conditioned XGradCAM for actual post-NMS predictions:

```bash
CAM_ROOT=/path/to/prediction/raw \
OUT_DIR=/path/to/prediction/rendered \
MODELS=imagenet_dino100e_ruod_cascade,realuw_dino100e_ruod_cascade \
bash tools/analysis/final_visualizations/detection_boxes/render_prediction_boxes.sh
```

The `with_boxes/` outputs show the prediction box in yellow and the best matched GT box in green. These prediction instances are model-specific and should not be presented as a controlled cross-model comparison.

## Full-image prediction panels

`render_supervision_predictions.py` is the ordinary prediction visualization
route. It runs MMDetection inference and converts `pred_instances` to the
official `supervision.Detections` object. Boxes, masks, and labels are rendered
by `supervision.BoxAnnotator`, `supervision.MaskAnnotator`, and
`supervision.LabelAnnotator`.

Defaults match the paper comparison panels:

```text
score threshold : 0.50
maximum boxes   : 15 per image
```

Detection models produce `box/`. Instance-segmentation models can produce all
three variants:

```text
box/
mask/
box_mask/
```

Example:

```bash
python tools/analysis/final_visualizations/detection_boxes/render_supervision_predictions.py \
  --config /path/to/runtime_or_resolved_config.py \
  --checkpoint /path/to/best.pth \
  --image-list /path/to/images.txt \
  --output-root /path/to/output/model \
  --model-name r50 \
  --task segment \
  --score-threshold 0.50 \
  --max-detections 15
```

`compose_prediction_panels.py` creates `Input | model A | model B` panels from
the corresponding output directories. This prediction route is separate from
the CAM scripts above and does not use fixed-GT XGradCAM.
