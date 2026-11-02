from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .bbox_mapping import (
    clip_boxes_to_image,
    project_boxes_to_feature,
    roundtrip_error,
    validate_boxes,
)
from .common import (
    ensure_empty_or_create,
    existing_file,
    file_sha256,
    read_jsonl,
    write_json,
)
from .config import load_config, resolve_model_paths
from .feature_io import save_feature_bundle
from .validate_mapping import render_mapping_debug


def _shape_pair(value: Any, fallback: Sequence[int]) -> tuple[int, int]:
    if value is None:
        return int(fallback[0]), int(fallback[1])
    return int(value[0]), int(value[1])


def _tensor_from_output(output: Any):
    if hasattr(output, "shape") and hasattr(output, "detach"):
        return output
    if isinstance(output, (tuple, list)):
        tensors = [
            value
            for value in output
            if hasattr(value, "shape") and hasattr(value, "detach")
        ]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f"Hook output is not a single tensor: {type(output).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract object-level ROI features from frozen backbones."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", default="", help="Optional comma-separated model IDs.")
    parser.add_argument("--debug-samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _selected_models(
    config: Mapping[str, Any],
    requested: str,
) -> List[Dict[str, Any]]:
    models = list(config["models"])
    if not requested:
        return models
    wanted = {part.strip() for part in requested.split(",") if part.strip()}
    found = {str(model["id"]) for model in models}
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown model IDs: {sorted(missing)}")
    return [model for model in models if str(model["id"]) in wanted]


def _prepare_pipeline(config_path: str) -> list[Dict[str, Any]]:
    from mmengine.config import Config

    cfg = Config.fromfile(str(existing_file(config_path)))
    pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
    if not pipeline:
        raise ValueError(f"Empty test pipeline: {config_path}")
    if not any(item.get("type") == "LoadAnnotations" for item in pipeline):
        insert_at = 1 if pipeline[0].get("type") == "LoadImageFromFile" else 0
        pipeline.insert(
            insert_at,
            {
                "type": "LoadAnnotations",
                "with_bbox": True,
                "with_label": True,
            },
        )
    if not any(item.get("type") == "PackDetInputs" for item in pipeline):
        pipeline.append({"type": "PackDetInputs"})
    return pipeline


def _engine():
    from mmdet.utils import register_all_modules

    register_all_modules()


def _tensor_boxes(value: Any):
    boxes = value.tensor if hasattr(value, "tensor") else value
    if not hasattr(boxes, "detach"):
        raise TypeError(f"Expected tensor boxes, got {type(value).__name__}")
    return boxes.detach()


def _infer_token_grid(
    token_count: int,
    pad_shape: Sequence[int],
    patch_size: int,
) -> tuple[int, int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive for token features")
    candidate_h = int(math.ceil(int(pad_shape[0]) / float(patch_size)))
    candidate_w = int(math.ceil(int(pad_shape[1]) / float(patch_size)))
    if candidate_h * candidate_w == token_count:
        return candidate_h, candidate_w
    target_ratio = int(pad_shape[0]) / float(int(pad_shape[1]))
    best = None
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        for grid_h, grid_w in ((height, token_count // height), (token_count // height, height)):
            score = abs((grid_h / float(grid_w)) - target_ratio)
            if best is None or score < best[0]:
                best = (score, grid_h, grid_w)
    if best is None:
        raise ValueError(f"Cannot infer a token grid for {token_count} tokens")
    return best[1], best[2]


def _resolve_token_grid(
    token_count: int,
    pad_shape: Sequence[int],
    patch_size: int,
    drop_tokens: int,
) -> tuple[int, int, int]:
    candidate_h = int(math.ceil(int(pad_shape[0]) / float(patch_size)))
    candidate_w = int(math.ceil(int(pad_shape[1]) / float(patch_size)))
    expected_count = candidate_h * candidate_w
    if token_count == expected_count:
        return candidate_h, candidate_w, 0
    if drop_tokens and token_count - drop_tokens == expected_count:
        return candidate_h, candidate_w, drop_tokens
    accepted = [count for count in (token_count, token_count - drop_tokens) if count > 0]
    for count in accepted:
        grid_h, grid_w = _infer_token_grid(count, pad_shape, patch_size)
        if grid_h * grid_w == count:
            return grid_h, grid_w, token_count - count
    raise ValueError(
        f"Cannot map {token_count} tokens to a patch grid for pad_shape={tuple(pad_shape)}"
    )


def _feature_to_chw(
    value: Any,
    layout: str,
    output_kind: str,
    pad_shape: Sequence[int],
    patch_size: int,
    drop_tokens: int,
):
    tensor = _tensor_from_output(value)
    if tensor.ndim == 4:
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"Expected batch size one, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if output_kind == "patch_tokens" or (
        output_kind == "auto" and tensor.ndim == 2
    ):
        token_count, channels = int(tensor.shape[0]), int(tensor.shape[1])
        grid_h, grid_w, tokens_to_drop = _resolve_token_grid(
            token_count, pad_shape, patch_size, drop_tokens
        )
        if tokens_to_drop:
            tensor = tensor[tokens_to_drop:]
        return tensor.reshape(grid_h, grid_w, channels).permute(2, 0, 1).contiguous()
    if layout == "channels_last":
        if tensor.ndim != 3:
            raise ValueError(f"Expected H x W x C, got {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1)
    elif layout != "channels_first":
        raise ValueError(f"Unknown feature layout: {layout}")
    if tensor.ndim != 3:
        raise ValueError(f"Expected C x H x W, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _crop_valid_feature(
    feature,
    img_shape: Sequence[int],
    pad_shape: Sequence[int],
):
    feature_h, feature_w = int(feature.shape[-2]), int(feature.shape[-1])
    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    pad_h, pad_w = int(pad_shape[0]), int(pad_shape[1])
    valid_h = min(
        feature_h,
        max(1, int(math.ceil(img_h * feature_h / float(max(pad_h, 1))))),
    )
    valid_w = min(
        feature_w,
        max(1, int(math.ceil(img_w * feature_w / float(max(pad_w, 1))))),
    )
    return feature[:, :valid_h, :valid_w], (valid_h, valid_w)


def _call_extract_feat(model: Any, inputs: Any, samples: Sequence[Any]) -> None:
    kwargs = {}
    signature = inspect.signature(model.extract_feat)
    if "batch_data_samples" in signature.parameters:
        kwargs["batch_data_samples"] = samples
    model.extract_feat(inputs, **kwargs)


def _build_manifest_groups(
    rows: Sequence[Dict[str, Any]],
) -> OrderedDict[tuple[str, int], List[tuple[int, Dict[str, Any]]]]:
    groups: OrderedDict[tuple[str, int], List[tuple[int, Dict[str, Any]]]] = (
        OrderedDict()
    )
    for index, row in enumerate(rows):
        key = (str(row["split"]), int(row["image_id"]))
        groups.setdefault(key, []).append((index, row))
    return groups


def _write_mapping_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = (
        "model_id",
        "layer",
        "instance_index",
        "instance_id",
        "image_id",
        "split",
        "img_height",
        "img_width",
        "pad_height",
        "pad_width",
        "feature_height",
        "feature_width",
        "valid_feature_height",
        "valid_feature_width",
        "bbox_network",
        "bbox_feature",
        "roundtrip_max_error",
        "roi_rows",
        "roi_columns",
        "roi_mean_abs",
        "network_box_clip_overshoot",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _debug_indices(
    rows: Sequence[Mapping[str, Any]],
    requested: int,
) -> set[int]:
    if requested <= 0:
        return set()
    strata: Dict[tuple[int, int], List[int]] = {}
    for index, row in enumerate(rows):
        key = (int(row["category_id"]), int(row["size_bin"]))
        strata.setdefault(key, []).append(index)
    keys = sorted(strata, key=str)
    offsets = {key: 0 for key in keys}
    selected: List[int] = []
    while len(selected) < min(int(requested), len(rows)):
        progressed = False
        for key in keys:
            values = strata[key]
            offset = offsets[key]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[key] += 1
            progressed = True
            if len(selected) >= requested:
                break
        if not progressed:
            break
    return set(selected)


def extract_model(
    spec: Mapping[str, Any],
    manifest_rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    pipeline: Sequence[Mapping[str, Any]],
    out_dir: Path,
    device: str,
    roi_options: Mapping[str, Any],
    debug_samples: int,
) -> None:
    import torch
    from mmengine.dataset import Compose
    from torchvision.ops import roi_align
    from ..backbone_analysis.model_adapter import load_model

    _engine()
    loaded = load_model(spec, device)
    model_id = str(spec["id"])
    layers = {str(key): str(value) for key, value in spec["layers"].items()}
    feature_layout = str(spec.get("feature_layout", "channels_first"))
    feature_output = str(spec.get("feature_output", "auto"))
    patch_size = int(spec.get("patch_size", 16))
    drop_tokens = int(spec.get("drop_tokens", 0))
    compose = Compose(list(pipeline))
    groups = _build_manifest_groups(manifest_rows)
    debug_rows = _debug_indices(manifest_rows, debug_samples)
    output_size = tuple(int(value) for value in roi_options.get("output_size", [7, 7]))
    sampling_ratio = int(roi_options.get("sampling_ratio", 2))
    aligned = bool(roi_options.get("aligned", True))
    box_clip_tolerance = float(roi_options.get("box_clip_tolerance", 2.0))

    per_layer: Dict[str, List[Any]] = {layer: [None] * len(manifest_rows) for layer in layers}
    mapping_rows: List[Dict[str, Any]] = []
    model_audit: Dict[str, Any] = {
        "model_id": model_id,
        "source": str(spec["source"]),
        "backbone": str(spec["backbone"]),
        "manifest_sha256": manifest_sha256,
        "config_sha256": file_sha256(spec["config"]),
        "checkpoint": str(spec["checkpoint"]),
        "feature_layout": feature_layout,
        "roi_align": {
            "output_size": list(output_size),
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
            "box_clip_tolerance": box_clip_tolerance,
        },
        "model_load_report": loaded.load_report,
        "layers": layers,
    }
    captures: Dict[str, Any] = {}
    handles = []
    try:
        for layer_id, module_name in layers.items():
            target = loaded.layers[layer_id]

            def hook(_module, _inputs, output, key=layer_id):
                captures[key] = output

            handles.append(target.register_forward_hook(hook))

        for image_position, ((split, image_id), image_rows) in enumerate(groups.items()):
            captures.clear()
            image_path = existing_file(image_rows[0][1]["image_path"])
            packed = compose(
                {
                    "img_id": int(image_id),
                    "img_path": str(image_path),
                    "instances": [
                        {
                            "bbox": [float(value) for value in row["bbox_xyxy"]],
                            "bbox_label": 0,
                            "ignore_flag": 0,
                        }
                        for _, row in image_rows
                    ],
                }
            )
            batch = loaded.model.data_preprocessor(
                {
                    "inputs": [packed["inputs"]],
                    "data_samples": [packed["data_samples"]],
                },
                training=False,
            )
            samples = batch["data_samples"]
            with torch.no_grad():
                _call_extract_feat(loaded.model, batch["inputs"], samples)
            sample = samples[0]
            metainfo = sample.metainfo
            img_shape = _shape_pair(
                metainfo.get("img_shape"), (int(image_rows[0][1]["image_height"]), int(image_rows[0][1]["image_width"]))
            )
            pad_shape = _shape_pair(
                metainfo.get("pad_shape", metainfo.get("batch_input_shape")),
                img_shape,
            )
            network_boxes = _tensor_boxes(sample.gt_instances.bboxes).float()
            if int(network_boxes.shape[0]) != len(image_rows):
                raise RuntimeError(
                    f"{model_id}: transformed box count differs from manifest"
                )
            network_numpy, network_clip_overshoot = clip_boxes_to_image(
                network_boxes.detach().cpu().numpy(),
                img_shape[0],
                img_shape[1],
                max_overshoot=box_clip_tolerance,
            )
            validate_boxes(network_numpy, img_shape[0], img_shape[1])

            for layer_id in layers:
                if layer_id not in captures:
                    raise RuntimeError(f"{model_id}: hook captured no {layer_id}")
                feature = _feature_to_chw(
                    captures[layer_id],
                    feature_layout,
                    feature_output,
                    pad_shape,
                    patch_size,
                    drop_tokens,
                )
                valid_feature, valid_hw = _crop_valid_feature(
                    feature, img_shape, pad_shape
                )
                feature_boxes = project_boxes_to_feature(
                    network_numpy, img_shape, valid_hw
                )
                feature_tensor = valid_feature.unsqueeze(0)
                boxes_tensor = torch.as_tensor(
                    feature_boxes, dtype=feature_tensor.dtype, device=feature_tensor.device
                )
                roi = roi_align(
                    feature_tensor,
                    [boxes_tensor],
                    output_size=output_size,
                    spatial_scale=1.0,
                    sampling_ratio=sampling_ratio,
                    aligned=aligned,
                )
                pooled = roi.mean(dim=(2, 3)).detach().cpu().numpy()
                if pooled.shape != (len(image_rows), valid_feature.shape[0]):
                    raise RuntimeError(
                        f"{model_id}/{layer_id}: invalid ROI output {pooled.shape}"
                    )
                if not np.isfinite(pooled).all():
                    raise RuntimeError(f"{model_id}/{layer_id}: non-finite ROI feature")
                for local_index, (manifest_index, row) in enumerate(image_rows):
                    per_layer[layer_id][manifest_index] = pooled[local_index]
                    projected = feature_boxes[local_index]
                    # network boxes are authoritative after resize; compare those.
                    network_projection_error = roundtrip_error(
                        np.asarray([network_numpy[local_index]], dtype=np.float64),
                        img_shape,
                        valid_hw,
                    )
                    mapping_rows.append(
                        {
                            "model_id": model_id,
                            "layer": layer_id,
                            "instance_index": manifest_index,
                            "instance_id": row["instance_id"],
                            "image_id": image_id,
                            "split": split,
                            "img_height": img_shape[0],
                            "img_width": img_shape[1],
                            "pad_height": pad_shape[0],
                            "pad_width": pad_shape[1],
                            "feature_height": int(feature.shape[-2]),
                            "feature_width": int(feature.shape[-1]),
                            "valid_feature_height": valid_hw[0],
                            "valid_feature_width": valid_hw[1],
                            "bbox_network": json.dumps(
                                network_numpy[local_index].tolist()
                            ),
                            "bbox_feature": json.dumps(projected.tolist()),
                            "roundtrip_max_error": float(
                                network_projection_error.max()
                            ),
                            "roi_rows": output_size[0],
                            "roi_columns": output_size[1],
                            "roi_mean_abs": float(np.abs(pooled[local_index]).mean()),
                            "network_box_clip_overshoot": network_clip_overshoot,
                        }
                    )
                    if manifest_index in debug_rows:
                        heatmap = (
                            valid_feature.detach()
                            .abs()
                            .mean(dim=0)
                            .float()
                            .cpu()
                            .numpy()
                        )
                        render_mapping_debug(
                            image_path=image_path,
                            output_path=(
                                out_dir
                                / "qa"
                                / model_id
                                / layer_id
                                / f"{manifest_index:05d}.png"
                            ),
                            original_box=row["bbox_xyxy"],
                            original_hw=(
                                int(row["image_height"]),
                                int(row["image_width"]),
                            ),
                            network_box=network_numpy[local_index],
                            img_hw=img_shape,
                            pad_hw=pad_shape,
                            feature_box=projected,
                            feature_heatmap=heatmap,
                        )
            print(
                f"[{model_id}] {image_position + 1}/{len(groups)} "
                f"{split}:{image_id}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
        loaded.close()

    bundle_dir = out_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for layer_id, values in per_layer.items():
        if any(value is None for value in values):
            raise RuntimeError(f"{model_id}/{layer_id}: incomplete feature matrix")
        matrix = np.stack(values).astype(np.float32)
        save_feature_bundle(
            bundle_dir / f"{model_id}__{layer_id}.npz",
            features=matrix,
            labels=np.asarray([row["category_id"] for row in manifest_rows]),
            instance_ids=[str(row["instance_id"]) for row in manifest_rows],
            image_ids=np.asarray([row["image_id"] for row in manifest_rows]),
            group_ids=[
                f"{row['split']}:{row['image_id']}" for row in manifest_rows
            ],
            bboxes=np.asarray(
                [row["bbox_xyxy"] for row in manifest_rows], dtype=np.float32
            ),
            split=[str(row["split"]) for row in manifest_rows],
            metadata={
                **model_audit,
                "layer": layer_id,
                "feature_shape": list(matrix.shape),
                "manifest": str(manifest_path),
            },
        )
    _write_mapping_rows(out_dir / "qa" / f"{model_id}.mapping.tsv", mapping_rows)
    write_json(out_dir / "qa" / f"{model_id}.audit.json", model_audit)


def main() -> None:
    args = parse_args()
    config = resolve_model_paths(load_config(args.config))
    manifest_path = existing_file(args.manifest)
    rows = read_jsonl(manifest_path)
    for row in rows:
        existing_file(row["image_path"])
        validate_boxes(
            np.asarray([row["bbox_xyxy"]], dtype=np.float64),
            int(row["image_height"]),
            int(row["image_width"]),
        )
    manifest_sha = file_sha256(manifest_path)
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    pipeline_by_config: Dict[str, List[Dict[str, Any]]] = {}
    roi_options = config.get("roi_align", {})
    device = args.device or str(config.get("device", "cuda:0"))
    models = _selected_models(config, args.models)
    for index, spec in enumerate(models):
        config_path = str(spec["config"])
        if config_path not in pipeline_by_config:
            pipeline_by_config[config_path] = _prepare_pipeline(config_path)
        extract_model(
            spec=spec,
            manifest_rows=rows,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            pipeline=pipeline_by_config[config_path],
            out_dir=out_dir,
            device=device,
            roi_options=roi_options,
            debug_samples=args.debug_samples,
        )
        print(
            f"Finished model {index + 1}/{len(models)}: {spec['id']}", flush=True
        )
    write_json(
        out_dir / "run.json",
        {
            "config": str(existing_file(args.config)),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "models": [str(model["id"]) for model in models],
        },
    )


if __name__ == "__main__":
    main()
