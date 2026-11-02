from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .common import file_sha256, read_json, write_json


REQUIRED_ARRAYS = (
    "features",
    "labels",
    "instance_ids",
    "image_ids",
    "group_ids",
    "bboxes",
    "split",
)


def save_feature_bundle(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    instance_ids: Sequence[str],
    image_ids: np.ndarray,
    group_ids: Sequence[str],
    bboxes: np.ndarray,
    split: Sequence[str],
    metadata: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        instance_ids=np.asarray(instance_ids, dtype=str),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=str),
        bboxes=np.asarray(bboxes, dtype=np.float32),
        split=np.asarray(split, dtype=str),
    )
    payload = dict(metadata)
    payload["bundle_sha256"] = file_sha256(output)
    write_json(output.with_suffix(".metadata.json"), payload)
    return output


def load_feature_bundle(
    path: str | Path,
    verify_array_hashes: bool = False,
) -> Dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as bundle:
        missing = [name for name in REQUIRED_ARRAYS if name not in bundle]
        if missing:
            raise KeyError(f"{source} is missing arrays: {missing}")
        result = {name: bundle[name] for name in REQUIRED_ARRAYS}
    metadata_path = source.with_suffix(".metadata.json")
    if metadata_path.is_file():
        result["metadata"] = read_json(metadata_path)
        if verify_array_hashes:
            expected = result["metadata"].get("bundle_sha256")
            actual = file_sha256(source)
            if expected and expected != actual:
                raise ValueError(f"Bundle hash mismatch: {source}")
    validate_feature_bundle(result, source)
    return result


def validate_feature_bundle(bundle: Mapping[str, Any], source: Any = "<memory>") -> None:
    features = np.asarray(bundle["features"])
    labels = np.asarray(bundle["labels"])
    instance_ids = np.asarray(bundle["instance_ids"])
    image_ids = np.asarray(bundle["image_ids"])
    group_ids = np.asarray(bundle["group_ids"])
    bboxes = np.asarray(bundle["bboxes"])
    split = np.asarray(bundle["split"])
    if features.ndim != 2:
        raise ValueError(f"{source}: features must be N x D, got {features.shape}")
    count = features.shape[0]
    if not np.isfinite(features).all():
        raise ValueError(f"{source}: features contain NaN or Inf")
    for name, values in (
        ("labels", labels),
        ("instance_ids", instance_ids),
        ("image_ids", image_ids),
        ("group_ids", group_ids),
        ("bboxes", bboxes),
        ("split", split),
    ):
        if len(values) != count:
            raise ValueError(
                f"{source}: {name} has {len(values)} rows, expected {count}"
            )
    if bboxes.shape != (count, 4):
        raise ValueError(f"{source}: bboxes must be N x 4, got {bboxes.shape}")
    if len(set(instance_ids.tolist())) != count:
        raise ValueError(f"{source}: duplicate instance_ids")
    if not np.isfinite(bboxes).all():
        raise ValueError(f"{source}: bboxes contain NaN or Inf")


def assert_aligned_bundles(bundles: Mapping[str, Mapping[str, Any]]) -> None:
    if not bundles:
        raise ValueError("No feature bundles were supplied")
    reference_name = next(iter(bundles))
    reference = bundles[reference_name]
    fields = ("instance_ids", "image_ids", "group_ids", "labels", "bboxes", "split")
    for name, bundle in bundles.items():
        for field in fields:
            if not np.array_equal(reference[field], bundle[field]):
                raise ValueError(
                    f"{name}: {field} differs from reference {reference_name}"
                )
