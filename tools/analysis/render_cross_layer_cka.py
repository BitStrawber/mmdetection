#!/usr/bin/env python3
"""Compute and render a 4x16 cross-layer CKA heatmap.

The output orientation is fixed for the requested paper figure:

* rows: four layers of the RUOD-trained reference model;
* columns: four layers from each pretrained comparison model.

An existing full matrix can be supplied with ``--matrix``. Otherwise, the
script computes linear CKA from the pooled feature arrays referenced by the
metadata file. It never infers cross-layer values from same-layer summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DISPLAY_NAMES = {
    "imagenet_dino100e": "ImageNet",
    "realuw_dino100e": "RealUW",
    "synthetic5_dino100e": "Synthetic5",
    "imagenet_dino100e_dfui": "ImageNet + DFUI",
    "imagenet_dino100e_backbone": "ImageNet",
    "realuw_dino100e_backbone": "RealUW",
    "synthetic5_dino100e_backbone": "Synthetic5",
    "imagenet_dino100e_dfui_backbone": "ImageNet + DFUI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="cross_layer_cka_4x16")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = ("y_axis_rows", "x_axis_model", "variant")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"metadata is missing required keys: {missing}")
    return metadata


def ordered_models(metadata: dict) -> list[str]:
    models: list[str] = []
    for row in metadata["y_axis_rows"]:
        model = row["model"]
        if model not in models:
            models.append(model)
    if len(models) != 4:
        raise ValueError(f"expected four pretrained models, found {len(models)}: {models}")
    return models


def ordered_layers(metadata: dict) -> list[str]:
    layers = list(metadata.get("layers_b") or metadata.get("layers_a") or [])
    if not layers:
        layers = []
        for row in metadata["y_axis_rows"]:
            layer = row["layer"]
            if layer not in layers:
                layers.append(layer)
    if len(layers) != 4:
        raise ValueError(f"expected four layers, found {len(layers)}: {layers}")
    return layers


def feature_path(root: Path, model: str, variant: str, layer: str) -> Path:
    candidates = (
        root / "features" / model / variant / f"{layer}.npy",
        root / model / variant / f"{layer}.npy",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "feature array not found; checked:\n  " + "\n  ".join(str(path) for path in candidates)
    )


def load_feature(path: Path, expected_samples: int | None) -> np.ndarray:
    feature = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    if feature.ndim != 2:
        raise ValueError(f"expected a 2D pooled feature array, got {feature.shape}: {path}")
    if expected_samples is not None and feature.shape[0] != expected_samples:
        raise ValueError(
            f"expected {expected_samples} samples, got {feature.shape[0]}: {path}"
        )
    if not np.isfinite(feature).all():
        raise ValueError(f"feature array contains NaN or Inf: {path}")
    return feature


def linear_cka(feature_x: np.ndarray, feature_y: np.ndarray) -> float:
    if feature_x.shape[0] != feature_y.shape[0]:
        raise ValueError(f"sample count mismatch: {feature_x.shape} vs {feature_y.shape}")
    x = feature_x - feature_x.mean(axis=0, keepdims=True)
    y = feature_y - feature_y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = float(np.sum(cross * cross))
    denominator = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if denominator <= 0.0:
        raise ValueError("linear CKA denominator is zero")
    return numerator / denominator


def compute_matrix(metadata: dict, feature_root: Path) -> tuple[np.ndarray, list[str], list[str]]:
    models = ordered_models(metadata)
    layers = ordered_layers(metadata)
    reference_model = metadata["x_axis_model"]
    variant = metadata["variant"]
    samples = metadata.get("samples")

    reference_features = {
        layer: load_feature(feature_path(feature_root, reference_model, variant, layer), samples)
        for layer in layers
    }
    comparison_features = {
        (model, layer): load_feature(feature_path(feature_root, model, variant, layer), samples)
        for model in models
        for layer in layers
    }

    matrix = np.empty((len(layers), len(models) * len(layers)), dtype=np.float64)
    for row_index, reference_layer in enumerate(layers):
        for model_index, model in enumerate(models):
            for layer_index, comparison_layer in enumerate(layers):
                column_index = model_index * len(layers) + layer_index
                matrix[row_index, column_index] = linear_cka(
                    reference_features[reference_layer],
                    comparison_features[(model, comparison_layer)],
                )
    return matrix, models, layers


def load_matrix(path: Path, models: Sequence[str], layers: Sequence[str]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.asarray(np.load(path), dtype=np.float64)
    elif path.suffix.lower() in {".tsv", ".csv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
        numeric_rows: list[list[float]] = []
        for row in rows:
            try:
                numeric_rows.append([float(value) for value in row])
            except ValueError:
                continue
        matrix = np.asarray(numeric_rows, dtype=np.float64)
    else:
        raise ValueError(f"unsupported matrix format: {path}")

    expected = (len(layers), len(models) * len(layers))
    transposed = (expected[1], expected[0])
    if matrix.shape == transposed:
        matrix = matrix.T
    if matrix.shape != expected:
        raise ValueError(f"expected matrix shape {expected} or {transposed}, got {matrix.shape}")
    return matrix


def write_tsv(
    path: Path,
    matrix: np.ndarray,
    models: Sequence[str],
    layers: Sequence[str],
    reference_model: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["reference_model", "reference_layer", "comparison_model", "comparison_layer", "linear_cka"]
        )
        for row_index, reference_layer in enumerate(layers):
            for model_index, model in enumerate(models):
                for layer_index, comparison_layer in enumerate(layers):
                    column_index = model_index * len(layers) + layer_index
                    writer.writerow(
                        [
                            reference_model,
                            reference_layer,
                            model,
                            comparison_layer,
                            f"{matrix[row_index, column_index]:.12g}",
                        ]
                    )


def render(
    matrix: np.ndarray,
    models: Sequence[str],
    layers: Sequence[str],
    out_stem: Path,
    vmin: float,
    vmax: float,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    color_map = LinearSegmentedColormap.from_list(
        "reference_blue_white_red",
        ("#3F57B3", "#91A5DE", "#F0EBE4", "#D18C7A", "#A63B31"),
        N=256,
    )
    figure, axis = plt.subplots(figsize=(11.2, 3.65))
    image = axis.imshow(
        matrix,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    layer_labels = [layer.upper() for _model in models for layer in layers]
    axis.set_xticks(np.arange(len(layer_labels)), layer_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(layers)), [layer.upper() for layer in layers])
    axis.set_xlabel("Pretrained model layer")
    axis.set_ylabel("ImageNet-to-RUOD reference layer")

    for model_index, model in enumerate(models):
        center = model_index * len(layers) + (len(layers) - 1) / 2
        axis.text(
            center,
            1.045,
            DISPLAY_NAMES.get(model, model),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
        if model_index:
            axis.axvline(model_index * len(layers) - 0.5, color="white", linewidth=2.0)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.25 or value > 0.78 else "black",
                fontsize=5.6,
            )

    color_bar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    color_bar.set_label("Linear CKA")
    color_bar.set_ticks(np.linspace(vmin, vmax, 6))
    figure.subplots_adjust(left=0.10, right=0.94, bottom=0.23, top=0.83)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out_stem.with_suffix(f".{suffix}"), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    models = ordered_models(metadata)
    layers = ordered_layers(metadata)
    feature_root = args.feature_root or Path(metadata["feature_root"])

    if args.matrix:
        matrix = load_matrix(args.matrix, models, layers)
        source = str(args.matrix)
    else:
        matrix, models, layers = compute_matrix(metadata, feature_root)
        source = str(feature_root)

    if not np.isfinite(matrix).all():
        raise ValueError("CKA matrix contains NaN or Inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = args.out_dir / args.stem
    np.save(out_stem.with_suffix(".npy"), matrix)
    write_tsv(
        out_stem.with_suffix(".tsv"),
        matrix,
        models,
        layers,
        metadata["x_axis_model"],
    )
    render(matrix, models, layers, out_stem, args.vmin, args.vmax)
    with out_stem.with_suffix(".metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "shape": list(matrix.shape),
                "orientation": "rows=RUOD reference layers; columns=pretrained model layers",
                "reference_model": metadata["x_axis_model"],
                "comparison_models": models,
                "layers": layers,
                "variant": metadata["variant"],
                "samples": metadata.get("samples"),
                "method": "linear CKA on column-centered pooled features",
                "source": source,
            },
            handle,
            indent=2,
        )
    print(f"Cross-layer CKA matrix: {matrix.shape}")
    print(f"Outputs: {out_stem}")


if __name__ == "__main__":
    main()
