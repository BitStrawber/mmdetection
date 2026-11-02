from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .common import ensure_empty_or_create, existing_file, read_json
from .config import load_config
from .feature_io import assert_aligned_bundles, load_feature_bundle
from .metrics import l2_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render object-level t-SNE panels from saved feature bundles."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _main_models(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    layers = config.get("analysis", {}).get("main_layers", {})
    output = []
    for model in config["models"]:
        backbone = str(model["backbone"])
        layer = str(model.get("main_layer") or layers.get(backbone, ""))
        if not layer:
            raise ValueError(f"No main layer for backbone: {backbone}")
        output.append({**model, "analysis_layer": layer})
    return output


def _read_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {
            row["model_id"]: {
                "fdr": float(row["fdr"]),
                "fdr_ci_low": float(row["fdr_ci_low"])
                if row["fdr_ci_low"]
                else float("nan"),
                "fdr_ci_high": float(row["fdr_ci_high"])
                if row["fdr_ci_high"]
                else float("nan"),
                "macro_f1": float(row["macro_f1"]),
            }
            for row in rows
        }


def _project(
    features: np.ndarray,
    seed: int,
    tsne_options: Mapping[str, Any],
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    joined = l2_normalize(features)
    pca_dim = min(
        int(tsne_options.get("pca_dimensions", 50)),
        joined.shape[0] - 1,
        joined.shape[1],
    )
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(joined)
    requested_perplexity = float(tsne_options.get("perplexity", 30.0))
    perplexity = min(requested_perplexity, max(1.0, (len(reduced) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=str(tsne_options.get("learning_rate", "auto")),
        init=str(tsne_options.get("init", "pca")),
        max_iter=int(tsne_options.get("max_iter", 2000)),
        random_state=seed,
        method=str(tsne_options.get("method", "barnes_hut")),
        angle=float(tsne_options.get("angle", 0.5)),
    ).fit_transform(reduced)


def _write_coordinates(
    path: Path,
    model: Mapping[str, Any],
    coordinates: np.ndarray,
    bundle: Mapping[str, Any],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "model_id",
                "backbone",
                "source",
                "layer",
                "instance_id",
                "group_id",
                "label",
                "x",
                "y",
            ]
        )
        for index, (x, y) in enumerate(coordinates):
            writer.writerow(
                [
                    model["id"],
                    model["backbone"],
                    model["source"],
                    model["analysis_layer"],
                    bundle["instance_ids"][index],
                    bundle["group_ids"][index],
                    int(bundle["labels"][index]),
                    float(x),
                    float(y),
                ]
            )


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(args.config)
    models = _main_models(config)
    feature_root = Path(args.features_dir).expanduser().resolve()
    metrics_root = Path(args.metrics_dir).expanduser().resolve()
    bundles = {}
    for model in models:
        path = feature_root / f"{model['id']}__{model['analysis_layer']}.npz"
        bundles[str(model["id"])] = load_feature_bundle(
            existing_file(path), verify_array_hashes=True
        )
    assert_aligned_bundles(bundles)
    metrics = _read_metrics(metrics_root / "metrics_summary.tsv")
    categories_path = config.get("categories")
    category_names = {}
    if categories_path:
        category_names = {
            int(row["id"]): str(row["name"])
            for row in read_json(existing_file(categories_path))
        }

    backbones = list(dict.fromkeys(str(model["backbone"]) for model in models))
    sources = list(dict.fromkeys(str(model["source"]) for model in models))
    lookup = {
        (str(model["backbone"]), str(model["source"])): model for model in models
    }
    if len(backbones) * len(sources) != len(models):
        raise ValueError("The model matrix must cover every backbone/source pair")
    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    coordinates_dir = out_dir / "coordinates"
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    tsne_options = config.get("analysis", {}).get("tsne", {})
    coordinates = {}
    for model in models:
        model_id = str(model["id"])
        values = _project(
            bundles[model_id]["features"], seed=seed, tsne_options=tsne_options
        )
        coordinates[model_id] = values
        _write_coordinates(
            coordinates_dir / f"{model_id}.tsv",
            model,
            values,
            bundles[model_id],
        )

    fig, axes = plt.subplots(
        len(backbones),
        len(sources),
        figsize=(
            float(config.get("figure", {}).get("width", 12.0)),
            float(config.get("figure", {}).get("height", 7.2)),
        ),
        squeeze=False,
        constrained_layout=True,
    )
    labels = np.asarray(next(iter(bundles.values()))["labels"], dtype=np.int64)
    classes = sorted(np.unique(labels).tolist())
    color_map = plt.get_cmap("tab10")
    colors = {category: color_map(index % 10) for index, category in enumerate(classes)}
    for row_index, backbone in enumerate(backbones):
        for column_index, source in enumerate(sources):
            axis = axes[row_index, column_index]
            model = lookup[(backbone, source)]
            model_id = str(model["id"])
            points = coordinates[model_id]
            for category in classes:
                mask = labels == category
                label = category_names.get(category, str(category))
                axis.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    s=float(config.get("figure", {}).get("point_size", 7.0)),
                    alpha=float(config.get("figure", {}).get("point_alpha", 0.72)),
                    c=[colors[category]],
                    label=label,
                    linewidths=0,
                )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{backbone} / {source}", fontsize=10)
            metric = metrics[model_id]
            fdr_text = f"FDR={metric['fdr']:.3f}"
            if np.isfinite(metric["fdr_ci_low"]):
                fdr_text += (
                    f" [{metric['fdr_ci_low']:.3f}, "
                    f"{metric['fdr_ci_high']:.3f}]"
                )
            axis.text(
                0.0,
                -0.05,
                f"{fdr_text}   Macro-F1={metric['macro_f1']:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#9AA0A6")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4,
            color=colors[category],
            label=category_names.get(category, str(category)),
        )
        for category in classes
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(classes), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    stem = out_dir / str(config.get("figure", {}).get("stem", "object_feature_tsne"))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=int(config.get("figure", {}).get("dpi", 300)),
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Figure: {stem}.png")


if __name__ == "__main__":
    main()
