from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .common import ensure_empty_or_create, existing_file, read_json, write_json
from .config import load_config
from .feature_io import assert_aligned_bundles, load_feature_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Fisher ratios, linear probes, and paired bootstrap CIs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("At least one feature has zero or near-zero norm")
    return values / norms


def fisher_discriminant_ratio(
    features: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("Features and labels have incompatible shapes")
    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or Inf")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    normalized = l2_normalize(values)
    centers = np.stack(
        [normalized[labels == category].mean(axis=0) for category in classes]
    )
    intra_terms = []
    for center, category in zip(centers, classes):
        residual = normalized[labels == category] - center
        intra_terms.append(float(np.mean(np.sum(residual * residual, axis=1))))
    intra = float(np.mean(intra_terms))
    pair_terms = []
    for left in range(len(classes)):
        for right in range(left + 1, len(classes)):
            delta = centers[left] - centers[right]
            pair_terms.append(float(np.dot(delta, delta)))
    inter = float(np.mean(pair_terms))
    return {
        "d_intra": intra,
        "d_inter": inter,
        "fdr": inter / (intra + eps),
    }


def _main_models(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    main_layers = config.get("analysis", {}).get("main_layers", {})
    models = []
    for spec in config["models"]:
        backbone = str(spec["backbone"])
        layer = str(spec.get("main_layer") or main_layers.get(backbone, ""))
        if not layer:
            raise ValueError(f"No main analysis layer for backbone {backbone}")
        models.append({**spec, "analysis_layer": layer})
    return models


def _linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classifier_options: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(labels)
    min_group_count = len(np.unique(groups))
    requested_splits = int(classifier_options.get("folds", 5))
    folds = min(requested_splits, min_group_count)
    if folds < 2:
        raise ValueError("At least two image groups are required for cross-validation")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    classifier = LogisticRegression(
        C=float(classifier_options.get("C", 1.0)),
        max_iter=int(classifier_options.get("max_iter", 5000)),
        solver=str(classifier_options.get("solver", "lbfgs")),
        random_state=seed,
    )
    pipeline = make_pipeline(StandardScaler(), classifier)
    predictions = cross_val_predict(
        pipeline,
        features,
        labels,
        groups=groups,
        cv=splitter,
        method="predict",
        n_jobs=int(classifier_options.get("n_jobs", 1)),
    )
    return {
        "predictions": np.asarray(predictions, dtype=np.int64),
        "classes": classes,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes),
        "folds": folds,
    }


def _bootstrap_indices(
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    selected_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
    return np.concatenate([members[index] for index in selected_groups])


def _bootstrap_estimates(
    normalized: Mapping[str, np.ndarray],
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    result = {
        model: {"fdr": [], "accuracy": [], "macro_f1": []}
        for model in normalized
    }
    for _ in range(samples):
        indices = _bootstrap_indices(groups, rng)
        sampled_labels = labels[indices]
        for model in normalized:
            values = normalized[model][indices]
            fdr = fisher_discriminant_ratio(values, sampled_labels)["fdr"]
            predicted = predictions[model][indices]
            result[model]["fdr"].append(fdr)
            result[model]["accuracy"].append(
                float(accuracy_score(sampled_labels, predicted))
            )
            result[model]["macro_f1"].append(
                float(
                    f1_score(
                        sampled_labels,
                        predicted,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
    return {
        model: {name: np.asarray(values) for name, values in metrics.items()}
        for model, metrics in result.items()
    }


def _interval(values: np.ndarray) -> List[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "model_id",
        "backbone",
        "source",
        "layer",
        "instances",
        "images",
        "d_intra",
        "d_inter",
        "fdr",
        "fdr_ci_low",
        "fdr_ci_high",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "macro_f1",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_rows(
    config: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows = []
    by_key = {
        (str(model["backbone"]), str(model["source"])): str(model["id"])
        for model in models
    }
    for backbone in sorted({str(model["backbone"]) for model in models}):
        sources = sorted(
            str(model["source"])
            for model in models
            if str(model["backbone"]) == backbone
        )
        for left_index, left in enumerate(sources):
            for right in sources[left_index + 1 :]:
                model_a = by_key[(backbone, left)]
                model_b = by_key[(backbone, right)]
                for metric in ("fdr", "accuracy", "macro_f1"):
                    difference = bootstrap[model_a][metric] - bootstrap[model_b][metric]
                    interval = _interval(difference)
                    rows.append(
                        {
                            "backbone": backbone,
                            "source_a": left,
                            "source_b": right,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric,
                            "difference": float(np.mean(difference)),
                            "ci_low": interval[0],
                            "ci_high": interval[1],
                        }
                    )
    return rows


def _write_pairwise(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "backbone",
        "source_a",
        "source_b",
        "model_a",
        "model_b",
        "metric",
        "difference",
        "ci_low",
        "ci_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    feature_root = Path(args.features_dir).expanduser().resolve()
    models = _main_models(config)
    bundles = {}
    for model in models:
        path = feature_root / f"{model['id']}__{model['analysis_layer']}.npz"
        bundles[str(model["id"])] = load_feature_bundle(
            existing_file(path), verify_array_hashes=True
        )
    assert_aligned_bundles(bundles)
    reference = bundles[str(models[0]["id"])]
    labels = np.asarray(reference["labels"], dtype=np.int64)
    groups = np.asarray(reference["group_ids"], dtype=str)
    normalized = {
        model_id: l2_normalize(bundle["features"])
        for model_id, bundle in bundles.items()
    }
    classifier_options = config.get("analysis", {}).get("linear_probe", {})
    probe = {
        model_id: _linear_probe(
            bundle["features"],
            labels,
            groups,
            classifier_options,
            seed=int(config.get("seed", 42)),
        )
        for model_id, bundle in bundles.items()
    }
    bootstrap_samples = max(int(args.bootstrap_samples), 0)
    bootstrap = {}
    if bootstrap_samples:
        bootstrap = _bootstrap_estimates(
            normalized=normalized,
            labels=labels,
            predictions={
                model_id: probe[model_id]["predictions"] for model_id in probe
            },
            groups=groups,
            samples=bootstrap_samples,
            seed=int(config.get("seed", 42)),
        )

    out_dir = ensure_empty_or_create(Path(args.out_dir), args.overwrite)
    summary_rows = []
    for model in models:
        model_id = str(model["id"])
        fdr = fisher_discriminant_ratio(
            bundles[model_id]["features"], labels
        )
        row = {
            **fdr,
            "model_id": model_id,
            "backbone": model["backbone"],
            "source": model["source"],
            "layer": model["analysis_layer"],
            "instances": len(labels),
            "images": len(np.unique(groups)),
            "accuracy": probe[model_id]["accuracy"],
            "macro_f1": probe[model_id]["macro_f1"],
            "fdr_ci_low": "",
            "fdr_ci_high": "",
            "accuracy_ci_low": "",
            "accuracy_ci_high": "",
            "macro_f1_ci_low": "",
            "macro_f1_ci_high": "",
        }
        if bootstrap:
            for metric in ("fdr", "accuracy", "macro_f1"):
                low, high = _interval(bootstrap[model_id][metric])
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    _write_summary(out_dir / "metrics_summary.tsv", summary_rows)
    pairwise = _pairwise_rows(config, models, bootstrap) if bootstrap else []
    _write_pairwise(out_dir / "pairwise_comparisons.tsv", pairwise)

    detailed = {
        "config": str(existing_file(args.config)),
        "bootstrap_samples": bootstrap_samples,
        "models": {
            model_id: {
                "accuracy": probe[model_id]["accuracy"],
                "macro_f1": probe[model_id]["macro_f1"],
                "folds": probe[model_id]["folds"],
                "classification_report": probe[model_id]["classification_report"],
                "confusion_matrix": probe[model_id]["confusion_matrix"].tolist(),
            }
            for model_id in probe
        },
        "pairwise_comparisons": pairwise,
    }
    write_json(out_dir / "metrics_details.json", detailed)
    if bootstrap:
        np.savez_compressed(
            out_dir / "bootstrap_estimates.npz",
            **{
                f"{model_id}__{metric}": values
                for model_id, metrics in bootstrap.items()
                for metric, values in metrics.items()
            },
        )
    print(f"Metrics: {out_dir / 'metrics_summary.tsv'}")
    print(f"Pairwise comparisons: {out_dir / 'pairwise_comparisons.tsv'}")


if __name__ == "__main__":
    main()
