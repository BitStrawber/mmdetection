#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute layer-wise parameter update magnitude and pairwise cosine similarity."
    )
    parser.add_argument("--architecture", required=True, choices=("resnet50", "vit_small"))
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat as NAME=/path/to/backbone.pth.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def parse_models(values: Sequence[str]) -> Dict[str, Path]:
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model must use NAME=PATH syntax: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or name in models:
            raise ValueError(f"Invalid or duplicate model name: {name}")
        models[name] = Path(path).expanduser().resolve()
    if len(models) < 2:
        raise ValueError("At least two comparison models are required")
    return models


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch
    import torch.nn.functional as F

    from tools.exp_2.analyze_pretraining_parameter_updates import (
        flatten_group,
        load_state,
        selected_keys,
        validate_common_state,
    )

    args = parse_args()
    model_paths = parse_models(args.model)
    init_path = Path(args.init_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    states = {"init": load_state(init_path)}
    for name, path in model_paths.items():
        states[name] = load_state(path)
    validate_common_state(args.architecture, states)

    groups = (
        ("Stage 0", "Stage 1", "Stage 2", "Stage 3")
        if args.architecture == "resnet50"
        else ("Block 1", "Block 4", "Block 8", "Block 12")
    )
    magnitude_rows: List[Dict[str, Any]] = []
    similarity_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "architecture": args.architecture,
        "init_checkpoint": str(init_path),
        "models": {name: str(path) for name, path in model_paths.items()},
        "groups": {},
    }

    for group in groups:
        keys = selected_keys(args.architecture, group, states["init"])
        vectors = flatten_group(args.architecture, group, states)
        init_vector = vectors["init"]
        init_norm = torch.linalg.vector_norm(init_vector)
        updates = {
            name: vectors[name] - init_vector for name in model_paths
        }
        magnitudes = {
            name: (torch.linalg.vector_norm(update) / (init_norm + args.eps)).item()
            for name, update in updates.items()
        }
        similarities = {}
        for left, right in itertools.combinations(model_paths, 2):
            pair = f"{left}-{right}"
            similarities[pair] = F.cosine_similarity(
                updates[left], updates[right], dim=0, eps=args.eps
            ).item()
            similarity_rows.append(
                {
                    "architecture": args.architecture,
                    "group": group,
                    "source_a": left,
                    "source_b": right,
                    "pair": pair,
                    "cosine_similarity": similarities[pair],
                    "parameter_count": int(init_vector.numel()),
                }
            )
        for name, value in magnitudes.items():
            magnitude_rows.append(
                {
                    "architecture": args.architecture,
                    "group": group,
                    "source": name,
                    "update_magnitude": value,
                    "parameter_count": int(init_vector.numel()),
                }
            )
        report["groups"][group] = {
            "parameter_keys": keys,
            "parameter_count": int(init_vector.numel()),
            "update_magnitude": magnitudes,
            "cosine_similarity": similarities,
        }
        print(
            f"{group}: parameters={init_vector.numel()} "
            f"magnitude={magnitudes} similarity={similarities}"
        )

    write_csv(
        output / "update_magnitude.csv",
        magnitude_rows,
        (
            "architecture",
            "group",
            "source",
            "update_magnitude",
            "parameter_count",
        ),
    )
    write_csv(
        output / "update_similarity.csv",
        similarity_rows,
        (
            "architecture",
            "group",
            "source_a",
            "source_b",
            "pair",
            "cosine_similarity",
            "parameter_count",
        ),
    )
    (output / "parameter_update_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={output}")
    print("RESULT=PASS_ADDITIONAL_BACKBONE_UPDATE_ANALYSIS")


if __name__ == "__main__":
    main()
