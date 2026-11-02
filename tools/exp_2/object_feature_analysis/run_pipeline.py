from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete object feature analysis pipeline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--debug-samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run(command: List[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir).expanduser().resolve()
    manifest_dir = root / "manifest"
    feature_dir = root / "features"
    metrics_dir = root / "metrics"
    figure_dir = root / "figures"
    overwrite = ["--overwrite"] if args.overwrite else []
    module = [
        sys.executable,
        "-m",
        "tools.exp_2.object_feature_analysis",
    ]
    _run(
        module
        + [
            "sample-instances",
            "--config",
            args.config,
            "--out-dir",
            str(manifest_dir),
        ]
        + overwrite
    )
    extract_command = module + [
        "extract-roi",
        "--config",
        args.config,
        "--manifest",
        str(manifest_dir / "instances.jsonl"),
        "--out-dir",
        str(feature_dir),
        "--debug-samples",
        str(args.debug_samples),
    ]
    if args.models:
        extract_command.extend(["--models", args.models])
    _run(extract_command + overwrite)
    _run(
        module
        + [
            "compute-metrics",
            "--config",
            args.config,
            "--features-dir",
            str(feature_dir / "bundles"),
            "--out-dir",
            str(metrics_dir),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ]
        + overwrite
    )
    _run(
        module
        + [
            "render-figures",
            "--config",
            args.config,
            "--features-dir",
            str(feature_dir / "bundles"),
            "--metrics-dir",
            str(metrics_dir),
            "--out-dir",
            str(figure_dir),
        ]
        + overwrite
    )
    print(f"Completed pipeline: {root}")


if __name__ == "__main__":
    main()
