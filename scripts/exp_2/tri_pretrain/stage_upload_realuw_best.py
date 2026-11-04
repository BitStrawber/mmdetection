#!/usr/bin/env python3
"""Stage and upload the curated RealUW random/task result snapshot.

The script is intentionally profile-based because the source checkpoints are
split between the ``fuping`` and ``fcp`` Unix accounts.  It selects only the
best stage checkpoints, copies the adjacent config/log/provenance files, writes
machine-readable metrics, and optionally uploads the staged tree to Hugging
Face.

Run ``fuping`` first, then run ``fcp`` with ``--upload --finalize``.  The fcp
step downloads the fuping partial manifest, merges both manifests, and verifies
that exactly 48 best checkpoints exist in the remote snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_REPO_ID = "BitStrawber/Output"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_REMOTE_ROOT = "realuw_random_task_best"
DEFAULT_STAGING_BASE = Path("/media/HDD2/XCX/exp_2/hf_staging")

EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")
TEST_METRIC_RE = re.compile(
    r"(bbox|segm)_mAP_copypaste:\s*"
    r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
)
SEGM_VAL_METRIC_RE = re.compile(
    r"coco/segm_mAP:\s*(\d+\.\d+).*?"
    r"coco/segm_mAP_50:\s*(\d+\.\d+).*?"
    r"coco/segm_mAP_75:\s*(\d+\.\d+)"
)

PROFILE_ROOTS = {
    "fuping": [
        (
            "dfui_realuw_random",
            Path(
                "/media/HDD2/XCX/exp_2/dfui_ruod_12exp_20261025/runs/"
                "dfui_realuw_random_crosssource_gpu0123_20261101"
            ),
        ),
        (
            "dfui_realuw_random",
            Path(
                "/media/HDD2/XCX/exp_2/dfui_ruod_12exp_20261025/runs/"
                "dfui_realuw_random_realuw_gpu23_retry02"
            ),
        ),
    ],
    "fcp": [
        (
            "dfui_realuw_task",
            Path(
                "/media/SSD1/XCX/exp_2/dfui_training_runs/"
                "crosshost_imagenet1k_on_fcp_uiis_20260828_114440"
            ),
        ),
        (
            "dfui_realuw_task",
            Path(
                "/media/SSD1/XCX/exp_2/dfui_training_runs/"
                "full_realuw_then_synthetic5_corrected_dfuiuiis_"
                "s1_ruod_s2_20260910_222035"
            ),
        ),
        (
            "sfui_realuw_random",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "sfui_realuw_random_imagenet_vits_r50_144e_"
                "gpu45_20261101_202735"
            ),
        ),
        (
            "sfui_realuw_random",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "sfui_realuw_random_synthetic_corrected_vits_r50_"
                "144e_gpu67_20261101_204106"
            ),
        ),
        (
            "sfui_realuw_random",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "sfui_realuw_random_vits_r50_144e_20261028_215352"
            ),
        ),
        (
            "sfui_realuw_task",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "aquaov255_thr070_full_sources_20261018_213844"
            ),
        ),
        (
            "sfui_realuw_task",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "realuw_r50_uiis_lr002_fresh_20261023_133814"
            ),
        ),
        (
            "sfui_realuw_task",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "synthetic5_r50_sfui_lr005_manual_20261023_024024"
            ),
        ),
        (
            "sfui_realuw_task",
            Path(
                "/media/HDD2/XCX/exp_2/sfui_uiis_runs/"
                "synthetic5_r50_uiis_lr002_fresh_20261023_135624"
            ),
        ),
    ],
}

EXPECTED_ROW_COUNTS = {
    "fuping": 12,
    "fcp": 36,
}

MANIFEST_FIELDS = [
    "experiment",
    "source",
    "backbone",
    "stage",
    "checkpoint_epoch",
    "metric",
    "metric_source",
    "independent_test",
    "AP",
    "AP50",
    "AP75",
    "original_checkpoint_path",
    "repo_checkpoint_path",
    "original_config_path",
    "repo_config_path",
    "original_log_paths",
    "repo_log_paths",
    "status",
    "checkpoint_sha256",
    "size_bytes",
]


@dataclass
class Artifact:
    experiment: str
    source: str
    backbone: str
    stage: str
    checkpoint_epoch: int
    checkpoint: Path
    work_dir: Path
    scope: Path
    config: Optional[Path]
    logs: List[Path]
    metric_log: Optional[Path]
    metric_source: str
    independent_test: bool
    ap: Optional[str]
    ap50: Optional[str]
    ap75: Optional[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()

    for path in paths:
        if not path.is_file():
            continue
        identity = str(path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        result.append(path)

    return result


def find_scope(work_dir: Path) -> Path:
    if work_dir.name == "work_dir":
        return work_dir.parent
    if work_dir.name == "work_dirs":
        return work_dir.parent
    if work_dir.parent.name == "work_dirs":
        return work_dir.parent.parent
    return work_dir


def infer_stage(work_dir: Path, scope: Path) -> Optional[str]:
    text = f"{work_dir.name} {scope.name}".lower()

    if "dfui48e" in text or "det48e" in text or "cascade48e" in text:
        return "DFUI"
    if "ruod24e" in text:
        return "RUOD"
    if "uiis" in text and "mask" in text:
        return "UIIS"
    if "sfui" in text and "mask" in text:
        return "SFUI"
    if "uiis_" in text:
        return "UIIS"
    if "sfui_" in text:
        return "SFUI"
    return None


def infer_source(work_dir: Path, scope: Path) -> str:
    text = f"{work_dir.name} {scope.name}".lower()

    if "imagenet" in text:
        return "ImageNet"
    if "synthetic5_merged" in text:
        return "Synthetic5-corrected"
    if "synthetic_corrected" in text:
        return "Synthetic5-corrected"
    if "synthetic5" in text:
        return "Synthetic5"
    if "realuw" in text:
        return "RealUW"
    return "UNKNOWN"


def infer_arch(work_dir: Path, scope: Path) -> str:
    text = f"{work_dir.name} {scope.name}".lower()
    if any(token in text for token in ("vits", "vit-small", "vit_small")):
        return "ViT-S"
    return "R50"


def source_slug(source: str) -> str:
    slugs = {
        "ImageNet": "imagenet",
        "RealUW": "realuw",
        "Synthetic5": "synthetic5",
        "Synthetic5-corrected": "synthetic5_corrected",
    }
    return slugs.get(source, source.lower())


def arch_slug(arch: str) -> str:
    return "vits" if arch == "ViT-S" else "r50"


def candidate_test_logs(scope: Path, work_dir: Path, stage: str) -> List[Path]:
    paths: List[Path] = []

    names: Dict[str, str] = {
        "DFUI": "dfui_test.log",
        "RUOD": "ruod_test.log",
        "UIIS": "uiis_test.log",
    }
    if stage in names:
        paths.append(scope / "logs" / names[stage])

    paths.extend(scope.rglob(f"{work_dir.name}_test.log"))
    return unique_paths(paths)


def all_scope_logs(scope: Path) -> List[Path]:
    paths = unique_paths(scope.rglob("*.log"))
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths


def parse_test_metric(
    stage: str, paths: Sequence[Path]
) -> Optional[Tuple[Path, str, str, str]]:
    expected_kind = "segm" if stage == "UIIS" else "bbox"
    candidates = []

    for path in paths:
        text = path.read_text(errors="replace")
        matches = [
            match
            for match in TEST_METRIC_RE.finditer(text)
            if match.group(1) == expected_kind
        ]
        if not matches:
            continue

        match = matches[-1]
        candidates.append(
            (
                path.stat().st_mtime,
                path,
                match.group(2),
                match.group(3),
                match.group(4),
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, path, ap, ap50, ap75 = candidates[-1]
    return path, ap, ap50, ap75


def parse_segm_val_metric(
    epoch: int, paths: Sequence[Path]
) -> Optional[Tuple[Path, str, str, str]]:
    marker = f"Epoch(val) [{epoch}]"
    candidates = []

    for path in paths:
        text = path.read_text(errors="replace")
        matches = []

        for line in text.splitlines():
            if marker not in line or "coco/segm_mAP:" not in line:
                continue
            match = SEGM_VAL_METRIC_RE.search(line)
            if match:
                matches.append(match)

        if not matches:
            continue

        match = matches[-1]
        candidates.append(
            (
                path.stat().st_mtime,
                path,
                match.group(1),
                match.group(2),
                match.group(3),
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, path, ap, ap50, ap75 = candidates[-1]
    return path, ap, ap50, ap75


def resolve_metric(
    stage: str,
    scope: Path,
    work_dir: Path,
    epoch: int,
) -> Tuple[Optional[Path], Optional[str], Optional[str], Optional[str], str, bool]:
    test_metric = parse_test_metric(
        stage, candidate_test_logs(scope, work_dir, stage)
    )

    if test_metric is not None:
        log, ap, ap50, ap75 = test_metric
        return log, ap, ap50, ap75, "test", True

    if stage in {"SFUI", "UIIS"}:
        val_metric = parse_segm_val_metric(epoch, all_scope_logs(scope))
        if val_metric is not None:
            log, ap, ap50, ap75 = val_metric
            return log, ap, ap50, ap75, "validation", False

    return None, None, None, None, "missing", False


def collect_stage_logs(
    stage: str,
    scope: Path,
    work_dir: Path,
    metric_log: Optional[Path],
) -> List[Path]:
    paths = set(path for path in work_dir.rglob("*.log") if path.is_file())

    for path in candidate_test_logs(scope, work_dir, stage):
        paths.add(path)

    if stage == "SFUI":
        paths.add(scope / "logs" / "sfui_train.log")
    if stage == "UIIS":
        paths.add(scope / "logs" / "uiis_train.log")
    if metric_log is not None:
        paths.add(metric_log)

    return unique_paths(sorted(paths))


def collect_artifacts(profile: str) -> List[Artifact]:
    artifacts: List[Artifact] = []
    seen = set()

    for experiment, root in PROFILE_ROOTS[profile]:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source root: {root}")

        for checkpoint in sorted(root.rglob("best_coco_*_mAP_epoch_*.pth")):
            checkpoint_text = str(checkpoint)

            if ".retired" in checkpoint_text or checkpoint_text in seen:
                continue

            work_dir = checkpoint.parent
            scope = find_scope(work_dir)
            stage = infer_stage(work_dir, scope)
            if stage is None:
                raise RuntimeError(f"Cannot infer stage: {checkpoint}")

            source = infer_source(work_dir, scope)
            if source == "UNKNOWN":
                raise RuntimeError(f"Cannot infer source: {checkpoint}")

            match = EPOCH_RE.search(checkpoint.name)
            if not match:
                raise RuntimeError(f"Cannot parse epoch: {checkpoint}")

            epoch = int(match.group(1))
            metric_log, ap, ap50, ap75, metric_source, independent_test = (
                resolve_metric(stage, scope, work_dir, epoch)
            )

            config = work_dir / "runtime_config.py"
            if not config.is_file():
                config = scope / "runtime_config.py"

            artifacts.append(
                Artifact(
                    experiment=experiment,
                    source=source,
                    backbone=infer_arch(work_dir, scope),
                    stage=stage,
                    checkpoint_epoch=epoch,
                    checkpoint=checkpoint,
                    work_dir=work_dir,
                    scope=scope,
                    config=config if config.is_file() else None,
                    logs=collect_stage_logs(
                        stage, scope, work_dir, metric_log
                    ),
                    metric_log=metric_log,
                    metric_source=metric_source,
                    independent_test=independent_test,
                    ap=ap,
                    ap50=ap50,
                    ap75=ap75,
                )
            )

            seen.add(checkpoint_text)

    artifacts.sort(
        key=lambda item: (
            item.experiment,
            item.source,
            item.backbone,
            item.stage,
        )
    )
    return artifacts


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_dev,
            source_stat.st_ino,
        ) == (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            return
        destination.unlink()

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_artifacts(
    artifacts: Sequence[Artifact], staging_root: Path
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for index, artifact in enumerate(artifacts, 1):
        combo = (
            f"{source_slug(artifact.source)}_{arch_slug(artifact.backbone)}"
        )
        stage_dir = (
            staging_root
            / artifact.experiment
            / combo
            / artifact.stage
        )
        weights_dir = stage_dir / "weights"
        config_dir = stage_dir / "config"
        logs_dir = stage_dir / "logs"
        provenance_dir = staging_root / artifact.experiment / combo / "provenance"

        checkpoint_destination = weights_dir / artifact.checkpoint.name
        copy_or_link(artifact.checkpoint, checkpoint_destination)

        config_destination: Optional[Path] = None
        if artifact.config is not None:
            config_destination = config_dir / "runtime_config.py"
            copy_or_link(artifact.config, config_destination)

        log_destinations: List[Path] = []
        for log in artifact.logs:
            destination = logs_dir / log.name
            if destination.exists():
                destination = logs_dir / f"{log.parent.name}__{log.name}"
            copy_or_link(log, destination)
            log_destinations.append(destination)

        provenance_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "run_provenance.tsv",
            "manifest.tsv",
            "pretrained_sources.tsv",
            "results.tsv",
            "status.tsv",
        ):
            source_file = artifact.scope / name
            if source_file.is_file():
                copy_or_link(source_file, provenance_dir / name)

        pipeline_logs_dir = provenance_dir / "pipeline_logs"
        source_log_dir = artifact.scope / "logs"
        if source_log_dir.is_dir():
            for log in source_log_dir.glob("*.log"):
                lowered = log.name.lower()
                if any(
                    token in lowered
                    for token in (
                        "launcher",
                        "master_pipeline",
                        "pipeline",
                        "convert",
                        "export",
                    )
                ):
                    copy_or_link(log, pipeline_logs_dir / log.name)

        origin_paths = stage_dir / "origin_paths.txt"
        origin_paths.write_text(
            "\n".join(
                [
                    f"checkpoint={artifact.checkpoint}",
                    f"work_dir={artifact.work_dir}",
                    f"run_scope={artifact.scope}",
                    f"config={artifact.config or ''}",
                    f"metric_log={artifact.metric_log or ''}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        metric_name = (
            "coco/bbox_mAP"
            if artifact.stage in {"DFUI", "RUOD"}
            else "coco/segm_mAP"
        )
        status = (
            "complete"
            if artifact.ap is not None and artifact.independent_test
            else "complete_validation_only"
            if artifact.ap is not None
            else "metric_missing"
        )

        metrics = {
            "experiment": artifact.experiment,
            "source": artifact.source,
            "backbone": artifact.backbone,
            "stage": artifact.stage,
            "checkpoint_epoch": artifact.checkpoint_epoch,
            "checkpoint_original": str(artifact.checkpoint),
            "checkpoint_repo": checkpoint_destination.relative_to(
                staging_root
            ).as_posix(),
            "metric": metric_name,
            "metric_source": artifact.metric_source,
            "independent_test": artifact.independent_test,
            "AP": artifact.ap,
            "AP50": artifact.ap50,
            "AP75": artifact.ap75,
            "metric_log_original": (
                str(artifact.metric_log)
                if artifact.metric_log is not None
                else ""
            ),
            "status": status,
        }
        metrics_destination = stage_dir / "metrics.json"
        metrics_destination.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        checkpoint_sha256 = sha256(checkpoint_destination)

        rows.append(
            {
                "experiment": artifact.experiment,
                "source": artifact.source,
                "backbone": artifact.backbone,
                "stage": artifact.stage,
                "checkpoint_epoch": artifact.checkpoint_epoch,
                "metric": metric_name,
                "metric_source": artifact.metric_source,
                "independent_test": artifact.independent_test,
                "AP": artifact.ap,
                "AP50": artifact.ap50,
                "AP75": artifact.ap75,
                "original_checkpoint_path": str(artifact.checkpoint),
                "repo_checkpoint_path": checkpoint_destination.relative_to(
                    staging_root
                ).as_posix(),
                "original_config_path": (
                    str(artifact.config) if artifact.config is not None else ""
                ),
                "repo_config_path": (
                    config_destination.relative_to(staging_root).as_posix()
                    if config_destination is not None
                    else ""
                ),
                "original_log_paths": ";".join(
                    str(path) for path in artifact.logs
                ),
                "repo_log_paths": ";".join(
                    path.relative_to(staging_root).as_posix()
                    for path in log_destinations
                ),
                "status": status,
                "checkpoint_sha256": checkpoint_sha256,
                "size_bytes": checkpoint_destination.stat().st_size,
            }
        )

        print(
            f"[{index:02d}/{len(artifacts)}] "
            f"{artifact.experiment} {artifact.source} "
            f"{artifact.backbone} {artifact.stage} "
            f"AP={artifact.ap} source={artifact.metric_source}"
        )

    return rows


def write_metadata(staging_root: Path) -> None:
    (staging_root / "ALIASES.tsv").write_text(
        "canonical_name\talias\tkind\tnote\n"
        "dfui_realuw_task\tDFUI_RUOD_UIIS_EASY\tdataset\t"
        "DFUI + RUOD Easy + UIIS Easy\n"
        "dfui_realuw_task\tdfui_ruod_uiis_easy\trun_alias\t"
        "historical experiment naming\n"
        "sfui_realuw_task\tAquaOV255 Easy\tdataset\t"
        "AquaOV255 0.70 threshold Easy split\n",
        encoding="utf-8",
    )

    (staging_root / "README.md").write_text(
        """# RealUW random/task best checkpoints

This snapshot contains 48 best checkpoints from four experiment groups:

- `dfui_realuw_random`
- `sfui_realuw_random`
- `dfui_realuw_task`
- `sfui_realuw_task`

Each group contains six pretraining/backbone combinations and two final
stages:

- DFUI and RUOD for detection experiments
- SFUI and UIIS for segmentation experiments

The DFUI/RUOD weights use bbox AP. The SFUI/UIIS weights use segmentation AP.

Two UIIS R50 entries have no independent test log. Their best validation
checkpoints and validation metrics are included. `metrics.json` marks these
entries with `metric_source=validation` and `independent_test=false`.

The complete machine-readable index is `MANIFEST.tsv`. Original server paths
are retained in the manifest, while repository paths are relative to this
directory.

Partial manifests are stored under `_partials/` for upload provenance.
""",
        encoding="utf-8",
    )


def write_manifest_and_checksums(
    staging_root: Path,
    profile: str,
    rows: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    partial_dir = staging_root / "_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = partial_dir / f"MANIFEST_{profile}.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    checksum_path = partial_dir / f"SHA256SUMS_{profile}.txt"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(staging_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("SHA256SUMS_"):
                continue
            handle.write(
                f"{sha256(path)}  {path.relative_to(staging_root).as_posix()}\n"
            )

    return manifest_path, checksum_path


def upload_staging(
    staging_root: Path,
    profile: str,
    repo_id: str,
    repo_type: str,
    remote_root: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    whoami = api.whoami()
    repo = api.repo_info(repo_id=repo_id, repo_type=repo_type)

    print(f"HF_USER={whoami.get('name')}")
    print(f"REPO_PRIVATE={repo.private}")
    print(f"UPLOAD={staging_root} -> {repo_id}:{remote_root}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        folder_path=str(staging_root),
        path_in_repo=remote_root,
        commit_message=f"Add {profile} artifacts for realuw_random_task_best",
        ignore_patterns=[
            "**/.DS_Store",
            "**/*.tmp",
        ],
    )
    print("UPLOAD=PASS")


def finalize_remote(
    staging_root: Path,
    profile: str,
    manifest_path: Path,
    checksum_path: Path,
    repo_id: str,
    repo_type: str,
    remote_root: str,
) -> None:
    if profile != "fcp":
        raise RuntimeError("--finalize must be run with the fcp profile")

    from huggingface_hub import HfApi, RepoFile, hf_hub_download

    api = HfApi()
    merge_dir = staging_root / "_finalize"
    merge_dir.mkdir(parents=True, exist_ok=True)

    fuping_manifest = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        filename=f"{remote_root}/_partials/MANIFEST_fuping.tsv",
        local_dir=merge_dir,
    )
    fuping_checksums = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        filename=f"{remote_root}/_partials/SHA256SUMS_fuping.txt",
        local_dir=merge_dir,
    )

    header: Optional[str] = None
    manifest_rows = []
    seen_manifest_rows = set()

    for path in (Path(fuping_manifest), manifest_path):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise RuntimeError(f"Empty manifest: {path}")
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise RuntimeError("Partial manifest headers differ")

        for line in lines[1:]:
            if not line or line in seen_manifest_rows:
                continue
            seen_manifest_rows.add(line)
            manifest_rows.append(line)

    if len(manifest_rows) != 48:
        raise RuntimeError(
            f"Expected 48 merged manifest rows, got {len(manifest_rows)}"
        )

    merged_manifest = staging_root / "MANIFEST.tsv"
    merged_manifest.write_text(
        "\n".join([header] + sorted(manifest_rows)) + "\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for path in (Path(fuping_checksums), checksum_path):
        checksum_lines.extend(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    merged_checksums = staging_root / "SHA256SUMS"
    merged_checksums.write_text(
        "\n".join(sorted(set(checksum_lines))) + "\n",
        encoding="utf-8",
    )

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        path_or_fileobj=str(merged_manifest),
        path_in_repo=f"{remote_root}/MANIFEST.tsv",
        commit_message="Add merged 48-checkpoint manifest",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        revision="main",
        path_or_fileobj=str(merged_checksums),
        path_in_repo=f"{remote_root}/SHA256SUMS",
        commit_message="Add merged SHA256 checksums",
    )

    remote_files = list(
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            revision="main",
            path_in_repo=remote_root,
            recursive=True,
        )
    )
    weights = [
        item
        for item in remote_files
        if isinstance(item, RepoFile)
        and item.path.endswith(".pth")
        and "best_coco_" in item.path
    ]

    print(f"REMOTE_BEST_WEIGHTS={len(weights)}")
    if len(weights) != 48:
        raise RuntimeError(
            f"Remote checkpoint count mismatch: {len(weights)} != 48"
        )
    print("FINALIZATION=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("fuping", "fcp"),
        help="Source account/profile whose checkpoints should be staged.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        help=f"Hugging Face repository type (default: {DEFAULT_REPO_TYPE}).",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Destination directory inside the repository "
        f"(default: {DEFAULT_REMOTE_ROOT}).",
    )
    parser.add_argument(
        "--staging-base",
        type=Path,
        default=DEFAULT_STAGING_BASE,
        help=f"Parent directory for staging (default: {DEFAULT_STAGING_BASE}).",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        help="Use an explicit staging directory instead of a timestamped one.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the staged directory to Hugging Face after validation.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "After fcp upload, merge the fuping partial manifest and verify "
            "that the remote snapshot contains 48 best checkpoints."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profile = args.profile
    expected = EXPECTED_ROW_COUNTS[profile]

    if args.finalize and not args.upload:
        raise SystemExit("--finalize requires --upload")

    if args.staging_root is not None:
        staging_root = args.staging_root
        if staging_root.exists() and any(staging_root.iterdir()):
            raise SystemExit(f"Staging directory is not empty: {staging_root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_root = (
            args.staging_base
            / f"realuw_random_task_best_{profile}_{timestamp}"
        )

    staging_root.mkdir(parents=True, exist_ok=False)

    artifacts = collect_artifacts(profile)
    if len(artifacts) != expected:
        for artifact in artifacts:
            print(
                artifact.experiment,
                artifact.source,
                artifact.backbone,
                artifact.stage,
                artifact.checkpoint,
            )
        raise RuntimeError(
            f"Expected {expected} artifacts, found {len(artifacts)}"
        )

    print(f"PROFILE={profile}")
    print(f"STAGING_ROOT={staging_root}")
    print(f"ARTIFACTS={len(artifacts)}")
    print(
        "BY_EXPERIMENT=",
        dict(Counter(artifact.experiment for artifact in artifacts)),
    )
    print(
        "BY_STAGE=",
        dict(Counter(artifact.stage for artifact in artifacts)),
    )

    rows = stage_artifacts(artifacts, staging_root)
    write_metadata(staging_root)
    manifest_path, checksum_path = write_manifest_and_checksums(
        staging_root, profile, rows
    )

    print(f"MANIFEST={manifest_path}")
    print(f"CHECKSUMS={checksum_path}")

    if args.upload:
        upload_staging(
            staging_root=staging_root,
            profile=profile,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            remote_root=args.remote_root,
        )
    else:
        print("UPLOAD=SKIPPED")

    if args.finalize:
        finalize_remote(
            staging_root=staging_root,
            profile=profile,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            remote_root=args.remote_root,
        )


if __name__ == "__main__":
    main()
