#!/usr/bin/env python3
"""Prepare configurable SFUI/UIIS Mask R-CNN configs and export backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config


def _set_leaf_dataset(
    dataset, *, root: Path, ann_file: Path, image_prefix: str, classes=None
) -> None:
    if "dataset" in dataset:
        _set_leaf_dataset(
            dataset["dataset"],
            root=root,
            ann_file=ann_file,
            image_prefix=image_prefix,
            classes=classes,
        )
        return
    if "datasets" in dataset:
        for child in dataset["datasets"]:
            _set_leaf_dataset(
                child, root=root, ann_file=ann_file, image_prefix=image_prefix, classes=classes
            )
        return
    dataset["data_root"] = str(root) + "/"
    dataset["ann_file"] = str(ann_file)
    dataset["data_prefix"] = dict(img=image_prefix)
    if classes is not None:
        dataset["metainfo"] = dict(classes=classes)


def _set_evaluator(cfg, name: str, ann_file: Path) -> None:
    evaluator = cfg.get(name)
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            item["ann_file"] = str(ann_file)
    elif evaluator is not None:
        evaluator["ann_file"] = str(ann_file)


def _set_mask_class_count(model, count: int) -> None:
    roi_head = model.get("roi_head")
    if roi_head is None:
        raise SystemExit("Config has no model.roi_head")

    changed = 0

    def update(head):
        nonlocal changed
        if isinstance(head, (list, tuple)):
            for item in head:
                update(item)
        elif isinstance(head, dict):
            if "num_classes" in head:
                head["num_classes"] = count
                changed += 1

    for key in ("bbox_head", "mask_head"):
        if key in roi_head:
            update(roi_head[key])
    if changed < 2:
        raise SystemExit(
            f"Expected bbox and mask heads with num_classes; updated only {changed} fields"
        )


def _set_schedule(cfg, epochs: int, milestones: str) -> None:
    old_epochs = int(cfg.train_cfg.max_epochs)
    cfg.train_cfg.max_epochs = epochs
    explicit = [int(value) for value in milestones.split(",") if value.strip()]
    for scheduler in cfg.get("param_scheduler", []):
        if "milestones" not in scheduler:
            continue
        if explicit:
            scheduler["milestones"] = explicit
        elif old_epochs > 0 and old_epochs != epochs:
            scheduler["milestones"] = [
                max(1, round(int(value) * epochs / old_epochs))
                for value in scheduler["milestones"]
            ]


def prepare_sfui(args) -> None:
    cfg = Config.fromfile(args.base_config)
    annotation = Path(args.train_ann).resolve()
    coco = json.loads(annotation.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    classes = tuple(item["name"] for item in categories)
    if not classes or len(set(classes)) != len(classes):
        raise SystemExit(f"Invalid/empty categories in SFUI annotation: {annotation}")
    _set_mask_class_count(cfg.model, len(classes))
    _set_leaf_dataset(
        cfg.train_dataloader.dataset,
        root=Path(args.data_root).resolve(),
        ann_file=annotation,
        image_prefix=args.train_prefix,
        classes=classes,
    )
    cfg.model.backbone["init_cfg"] = dict(type="Pretrained", checkpoint=args.init_checkpoint)
    cfg.load_from = None
    _set_schedule(cfg, args.epochs, args.milestones)
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.default_hooks.checkpoint.max_keep_ckpts = args.max_keep_ckpts
    cfg.default_hooks.checkpoint.save_last = True
    if args.val_ann:
        cfg.default_hooks.checkpoint.save_best = "coco/segm_mAP"
    else:
        cfg.default_hooks.checkpoint.pop("save_best", None)

    if args.val_ann:
        val_ann = Path(args.val_ann).resolve()
        _set_leaf_dataset(
            cfg.val_dataloader.dataset,
            root=Path(args.data_root).resolve(),
            ann_file=val_ann,
            image_prefix=args.val_prefix,
            classes=classes,
        )
        _set_evaluator(cfg, "val_evaluator", val_ann)
    else:
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.val_cfg = None

    test_ann = Path(args.test_ann or args.train_ann).resolve()
    _set_leaf_dataset(
        cfg.test_dataloader.dataset,
        root=Path(args.data_root).resolve(),
        ann_file=test_ann,
        image_prefix=args.test_prefix or args.train_prefix,
    )
    _set_evaluator(cfg, "test_evaluator", test_ann)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(args.output)
    print(
        f"SFUI_CONFIG={args.output} classes={len(classes)} epochs={args.epochs} "
        f"lr={args.lr if args.lr is not None else 'CONFIG_DEFAULT'} "
        f"train_ann={annotation} val_ann={args.val_ann or 'DISABLED'} "
        f"test_ann={test_ann}"
    )


def prepare_uiis(args) -> None:
    cfg = Config.fromfile(args.base_config)
    root = Path(args.data_root).resolve()
    train_ann = (root / "annotations" / args.train_ann).resolve()
    val_ann = (root / "annotations" / args.val_ann).resolve()
    if not train_ann.is_file() or not val_ann.is_file():
        raise SystemExit(f"Missing UIIS annotations: {train_ann} or {val_ann}")

    _set_leaf_dataset(
        cfg.train_dataloader.dataset,
        root=root,
        ann_file=train_ann,
        image_prefix=args.train_prefix,
    )
    _set_leaf_dataset(
        cfg.val_dataloader.dataset,
        root=root,
        ann_file=val_ann,
        image_prefix=args.val_prefix,
    )
    _set_leaf_dataset(
        cfg.test_dataloader.dataset,
        root=root,
        ann_file=val_ann,
        image_prefix=args.val_prefix,
    )
    _set_evaluator(cfg, "val_evaluator", val_ann)
    _set_evaluator(cfg, "test_evaluator", val_ann)
    cfg.model.backbone["init_cfg"] = dict(type="Pretrained", checkpoint=args.init_checkpoint)
    cfg.load_from = None
    _set_schedule(cfg, args.epochs, args.milestones)
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.default_hooks.checkpoint.max_keep_ckpts = args.max_keep_ckpts
    cfg.default_hooks.checkpoint.save_last = True
    cfg.default_hooks.checkpoint.save_best = "coco/segm_mAP"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(args.output)
    print(
        f"UIIS_CONFIG={args.output} epochs={args.epochs} "
        f"lr={args.lr if args.lr is not None else 'CONFIG_DEFAULT'} "
        f"train_ann={train_ann} val_ann={val_ann} init={args.init_checkpoint}"
    )


def export_backbone(args) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    backbone = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.") and torch.is_tensor(value)
    }
    if not backbone:
        raise SystemExit(f"No backbone.* tensors found in {args.checkpoint}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": backbone, "meta": {"source_checkpoint": str(Path(args.checkpoint).resolve())}},
        output,
    )
    print(f"BACKBONE={output} tensors={len(backbone)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    sfui = commands.add_parser("prepare-sfui")
    for flag in ("base-config", "output", "init-checkpoint", "data-root", "train-ann"):
        sfui.add_argument("--" + flag, required=True)
    sfui.add_argument("--val-ann")
    sfui.add_argument("--test-ann")
    sfui.add_argument("--train-prefix", default="train/images/")
    sfui.add_argument("--val-prefix", default="train/images/")
    sfui.add_argument("--test-prefix")
    sfui.add_argument("--epochs", type=int, default=48)
    sfui.add_argument("--lr", type=float)
    sfui.add_argument("--max-keep-ckpts", type=int, default=5)
    sfui.add_argument("--milestones", default="")
    sfui.set_defaults(func=prepare_sfui)

    uiis = commands.add_parser("prepare-uiis")
    for flag, default in (
        ("base-config", None),
        ("output", None),
        ("init-checkpoint", None),
        ("data-root", None),
        ("train-ann", "instances_train.json"),
        ("val-ann", "instances_val.json"),
    ):
        option = "--" + flag
        if default is None:
            uiis.add_argument(option, required=True)
        else:
            uiis.add_argument(option, default=default)
    uiis.add_argument("--train-prefix", default="train/")
    uiis.add_argument("--val-prefix", default="val/")
    uiis.add_argument("--epochs", type=int, default=24)
    uiis.add_argument("--lr", type=float)
    uiis.add_argument("--max-keep-ckpts", type=int, default=5)
    uiis.add_argument("--milestones", default="")
    uiis.set_defaults(func=prepare_uiis)

    export = commands.add_parser("export-backbone")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=export_backbone)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
