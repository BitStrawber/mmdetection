#!/usr/bin/env python3
"""Create COCO A/B folds, cross-filter them by per-image bbox AP, and sample DFUI size."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_coco(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_coco(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def split(args: argparse.Namespace) -> None:
    coco = read_coco(args.annotation)
    images = list(coco.get("images", []))
    if len(images) < 2:
        raise SystemExit("COCO annotation must contain at least two images")
    random.Random(args.seed).shuffle(images)
    midpoint = len(images) // 2
    folds = {
        "A": {int(image["id"]) for image in images[:midpoint]},
        "B": {int(image["id"]) for image in images[midpoint:]},
    }
    if folds["A"] & folds["B"] or folds["A"] | folds["B"] != {
        int(image["id"]) for image in images
    }:
        raise SystemExit("A/B split validation failed")
    for name, ids in folds.items():
        part = dict(coco)
        part["images"] = [image for image in coco["images"] if int(image["id"]) in ids]
        part["annotations"] = [
            ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in ids
        ]
        write_coco(part, args.output / f"train_{name}.json")
        print(f"fold_{name}: images={len(part['images'])} annotations={len(part['annotations'])}")
    (args.output / "split.json").write_text(
        json.dumps({"seed": args.seed, "images": len(images), "A": len(folds["A"]), "B": len(folds["B"])}, indent=2) + "\n",
        encoding="utf-8",
    )


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def per_image_ap(predictions: list[tuple[list[float], float, int]], ground_truth: list[dict[str, Any]]) -> float:
    if not ground_truth:
        return 1.0
    if not predictions:
        return 0.0
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in ground_truth:
        by_category[int(ann["category_id"])].append(ann)
    predictions.sort(key=lambda item: item[1], reverse=True)
    aps = []
    for threshold_index in range(10):
        iou_threshold = 0.50 + 0.05 * threshold_index
        matched: dict[int, set[int]] = defaultdict(set)
        tp, fp = [], []
        for box, _score, category_id in predictions:
            candidates = by_category.get(category_id, [])
            best_iou, best_index = 0.0, -1
            for index, ann in enumerate(candidates):
                if index in matched[category_id]:
                    continue
                overlap = box_iou(box, [float(value) for value in ann["bbox"]])
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            is_tp = best_iou >= iou_threshold
            tp.append(1.0 if is_tp else 0.0)
            fp.append(0.0 if is_tp else 1.0)
            if is_tp:
                matched[category_id].add(best_index)
        tp_sum = fp_sum = ap = 0.0
        precision_recall = []
        for true_positive, false_positive in zip(tp, fp):
            tp_sum += true_positive
            fp_sum += false_positive
            precision_recall.append((tp_sum / max(tp_sum + fp_sum, 1e-12), tp_sum / len(ground_truth)))
        for point in range(101):
            recall = point / 100
            ap += max((p for p, r in precision_recall if r >= recall), default=0.0) / 101
        aps.append(ap)
    return sum(aps) / len(aps)


def filter_fold(args: argparse.Namespace) -> None:
    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as exc:
        raise SystemExit(f"MMDetection is required for filtering: {exc}") from exc

    coco = read_coco(args.annotation)
    image_root = args.image_root
    category_ids = [int(item["id"]) for item in sorted(coco["categories"], key=lambda x: int(x["id"]))]
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations[int(ann["image_id"])].append(ann)

    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    kept, kept_ids = [], set()
    score_rows = ["image_id\tfile_name\tinstances\tper_image_bbox_map\tselected\n"]
    for index, image in enumerate(coco["images"], 1):
        path = image_root / image["file_name"]
        if not path.is_file():
            raise SystemExit(f"Missing COCO image: {path}")
        result = inference_detector(model, str(path))
        pred = result.pred_instances.cpu()
        boxes = pred.bboxes.numpy()
        scores = pred.scores.numpy()
        labels = pred.labels.numpy()
        predictions = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            label = int(label)
            if x2 > x1 and y2 > y1 and label < len(category_ids):
                predictions.append(([x1, y1, x2 - x1, y2 - y1], float(score), category_ids[label]))
        image_id = int(image["id"])
        gt = annotations[image_id]
        score = per_image_ap(predictions, gt)
        selected = score >= args.threshold
        if selected:
            kept.append(image)
            kept_ids.add(image_id)
        score_rows.append(f"{image_id}\t{image['file_name']}\t{len(gt)}\t{score:.8f}\t{int(selected)}\n")
        if index % 1000 == 0 or index == len(coco["images"]):
            print(f"filtered={index}/{len(coco['images'])} retained={len(kept)}", flush=True)

    output = dict(coco)
    output["images"] = kept
    output["annotations"] = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in kept_ids]
    write_coco(output, args.output / "instances_easy.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_image_scores.tsv").write_text("".join(score_rows), encoding="utf-8")
    summary = {
        "source_annotation": str(args.annotation.resolve()),
        "teacher_config": str(args.config.resolve()),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "images_examined": len(coco["images"]),
        "images_selected": len(kept),
        "annotations_selected": len(output["annotations"]),
        "threshold": args.threshold,
        "metric": "RUOD/Aqua-compatible custom per-image bbox AP averaged over IoU 0.50:0.95, 101-point interpolation",
        "selection_rule": "retain if per-image bbox AP >= threshold; retained annotations remain original COCO ground truth",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def sample(args: argparse.Namespace) -> None:
    import os

    easy = read_coco(args.easy_annotation)
    images = list(easy.get("images", []))
    if len(images) < args.count:
        raise SystemExit(f"Easy pool has {len(images)} images, fewer than requested {args.count}")
    random.Random(args.seed).shuffle(images)
    chosen = images[:args.count]
    train_images, val_images = chosen[:args.train_count], chosen[args.train_count:]
    if len(val_images) != args.val_count or args.train_count + args.val_count != args.count:
        raise SystemExit("Requested train/val counts do not sum to sample count")
    args.output.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in easy.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)
    categories = easy.get("categories", [])
    info = easy.get("info", {})
    licenses = easy.get("licenses", [])
    image_dir = args.output / "images"
    image_dir.mkdir()
    for image in chosen:
        source = args.image_root / image["file_name"]
        target = image_dir / Path(image["file_name"]).name
        if target.exists():
            raise SystemExit(f"Sample filename collision: {target.name}")
        try:
            os.link(source, target)
        except OSError:
            import shutil
            shutil.copy2(source, target)
    for split_name, split_images in (("train", train_images), ("val", val_images)):
        ids = {int(image["id"]) for image in split_images}
        part = {
            "info": info,
            "licenses": licenses,
            "images": [dict(image, file_name=Path(image["file_name"]).name) for image in split_images],
            "annotations": [ann for image_id in ids for ann in annotations_by_image[image_id]],
            "categories": categories,
        }
        write_coco(part, args.output / "annotations" / f"instances_{split_name}.json")
        print(f"{split_name}: images={len(part['images'])} annotations={len(part['annotations'])}")
    metadata = {
        "source": str(args.easy_annotation.resolve()), "seed": args.seed,
        "easy_pool_images": len(easy.get("images", [])), "sampled_images": args.count,
        "train_images": args.train_count, "val_images": args.val_count,
        "image_storage": "hard links to COCO train2017",
    }
    (args.output / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT=PASS_DFUI_COCO_EASY_SAMPLE output={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split")
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=split)
    p = sub.add_parser("filter")
    for name in ("annotation", "image-root", "config", "checkpoint", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=filter_fold)
    p = sub.add_parser("sample")
    for name in ("easy-annotation", "image-root", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--count", type=int, default=11494)
    p.add_argument("--train-count", type=int, default=9770)
    p.add_argument("--val-count", type=int, default=1724)
    p.add_argument("--seed", type=int, default=20261016)
    p.set_defaults(func=sample)
    args = parser.parse_args()
    if getattr(args, "threshold", 0.0) < 0 or getattr(args, "threshold", 1.0) > 1:
        raise SystemExit("threshold must be in [0, 1]")
    args.func(args)


if __name__ == "__main__":
    main()
