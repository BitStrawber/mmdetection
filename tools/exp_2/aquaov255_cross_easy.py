#!/usr/bin/env python3
"""Build RUOD-style A/B folds and cross-filter AquaOV255 by per-image bbox AP."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_coco(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
        handle.write("\n")


def split(args: argparse.Namespace) -> None:
    source = load_coco(args.annotation)
    images = list(source.get("images", []))
    if len(images) < 2:
        raise SystemExit("Need at least two training images to create A/B folds")
    random.seed(args.seed)
    random.shuffle(images)
    midpoint = len(images) // 2
    image_ids = {
        "A": {int(image["id"]) for image in images[:midpoint]},
        "B": {int(image["id"]) for image in images[midpoint:]},
    }
    if image_ids["A"] & image_ids["B"]:
        raise SystemExit("Internal error: A/B image ids overlap")

    args.output.mkdir(parents=True, exist_ok=False)
    for fold in ("A", "B"):
        fold_coco = dict(source)
        fold_coco["images"] = [image for image in source["images"] if int(image["id"]) in image_ids[fold]]
        fold_coco["annotations"] = [
            annotation for annotation in source.get("annotations", [])
            if int(annotation["image_id"]) in image_ids[fold]
        ]
        write_json(fold_coco, args.output / f"train_{fold}.json")

    with (args.output / "fold_manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write("fold\timage_id\tfile_name\n")
        fold_by_id = {image_id: fold for fold, ids in image_ids.items() for image_id in ids}
        for image in source["images"]:
            handle.write(f"{fold_by_id[int(image['id'])]}\t{image['id']}\t{image['file_name']}\n")
    summary = {
        "source_annotation": str(args.annotation.resolve()),
        "seed": args.seed,
        "split_rule": "random.shuffle(images); A=images[:N//2], B=images[N//2:]",
        "images": len(images),
        "A_images": len(image_ids["A"]),
        "B_images": len(image_ids["B"]),
        "A_annotations": sum(int(item["image_id"]) in image_ids["A"] for item in source.get("annotations", [])),
        "B_annotations": sum(int(item["image_id"]) in image_ids["B"] for item in source.get("annotations", [])),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2))
    print("RESULT=PASS_AQUAOV255_AB_SPLIT")


def box_iou(box_xywh: list[float], gt_xywh: list[float]) -> float:
    x1 = max(box_xywh[0], gt_xywh[0])
    y1 = max(box_xywh[1], gt_xywh[1])
    x2 = min(box_xywh[0] + box_xywh[2], gt_xywh[0] + gt_xywh[2])
    y2 = min(box_xywh[1] + box_xywh[3], gt_xywh[1] + gt_xywh[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_xywh[2] * box_xywh[3] + gt_xywh[2] * gt_xywh[3] - intersection
    return intersection / union if union > 0 else 0.0


def image_bbox_map(
    predictions: list[tuple[list[float], float, int]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Match the current RUOD Easy per-image AP implementation."""
    if not ground_truth:
        return 1.0
    if not predictions:
        return 0.0

    gt_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_class[int(item["category_id"])].append(item)
    predictions.sort(key=lambda item: item[1], reverse=True)
    gt_count = len(ground_truth)
    aps: list[float] = []

    for iou_threshold in [0.50 + 0.05 * index for index in range(10)]:
        true_positive = [0.0] * len(predictions)
        false_positive = [0.0] * len(predictions)
        matched: dict[int, set[int]] = defaultdict(set)
        for rank, (pred_box, _score, category_id) in enumerate(predictions):
            candidates = gt_by_class.get(category_id, [])
            best_iou, best_index = 0.0, -1
            for gt_index, gt in enumerate(candidates):
                if gt_index in matched[category_id]:
                    continue
                iou = box_iou(pred_box, [float(value) for value in gt["bbox"]])
                if iou > best_iou:
                    best_iou, best_index = iou, gt_index
            if best_iou >= iou_threshold:
                true_positive[rank] = 1.0
                matched[category_id].add(best_index)
            else:
                false_positive[rank] = 1.0

        tp_cumulative: list[float] = []
        fp_cumulative: list[float] = []
        tp_sum = fp_sum = 0.0
        for tp, fp in zip(true_positive, false_positive):
            tp_sum += tp
            fp_sum += fp
            tp_cumulative.append(tp_sum)
            fp_cumulative.append(fp_sum)
        recalls = [value / max(gt_count, 1) for value in tp_cumulative]
        precisions = [tp / max(tp + fp, 1e-12) for tp, fp in zip(tp_cumulative, fp_cumulative)]
        ap = 0.0
        for point in range(101):
            recall_point = point / 100
            eligible = [precision for precision, recall in zip(precisions, recalls) if recall >= recall_point]
            ap += (max(eligible) if eligible else 0.0) / 101
        aps.append(ap)
    return sum(aps) / len(aps)


def filter_fold(args: argparse.Namespace) -> None:
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be in [0, 1]")
    if args.output.exists():
        if args.skip_existing and (args.output / ".complete").is_file():
            print(f"REUSE_COMPLETE={args.output}")
            return
        raise SystemExit(f"Output already exists: {args.output}")

    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as error:
        raise SystemExit(f"MMDetection inference dependencies unavailable: {error}") from error

    coco = load_coco(args.annotation)
    images_root = args.images
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    args.output.mkdir(parents=True)
    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    selected_images: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    rows = ["image_id\tfile_name\tgt_instances\tpred_instances\tper_image_bbox_map\tselected\n"]
    image_count = len(coco.get("images", []))
    for index, image in enumerate(coco.get("images", []), start=1):
        image_path = images_root / image["file_name"]
        if not image_path.is_file():
            raise SystemExit(f"Missing fold image: {image_path}")
        result = inference_detector(model, str(image_path))
        instances = result.pred_instances.cpu()
        boxes = instances.bboxes.numpy()
        scores = instances.scores.numpy()
        labels = instances.labels.numpy()
        predictions: list[tuple[list[float], float, int]] = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(([x1, y1, x2 - x1, y2 - y1], float(score), int(label) + 1))
        image_id = int(image["id"])
        ground_truth = annotations_by_image[image_id]
        score = image_bbox_map(predictions, ground_truth)
        keep = score >= args.threshold
        if keep:
            selected_images.append(image)
            selected_ids.add(image_id)
        rows.append(
            f"{image_id}\t{image['file_name']}\t{len(ground_truth)}\t{len(predictions)}\t{score:.8f}\t{int(keep)}\n"
        )
        if index % 100 == 0 or index == image_count:
            print(f"filtered={index}/{image_count} retained={len(selected_images)}", flush=True)

    output_coco = dict(coco)
    output_coco["images"] = selected_images
    output_coco["annotations"] = [
        annotation for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) in selected_ids
    ]
    write_json(output_coco, args.output / "instances_easy.json")
    (args.output / "per_image_scores.tsv").write_text("".join(rows), encoding="utf-8")
    summary = {
        "source_annotation": str(args.annotation.resolve()),
        "teacher_config": str(args.config.resolve()),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "images_examined": image_count,
        "images_selected": len(selected_images),
        "annotations_selected": len(output_coco["annotations"]),
        "threshold": args.threshold,
        "metric": "custom per-image bbox AP averaged over IoU 0.50:0.95 at 0.05 increments, 101-point interpolation",
        "selection_rule": "retain if per_image_bbox_map >= threshold; empty-GT image receives 1.0, GT with no predictions receives 0.0",
        "retained_annotations_are_original_ground_truth": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2))
    print("RESULT=PASS_AQUAOV255_CROSS_EASY_FILTER")


def merge(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    parts = [load_coco(path) for path in args.annotations]
    categories = parts[0].get("categories", [])
    if any(part.get("categories", []) != categories for part in parts[1:]):
        raise SystemExit("Easy-fold COCO category definitions differ")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for part in parts:
        ids = {int(image["id"]) for image in part.get("images", [])}
        if seen_ids & ids:
            raise SystemExit("A/B easy outputs unexpectedly contain overlapping image ids")
        seen_ids |= ids
        images.extend(part.get("images", []))
        annotations.extend(part.get("annotations", []))
    result = {"images": images, "annotations": annotations, "categories": categories}
    write_json(result, args.output)
    print(json.dumps({"images": len(images), "annotations": len(annotations), "categories": len(categories)}, indent=2))
    print("RESULT=PASS_AQUAOV255_EASY_MERGE")


def partition(args: argparse.Namespace) -> None:
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be strictly between 0 and 1")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    coco = load_coco(args.annotation)
    images = list(coco.get("images", []))
    if len(images) < 2:
        raise SystemExit("Need at least two Easy images to create train/val partitions")
    random.Random(args.seed).shuffle(images)
    val_count = max(1, round(len(images) * args.val_ratio))
    val_ids = {int(image["id"]) for image in images[:val_count]}
    train_ids = {int(image["id"]) for image in images[val_count:]}
    if not train_ids or train_ids & val_ids:
        raise SystemExit("Invalid Easy train/val partition")

    args.output.mkdir(parents=True)
    summary: dict[str, Any] = {
        "source_annotation": str(args.annotation.resolve()),
        "seed": args.seed,
        "val_ratio_requested": args.val_ratio,
        "split_rule": "seeded random shuffle; rounded validation count",
    }
    for name, ids in (("train", train_ids), ("val", val_ids)):
        part = dict(coco)
        part["images"] = [image for image in coco["images"] if int(image["id"]) in ids]
        part["annotations"] = [
            annotation for annotation in coco.get("annotations", [])
            if int(annotation["image_id"]) in ids
        ]
        write_json(part, args.output / f"instances_{name}.json")
        summary[f"{name}_images"] = len(part["images"])
        summary[f"{name}_annotations"] = len(part["annotations"])
    summary["total_images"] = len(images)
    summary["actual_val_ratio"] = len(val_ids) / len(images)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / ".complete").touch()
    print(json.dumps(summary, indent=2))
    print("RESULT=PASS_AQUAOV255_EASY_TRAIN_VAL_PARTITION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--annotation", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.set_defaults(func=split)

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--images", type=Path, required=True)
    filter_parser.add_argument("--annotation", type=Path, required=True)
    filter_parser.add_argument("--config", type=Path, required=True)
    filter_parser.add_argument("--checkpoint", type=Path, required=True)
    filter_parser.add_argument("--output", type=Path, required=True)
    filter_parser.add_argument("--threshold", type=float, default=0.60)
    filter_parser.add_argument("--device", default="cuda:0")
    filter_parser.add_argument("--skip-existing", action="store_true")
    filter_parser.set_defaults(func=filter_fold)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.set_defaults(func=merge)

    partition_parser = subparsers.add_parser("partition")
    partition_parser.add_argument("--annotation", type=Path, required=True)
    partition_parser.add_argument("--output", type=Path, required=True)
    partition_parser.add_argument("--val-ratio", type=float, default=0.10)
    partition_parser.add_argument("--seed", type=int, default=20261007)
    partition_parser.set_defaults(func=partition)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
