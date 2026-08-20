#!/usr/bin/env python3
"""Render scored Cascade R-CNN detections on a shared RUOD image manifest.

One invocation loads one or more complete detector checkpoints and writes two
paper-facing variants for every image: a uniform-color version and a version
whose boxes are colored by detector identity. The companion shell launcher
assigns one detector to each requested GPU and composes model-comparison panels
after all workers finish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.exp_2.backbone_analysis.model_adapter import load_model  # noqa: E402


MODEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    'imagenet_dino100e_ruod_cascade': (0, 114, 178),
    'realuw_dino100e_ruod_cascade': (230, 159, 0),
    'synthetic5_dino100e_ruod_cascade': (0, 158, 115),
    'imagenet_dino100e_dfui_ruod_cascade': (204, 121, 167),
}
UNIFORM_COLOR = (255, 212, 0)
MODEL_LABELS: Dict[str, str] = {
    'imagenet_dino100e_ruod_cascade': 'ImageNet',
    'realuw_dino100e_ruod_cascade': 'RealUW',
    'synthetic5_dino100e_ruod_cascade': 'Synthetic5',
    'imagenet_dino100e_dfui_ruod_cascade': 'ImageNet + DFUI',
}


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected an object')
            rows.append(row)
    if not rows:
        raise ValueError(f'No sample rows found in {path}')
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write('\n')
    os.replace(temporary, path)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        'C:/Windows/Fonts/arial.ttf',
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def clamped(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def annotation_style(
    image_width: int,
    image_height: int,
    line_width: int,
    font_scale: float,
    font_min_size: int,
    font_max_size: int,
) -> Tuple[int, ImageFont.ImageFont, int]:
    short_side = min(image_width, image_height)
    resolved_line_width = (
        int(line_width) if line_width > 0 else clamped(short_side / 210.0, 2, 6))
    font_size = clamped(short_side * font_scale, font_min_size, font_max_size)
    padding = clamped(font_size * 0.30, 3, 8)
    return resolved_line_width, load_font(font_size), padding


def text_color(background: Tuple[int, int, int]) -> Tuple[int, int, int]:
    red, green, blue = background
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (18, 18, 18) if luminance >= 145 else (255, 255, 255)


def clipped_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    if width < 2 or height < 2:
        raise ValueError(f'Cannot draw a detection on an invalid image size: {width}x{height}')
    left = max(0, min(width - 2, int(np.floor(x1))))
    top = max(0, min(height - 2, int(np.floor(y1))))
    right = max(left + 1, min(width - 1, int(np.ceil(x2))))
    bottom = max(top + 1, min(height - 1, int(np.ceil(y2))))
    return left, top, right, bottom


def draw_predictions(
    image_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    color: Tuple[int, int, int],
    line_width: int,
    font_scale: float,
    font_min_size: int,
    font_max_size: int,
    include_score: bool,
) -> Image.Image:
    with Image.open(image_path) as source:
        canvas = source.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    resolved_width, font, padding = annotation_style(
        width, height, line_width, font_scale, font_min_size, font_max_size)
    foreground = text_color(color)

    # Draw low-confidence boxes first so high-confidence boxes remain visible.
    for row in sorted(predictions, key=lambda item: float(item['score'])):
        left, top, right, bottom = clipped_box(row['bbox_xyxy'], width, height)
        draw.rectangle((left, top, right, bottom), outline=color, width=resolved_width)
        label = str(row['class_name'])
        if include_score:
            label = f'{label} {float(row["score"]):.2f}'
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding
        label_left = min(max(0, left), max(0, width - label_width))
        label_top = top - label_height - 2
        if label_top < 0:
            label_top = min(height - label_height, bottom + 2)
        draw.rounded_rectangle(
            (label_left, label_top, label_left + label_width, label_top + label_height),
            radius=max(2, padding // 2), fill=color)
        draw.text(
            (label_left + padding, label_top + padding - bounds[1]),
            label, font=font, fill=foreground)
    return canvas


def image_destination(root: Path, color_mode: str, model_id: str, image_id: int) -> Path:
    return root / color_mode / 'images' / model_id / f'image_{image_id:08d}.png'


def model_color(model_id: str, position: int) -> Tuple[int, int, int]:
    fallback = ((0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167))
    return MODEL_COLORS.get(model_id, fallback[position % len(fallback)])


def class_name(classes: Sequence[str], label: int) -> str:
    if 0 <= label < len(classes):
        return str(classes[label])
    return f'class-{label}'


def selected_specs(config: Mapping[str, Any], selected: Sequence[str]) -> List[Mapping[str, Any]]:
    specs = list(config.get('models', []))
    selected_set = set(selected)
    if selected_set:
        specs = [spec for spec in specs if str(spec.get('id')) in selected_set]
    for spec in specs:
        if str(spec.get('kind', 'detector')) != 'detector':
            raise ValueError(f'{spec.get("id")}: only complete detector models are supported')
    if not specs:
        raise ValueError('No detector model is selected')
    return specs


def render_model(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.models_config).expanduser().resolve()
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError('Both --manifest and --models-config must exist')
    manifest = read_jsonl(manifest_path)
    specs = selected_specs(read_json(config_path), parse_csv(args.models))
    root = Path(args.out_dir).expanduser().resolve()
    color_modes = parse_csv(args.color_modes)
    if not set(color_modes).issubset({'uniform', 'model'}):
        raise ValueError('--color-modes must contain only uniform and/or model')
    if not color_modes:
        raise ValueError('At least one color mode is required')
    if args.max_detections <= 0:
        raise ValueError('--max-detections must be positive')
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError('--score-threshold must be in [0, 1]')

    from mmdet.apis import inference_detector

    for model_position, spec in enumerate(specs):
        loaded = load_model(spec, args.device)
        model_id = loaded.model_id
        classes = list((getattr(loaded.model, 'dataset_meta', {}) or {}).get('classes', []))
        report_rows: List[Dict[str, Any]] = []
        try:
            for position, row in enumerate(manifest, 1):
                image_id = int(row['image_id'])
                image_path = Path(str(row['image_path'])).expanduser().resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f'Missing image: {image_path}')
                result = inference_detector(loaded.model, str(image_path))
                instances = result.pred_instances.cpu()
                boxes = instances.bboxes.numpy().astype(np.float32)
                scores = instances.scores.numpy().astype(np.float32)
                labels = instances.labels.numpy().astype(np.int64)
                order = np.argsort(-scores)
                predictions: List[Dict[str, Any]] = []
                for rank, index in enumerate(order, 1):
                    score = float(scores[index])
                    if score < args.score_threshold:
                        continue
                    bbox = boxes[index].tolist()
                    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < args.minimum_box_area:
                        continue
                    label = int(labels[index])
                    predictions.append({
                        'rank': rank,
                        'bbox_xyxy': [float(value) for value in bbox],
                        'score': score,
                        'label': label,
                        'class_name': class_name(classes, label),
                    })
                    if len(predictions) >= args.max_detections:
                        break
                colors = {
                    'uniform': UNIFORM_COLOR,
                    'model': model_color(model_id, model_position),
                }
                for mode in color_modes:
                    destination = image_destination(root, mode, model_id, image_id)
                    if args.overwrite or not destination.is_file():
                        visual = draw_predictions(
                            image_path, predictions, colors[mode], args.line_width,
                            args.font_scale, args.font_min_size, args.font_max_size,
                            args.include_score)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        visual.save(destination, format='PNG', compress_level=args.png_compress_level)
                report_rows.append({
                    'model': model_id,
                    'image_id': image_id,
                    'image_path': str(image_path),
                    'score_threshold': args.score_threshold,
                    'max_detections': args.max_detections,
                    'predictions': predictions,
                })
                print(
                    f'[{model_id}] {position}/{len(manifest)} image={image_id} '
                    f'predictions={len(predictions)}', flush=True)
            write_jsonl(root / 'metadata' / f'{model_id}.jsonl', report_rows)
            write_json(root / 'metadata' / f'{model_id}.load.json', loaded.load_report)
        finally:
            loaded.close()


def make_tile(image_path: Path, label: str, tile_width: int, tile_height: int) -> Image.Image:
    with Image.open(image_path) as image:
        source = image.convert('RGB')
    source.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
    header_height = max(30, int(tile_height * 0.075))
    tile = Image.new('RGB', (tile_width, tile_height + header_height), (255, 255, 255))
    left = (tile_width - source.width) // 2
    top = header_height + (tile_height - source.height) // 2
    tile.paste(source, (left, top))
    draw = ImageDraw.Draw(tile)
    font = load_font(max(14, min(22, int(header_height * 0.48))))
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.text(((tile_width - (bounds[2] - bounds[0])) // 2, (header_height - (bounds[3] - bounds[1])) // 2 - bounds[1]), label, font=font, fill=(20, 20, 20))
    return tile


def compose_panels(args: argparse.Namespace) -> None:
    manifest = read_jsonl(Path(args.manifest).expanduser().resolve())
    specs = selected_specs(read_json(Path(args.models_config).expanduser().resolve()), parse_csv(args.models))
    models = [str(spec['id']) for spec in specs]
    root = Path(args.out_dir).expanduser().resolve()
    color_modes = parse_csv(args.color_modes)
    for mode in color_modes:
        panel_root = root / mode / 'panels_2x2'
        for position, row in enumerate(manifest, 1):
            image_id = int(row['image_id'])
            paths = [image_destination(root, mode, model_id, image_id) for model_id in models]
            if not all(path.is_file() for path in paths):
                missing = [str(path) for path in paths if not path.is_file()]
                raise FileNotFoundError(f'image {image_id}: missing rendered detector outputs: {missing}')
            tiles = [make_tile(path, MODEL_LABELS.get(model, model), args.panel_tile_width, args.panel_tile_height) for path, model in zip(paths, models)]
            gap = 6
            cell_width, cell_height = tiles[0].size
            panel = Image.new('RGB', (2 * cell_width + gap, 2 * cell_height + gap), (255, 255, 255))
            for index, tile in enumerate(tiles):
                panel.paste(tile, ((index % 2) * (cell_width + gap), (index // 2) * (cell_height + gap)))
            destination = panel_root / f'image_{image_id:08d}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not destination.is_file():
                panel.save(destination, format='PNG', compress_level=args.png_compress_level)
            print(f'[{mode} panels] {position}/{len(manifest)} image={image_id}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Shared RUOD sample manifest.jsonl.')
    parser.add_argument('--models-config', required=True, help='Detector model configuration JSON.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', default='', help='Optional comma-separated detector IDs.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--color-modes', default='uniform,model')
    parser.add_argument('--score-threshold', type=float, default=0.30)
    parser.add_argument('--max-detections', type=int, default=30)
    parser.add_argument('--minimum-box-area', type=float, default=4.0)
    parser.add_argument('--line-width', type=int, default=0, help='0 selects an adaptive width.')
    parser.add_argument('--font-scale', type=float, default=0.032)
    parser.add_argument('--font-min-size', type=int, default=12)
    parser.add_argument('--font-max-size', type=int, default=24)
    parser.add_argument('--include-score', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--panel-tile-width', type=int, default=640)
    parser.add_argument('--panel-tile-height', type=int, default=480)
    parser.add_argument('--png-compress-level', type=int, default=3)
    parser.add_argument('--compose-panels', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError('--png-compress-level must be in [0, 9]')
    if args.compose_panels:
        compose_panels(args)
    else:
        render_model(args)


if __name__ == '__main__':
    main()
