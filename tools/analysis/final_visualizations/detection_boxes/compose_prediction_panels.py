#!/usr/bin/env python3
"""Compose Original | model A | model B prediction comparison panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left-label", default="R50")
    parser.add_argument("--right-label", default="ViT-S")
    parser.add_argument("--input-label", default="Input")
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=480)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_tile(image: Image.Image, width: int, height: int) -> Image.Image:
    tile = Image.new("RGB", (width, height), "white")
    image = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def main() -> None:
    args = parse_args()
    image_list = Path(args.image_list).expanduser().resolve()
    left_dir = Path(args.left_dir).expanduser().resolve()
    right_dir = Path(args.right_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        Path(line.strip()).expanduser().resolve()
        for line in image_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    titles = (args.input_label, args.left_label, args.right_label)
    label_font = font(24)
    header_height = 40
    gap = 4

    for index, source in enumerate(sources, start=1):
        left = left_dir / f"{index:03d}_{source.stem}.png"
        right = right_dir / f"{index:03d}_{source.stem}.png"
        for required in (source, left, right):
            if not required.is_file():
                raise FileNotFoundError(f"Missing panel source: {required}")

        panel = Image.new(
            "RGB",
            (
                args.cell_width * 3 + gap * 2,
                args.cell_height + header_height,
            ),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        images = (Image.open(source), Image.open(left), Image.open(right))
        for column, (image, title) in enumerate(zip(images, titles)):
            x = column * (args.cell_width + gap)
            draw.rectangle(
                (x, 0, x + args.cell_width, header_height),
                fill=(250, 250, 250),
            )
            bounds = draw.textbbox((0, 0), title, font=label_font)
            text_width = bounds[2] - bounds[0]
            draw.text(
                (x + (args.cell_width - text_width) // 2, 7),
                title,
                font=label_font,
                fill=(30, 30, 30),
            )
            panel.paste(
                fit_tile(image, args.cell_width, args.cell_height),
                (x, header_height),
            )
            draw.rectangle(
                (
                    x,
                    header_height,
                    x + args.cell_width - 1,
                    header_height + args.cell_height - 1,
                ),
                outline=(200, 200, 200),
                width=1,
            )

        destination = output_dir / f"{index:03d}_{source.stem}_panel.png"
        panel.save(destination)
        if index % 20 == 0 or index == len(sources):
            print(f"composed={index}/{len(sources)}", flush=True)

    print(f"RESULT=PASS_PREDICTION_PANELS output={output_dir}")


if __name__ == "__main__":
    main()
