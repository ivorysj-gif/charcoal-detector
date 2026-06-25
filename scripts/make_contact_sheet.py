from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.charcoal_detector.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a raw/mask/prediction contact sheet."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/charcoal_tiny_unet.pt"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contact_sheet.png"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(args.checkpoint), device=device)

    pairs = collect_pairs(args.image_dir, args.mask_dir)
    selected = prioritize_examples(pairs, args.limit)
    rows = [
        make_row(raw_path, mask_path, model, device, args.image_size, args.thumb_size)
        for raw_path, mask_path in selected
    ]

    sheet = stack_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote contact sheet: {args.output}")


def collect_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for raw_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / f"{raw_path.stem}.png"
        if mask_path.exists():
            pairs.append((raw_path, mask_path))
    return pairs


def prioritize_examples(
    pairs: list[tuple[Path, Path]],
    limit: int,
) -> list[tuple[Path, Path]]:
    positive = []
    empty = []
    for raw_path, mask_path in pairs:
        if mask_nonzero(mask_path):
            positive.append((raw_path, mask_path))
        else:
            empty.append((raw_path, mask_path))

    empty_count = min(len(empty), max(1, limit // 4))
    positive_count = max(0, limit - empty_count)
    return positive[:positive_count] + empty[:empty_count]


def mask_nonzero(mask_path: Path) -> bool:
    with Image.open(mask_path).convert("L") as mask:
        return any(pixel != 0 for pixel in mask.getdata())


def make_row(
    raw_path: Path,
    mask_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    thumb_size: int,
) -> Image.Image:
    raw = Image.open(raw_path).convert("RGB")
    truth_mask = Image.open(mask_path).convert("L")
    prediction_mask = predict_mask(raw, model, device, image_size)

    raw_panel = panel(raw, "raw", thumb_size)
    truth_panel = panel(draw_overlay(raw, truth_mask, (60, 110, 255)), "truth", thumb_size)
    pred_panel = panel(draw_overlay(raw, prediction_mask, (255, 80, 40)), "prediction", thumb_size)

    label_width = 260
    row = Image.new("RGB", (label_width + thumb_size * 3, thumb_size + 34), "white")
    draw = ImageDraw.Draw(row)
    draw.text((8, 8), raw_path.stem[:42], fill=(0, 0, 0))
    row.paste(raw_panel, (label_width, 0))
    row.paste(truth_panel, (label_width + thumb_size, 0))
    row.paste(pred_panel, (label_width + thumb_size * 2, 0))
    return row


def predict_mask(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> Image.Image:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor)["out"].argmax(dim=1).squeeze(0).cpu().numpy()
    return Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").resize(
        image.size,
        Image.Resampling.NEAREST,
    )


def draw_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
) -> Image.Image:
    overlay = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (*color, 0))
    alpha = mask.point(lambda value: 120 if value else 0)
    tint.putalpha(alpha)
    overlay.alpha_composite(tint)
    return overlay.convert("RGB")


def panel(image: Image.Image, title: str, thumb_size: int) -> Image.Image:
    resized = image.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
    output = Image.new("RGB", (thumb_size, thumb_size + 34), "white")
    output.paste(resized, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.rectangle([0, 0, thumb_size - 1, thumb_size - 1], outline=(180, 180, 180), width=1)
    draw.text((8, thumb_size + 8), title, fill=(0, 0, 0))
    return output


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("No rows to render.")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


if __name__ == "__main__":
    main()

