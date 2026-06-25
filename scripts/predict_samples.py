from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.charcoal_detector.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quick model predictions.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/charcoal_tiny_unet.pt"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(args.checkpoint), device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() == ".png")
    for image_path in image_paths[: args.limit]:
        original = Image.open(image_path).convert("RGB")
        resized = original.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
        tensor = image_to_tensor(resized).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)["out"]
            charcoal_probability = logits.softmax(dim=1)[:, 1].squeeze(0).cpu().numpy()
            prediction = (charcoal_probability >= args.threshold).astype(np.uint8)

        mask = Image.fromarray(prediction * 255, mode="L").resize(
            original.size,
            Image.Resampling.NEAREST,
        )
        overlay = draw_overlay(original, mask)
        overlay.save(args.output_dir / f"{image_path.stem}_prediction.png")

    print(f"Wrote predictions to {args.output_dir}")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def draw_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    overlay = image.copy().convert("RGBA")
    color = Image.new("RGBA", image.size, (255, 80, 40, 0))
    alpha = mask.point(lambda value: 110 if value else 0)
    color.putalpha(alpha)
    overlay.alpha_composite(color)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, image.width - 1, image.height - 1], outline=(255, 80, 40, 180), width=1)
    return overlay.convert("RGB")


if __name__ == "__main__":
    main()
