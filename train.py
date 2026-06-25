from pathlib import Path
import argparse

from src.charcoal_detector.training import TrainingConfig, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the charcoal segmentation model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    checkpoint_path = train(
        TrainingConfig(
            image_dir=Path("data/raw"),
            mask_dir=Path("data/masks"),
            output_path=Path("models/charcoal_tiny_unet.pt"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
    )
    print(f"saved checkpoint: {checkpoint_path}")
