from pathlib import Path

from src.charcoal_detector.training import TrainingConfig, train


if __name__ == "__main__":
    checkpoint_path = train(
        TrainingConfig(
            image_dir=Path("data/raw"),
            mask_dir=Path("data/masks"),
            output_path=Path("models/charcoal_fcn.pt"),
        )
    )
    print(f"saved checkpoint: {checkpoint_path}")

