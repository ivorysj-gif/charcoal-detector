from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

from .models import build_segmentation_model

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class TrainingConfig:
    image_dir: Path = Path("data/raw")
    mask_dir: Path = Path("data/masks")
    output_path: Path = Path("models/charcoal_tiny_unet.pt")
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-4
    validation_fraction: float = 0.2
    image_size: int = 512


class CharcoalMaskDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int) -> None:
        self.image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )

        image_tensor = self.transforms(image)
        mask_tensor = torch.from_numpy(np.asarray(mask).copy())
        mask_tensor = (mask_tensor > 0).long()
        return image_tensor, mask_tensor


def train(config: TrainingConfig) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    dataset = CharcoalMaskDataset(config.image_dir, config.mask_dir, config.image_size)

    if len(dataset) < 2:
        raise ValueError("Add at least two image/mask pairs before training.")

    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1)

    model = build_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        validation_loss = _evaluate(model, validation_loader, criterion, device)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss / max(1, len(train_loader)):.4f} "
            f"validation_loss={validation_loss:.4f}"
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, config.output_path)
    return config.output_path


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            losses.append(float(criterion(outputs, masks).detach().cpu()))
    return sum(losses) / max(1, len(losses))
