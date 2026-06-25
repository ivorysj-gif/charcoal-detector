from __future__ import annotations

import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50


def build_segmentation_model(num_classes: int = 2) -> nn.Module:
    model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def load_model(checkpoint_path: str, device: str | torch.device = "cpu") -> nn.Module:
    model = build_segmentation_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

