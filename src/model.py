from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.config import FREEZE_BACKBONE

from .config import CLASS_NAMES, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    if FREEZE_BACKBONE:
        for parameter in model.parameters():
            parameter.requires_grad = False
        # Keep the newly created classification head trainable.
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return model


def build_model_from_checkpoint(checkpoint, device: torch.device):
    model = build_model(num_classes=len(checkpoint.get("class_names", CLASS_NAMES)), pretrained=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # A small compatibility shim for checkpoints saved from DataParallel.
    cleaned = OrderedDict((k.removeprefix("module."), v) for k, v in state_dict.items())
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model
