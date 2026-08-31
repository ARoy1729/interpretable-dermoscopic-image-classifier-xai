from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


@dataclass
class Prediction:
    index: int
    label: str
    confidence: float
    probabilities: np.ndarray


class GradCAM:
    """A compact Grad-CAM implementation tied to DenseNet121's final dense block."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.features.denseblock4
        self.activations = None
        self.gradients = None
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        #self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        
        # Grad-CAM needs gradients even when the DenseNet backbone
        # was frozen during training.
        #if self.activations.requires_grad:
        #    self.activations.retain_grad()
        
        # intermediate feature map still needs a gradient for Grad-CAM.
        if output.requires_grad:
            output.retain_grad()

    #def _backward_hook(self, _module, _grad_input, grad_output):
    #    self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # The DenseNet backbone is frozen during CPU training, but Grad-CAM
        # still needs gradients through the feature maps during explanation.
        with torch.enable_grad():
            logits = self.model(input_tensor)
            score = logits[:, class_index].sum()
            score.backward()

        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def close(self):
        self._forward_handle.remove()
        #self._backward_handle.remove()


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.float()


def predict(model, image_tensor: torch.Tensor, class_names, device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())
    return Prediction(
        index=index,
        label=class_names[index],
        confidence=float(probabilities[index].item()),
        probabilities=probabilities.detach().cpu().numpy(),
    )


def _normalise_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = np.maximum(values, 0)
    values -= values.min()
    max_value = values.max()
    return values / (max_value + 1e-8)


def shap_attribution(model, image_tensor: torch.Tensor, class_index: int, background: torch.Tensor, device):
    """Use SHAP expected gradients and return a spatial importance map.

    We explain only the predicted class so the app stays interactive; the same
    uploaded image is passed to both Grad-CAM and SHAP.
    """
    import shap

    class_model = _SingleClassWrapper(model, class_index).to(device).eval()
    background = background.to(device)
    image_tensor = image_tensor.to(device)
    explainer = shap.GradientExplainer(class_model, background)
    shap_values = explainer.shap_values(image_tensor, nsamples=80)

    if isinstance(shap_values, list):
        values = shap_values[0]
    else:
        values = shap_values
    values = np.asarray(values)

    if values.ndim == 5:
        values = values[..., 0]
    # Shape is usually N,C,H,W for PyTorch image explainers.
    if values.ndim == 4:
        spatial = np.mean(np.abs(values[0]), axis=0)
    else:
        raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")
    return _normalise_map(spatial)


class _SingleClassWrapper(torch.nn.Module):
    def __init__(self, model, class_index: int):
        super().__init__()
        self.model = model
        self.class_index = class_index

    def forward(self, x):
        return self.model(x)[:, self.class_index : self.class_index + 1]


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    heatmap_img = heatmap_img.resize(image.size)

    # A pure PIL palette keeps the runtime light and avoids OpenCV just for color mapping.
    palette = []
    for i in range(256):
        x = i / 255.0
        r = int(255 * min(1.0, 2.0 * x))
        g = int(255 * min(1.0, 2.0 * (1 - abs(x - 0.5))))
        b = int(255 * min(1.0, 2.0 * (1 - x)))
        palette.extend([r, g, b])
    colour = heatmap_img.convert("P")
    colour.putpalette(palette + [0] * (768 - len(palette)))
    colour = colour.convert("RGB")
    return Image.blend(image, colour, alpha)
